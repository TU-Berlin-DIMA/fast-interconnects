/*
 * Copyright 2019 Clemens Lutz, German Research Center for Artificial Intelligence
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use super::radix_partition::{fanout, Tuple};
use crate::error::{ErrorKind, Result};
use numa_gpu::runtime::allocator::MemAllocFn;
use numa_gpu::runtime::memory::{LaunchableMutPtr, LaunchablePtr, LaunchableSlice, Mem};
use rustacuda::context::CurrentContext;
use rustacuda::device::DeviceAttribute;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::launch;
use rustacuda::memory::{DeviceBox, DeviceBuffer, DeviceCopy};
use rustacuda::module::Module;
use rustacuda::stream::Stream;
use std::convert::TryInto;
use std::ffi::CString;
use std::mem;
use std::ops::{Index, IndexMut};

// extern "C" {
//     fn gpu_swwc_buffer_bytes() -> usize;
// }

/// Arguments to the C/C++ partitioning function.
///
/// Note that the struct's layout must be kept in sync with its counterpart in
/// C/C++.
#[repr(C)]
#[derive(Debug)]
struct RadixPartitionArgs<T> {
    // Inputs
    partition_attr_data: LaunchablePtr<T>,
    payload_attr_data: LaunchablePtr<T>,
    data_len: usize,
    padding_len: usize,
    radix_bits: u32,

    // State
    tmp_partition_offsets: LaunchableMutPtr<u64>,
    write_combine_buffer: LaunchableMutPtr<T>,

    // Outputs
    partition_offsets: LaunchableMutPtr<u64>,
    partitioned_relation: LaunchableMutPtr<Tuple<T, T>>,
}

unsafe impl<T: DeviceCopy> DeviceCopy for RadixPartitionArgs<T> {}

/// A set of buffers used for software write-combining.
///
/// The original code by Cagri Balkesen allocates these SWWC buffers on the stack.
/// In contrast, this implementation allocates the SWWC buffers on the heap,
/// because new CPUs have very large L3 caches (> 100 MB). Allocating on the
/// stack risks a stack overflow on these CPUs. This can occur when using a large
/// fanout (i.e., a high number of radix bits).
///
/// # Invariants
///
/// * The `radix_bits` must match in `PartitionedRelation` and `GpuRadixPartitioner`.
///
/// * The backing memory must be aligned to the cache-line size of the machine.
///   Hint: `Mem::Numa` alignes to the page size, which is a multiple of the
///   cache-line size
#[derive(Debug)]
struct WriteCombineBuffer {
    buffers: DeviceBuffer<u64>,
}

impl WriteCombineBuffer {
    /// Creates a new set of SWWC buffers.
    fn new(radix_bits: u32) -> Result<Self> {
        // FIXME: Find a DRY way to share buffer size between CUDA and Rust
        // let buffer_bytes = unsafe { gpu_swwc_buffer_bytes() };
        let buffer_bytes = 0_usize;
        let bytes = buffer_bytes * fanout(radix_bits);
        let buffers = unsafe { DeviceBuffer::uninitialized(bytes / mem::size_of::<u64>())? };

        Ok(Self { buffers })
    }

    /// Computes the number of tuples per SWWC buffer.
    ///
    /// Note that `WriteCombineBuffer` contains one SWWC buffer per
    fn tuples_per_buffer<T: Sized>() -> usize {
        // FIXME: Find a DRY way to share buffer size between CUDA and Rust
        // let buffer_bytes = unsafe { gpu_swwc_buffer_bytes() };
        let buffer_bytes = 0_usize;
        buffer_bytes / mem::size_of::<T>()
    }
}

pub trait GpuRadixPartitionable: Sized + DeviceCopy {
    fn partition_impl(
        rp: &mut GpuRadixPartitioner,
        partition_attr: LaunchableSlice<Self>,
        payload_attr: LaunchableSlice<Self>,
        partitioned_relation: &mut PartitionedRelation<Tuple<Self, Self>>,
        stream: &Stream,
    ) -> Result<()>;
}

/// A radix-partitioned relation, optionally with padding in front of each
/// partition.
///
/// # Invariants
///
/// The `radix_bits` must match in `WriteCombineBuffer` and `CpuRadixPartitioner`.
#[derive(Debug)]
pub struct PartitionedRelation<T: DeviceCopy> {
    pub(super) relation: Mem<T>,
    pub(super) offsets: Mem<u64>,
    pub(super) radix_bits: u32,
}

impl<T: DeviceCopy> PartitionedRelation<T> {
    /// Creates a new partitioned relation, and automatically includes the
    /// necessary padding and metadata.
    pub fn new(
        len: usize,
        radix_bits: u32,
        partition_alloc_fn: MemAllocFn<T>,
        offsets_alloc_fn: MemAllocFn<u64>,
    ) -> Self {
        let padding_len = WriteCombineBuffer::tuples_per_buffer::<T>();
        let num_partitions = fanout(radix_bits);
        let relation_len = len + num_partitions * padding_len;

        let relation = partition_alloc_fn(relation_len);
        let offsets = offsets_alloc_fn(num_partitions);

        Self {
            relation,
            offsets,
            radix_bits,
        }
    }

    /// Returns the total number of elements in the relation (excluding padding).
    pub fn len(&self) -> usize {
        let num_partitions = fanout(self.radix_bits);

        self.relation.len() - num_partitions * self.padding_len()
    }

    /// Returns the number of partitions.
    pub fn partitions(&self) -> usize {
        fanout(self.radix_bits)
    }

    /// Returns the number of padding elements per partition.
    pub(super) fn padding_len(&self) -> usize {
        WriteCombineBuffer::tuples_per_buffer::<T>()
    }
}

/// Returns the specified partition as a subslice of the relation.
impl<T: DeviceCopy> Index<usize> for PartitionedRelation<T> {
    type Output = [T];

    fn index(&self, i: usize) -> &Self::Output {
        let (offsets, relation): (&[u64], &[T]) =
            match ((&self.offsets).try_into(), (&self.relation).try_into()) {
                (Ok(offsets), Ok(relation)) => (offsets, relation),
                _ => panic!("Trying to dereference device memory!"),
            };

        let begin = offsets[i] as usize;
        let end = if i + 1 < self.offsets.len() {
            offsets[i + 1] as usize - self.padding_len()
        } else {
            relation.len()
        };

        &relation[begin..end]
    }
}

/// Returns the specified partition as a mutable subslice of the relation.
impl<T: DeviceCopy> IndexMut<usize> for PartitionedRelation<T> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        let padding_len = self.padding_len();
        let (offsets, relation): (&mut [u64], &mut [T]) = match (
            (&mut self.offsets).try_into(),
            (&mut self.relation).try_into(),
        ) {
            (Ok(offsets), Ok(relation)) => (offsets, relation),
            _ => panic!("Trying to dereference device memory!"),
        };

        let begin = offsets[i] as usize;
        let end = if i + 1 < offsets.len() {
            offsets[i + 1] as usize - padding_len
        } else {
            relation.len()
        };

        &mut relation[begin..end]
    }
}

/// Specifies the radix partition algorithm.
#[derive(Copy, Clone, Debug)]
pub enum GpuRadixPartitionAlgorithm {
    /// Chunked radix partition.
    Chunked,
    Block,
}

#[derive(Debug)]
enum RadixPartitionState {
    Chunked(Mem<u64>),
    Block(Mem<u64>),
}

#[derive(Debug)]
pub struct GpuRadixPartitioner {
    radix_bits: u32,
    state: RadixPartitionState,
    module: Module,
    grid_size: GridSize,
    block_size: BlockSize,
}

impl GpuRadixPartitioner {
    /// Creates a new CPU radix partitioner.
    pub fn new(
        algorithm: GpuRadixPartitionAlgorithm,
        radix_bits: u32,
        alloc_fn: MemAllocFn<u64>,
        grid_size: GridSize,
        block_size: BlockSize,
    ) -> Result<Self> {
        let num_partitions = fanout(radix_bits);

        let state = match algorithm {
            GpuRadixPartitionAlgorithm::Chunked => {
                RadixPartitionState::Chunked(alloc_fn(num_partitions))
            }
            GpuRadixPartitionAlgorithm::Block => {
                RadixPartitionState::Block(alloc_fn(num_partitions * block_size.x as usize))
            }
        };

        let module_path = CString::new(env!("CUDAUTILS_PATH")).map_err(|_| {
            ErrorKind::NulCharError(
                "Failed to load CUDA module, check your CUDAUTILS_PATH".to_string(),
            )
        })?;

        let module = Module::load_from_file(&module_path)?;

        Ok(Self {
            radix_bits,
            state,
            module,
            grid_size,
            block_size,
        })
    }

    /// Radix-partitions a relation by its key attribute.
    ///
    /// See the module-level documentation for details on the algorithm.
    pub fn partition<T: DeviceCopy + GpuRadixPartitionable>(
        &mut self,
        partition_attr: LaunchableSlice<T>,
        payload_attr: LaunchableSlice<T>,
        partitioned_relation: &mut PartitionedRelation<Tuple<T, T>>,
        stream: &Stream,
    ) -> Result<()> {
        T::partition_impl(
            self,
            partition_attr,
            payload_attr,
            partitioned_relation,
            stream,
        )
    }
}

macro_rules! impl_gpu_radix_partition_for_type {
    ($Type:ty, $Suffix:expr) => {
        impl GpuRadixPartitionable for $Type {
            paste::item! {
                fn partition_impl(
                    rp: &mut GpuRadixPartitioner,
                    partition_attr: LaunchableSlice<$Type>,
                    payload_attr: LaunchableSlice<$Type>,
                    partitioned_relation: &mut PartitionedRelation<Tuple<$Type, $Type>>,
                    stream: &Stream,
                    ) -> Result<()> {
                    if partition_attr.len() != payload_attr.len() {
                        Err(ErrorKind::InvalidArgument(
                                "Partition and payload attributes have different sizes".to_string(),
                                ))?;
                    }
                    if partitioned_relation.radix_bits != rp.radix_bits {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionedRelation has mismatching radix bits".to_string(),
                                ))?;
                    }

                    let data_len = partition_attr.len();
                    let (tmp_partition_offsets, write_combine_buffer) = match rp.state {
                        RadixPartitionState::Chunked(ref mut offsets) => (
                            offsets.as_launchable_mut_ptr(),
                            LaunchableMutPtr::null_mut(),
                            ),
                            RadixPartitionState::Block(ref mut offsets) => (
                                offsets.as_launchable_mut_ptr(),
                                LaunchableMutPtr::null_mut(),
                                ),
                    };

                    let args = RadixPartitionArgs {
                        partition_attr_data: partition_attr.as_launchable_ptr(),
                        payload_attr_data: payload_attr.as_launchable_ptr(),
                        data_len,
                        padding_len: partitioned_relation.padding_len(),
                        radix_bits: rp.radix_bits,
                        tmp_partition_offsets,
                        write_combine_buffer,
                        partition_offsets: partitioned_relation.offsets.as_launchable_mut_ptr(),
                        partitioned_relation: partitioned_relation.relation.as_launchable_mut_ptr(),
                    };

                    let mut device_args = DeviceBox::new(&args)?;

                    let module = &rp.module;
                    let grid_size = rp.grid_size.clone();
                    let block_size = rp.block_size.clone();

                    match rp.state {
                        RadixPartitionState::Chunked(_) => unsafe {
                            launch!(
                                module.[<gpu_chunked_radix_partition_ $Suffix _ $Suffix>]<<<
                                grid_size,
                                block_size,
                                0,
                                stream
                                >>>(
                                    device_args.as_device_ptr()
                                   ))?;
                        },
                        RadixPartitionState::Block(_) => {
                            let device = CurrentContext::get_device()?;
                            let warp_size = device.get_attribute(DeviceAttribute::WarpSize)? as u32;
                            let max_shared_mem_bytes =
                                device.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)? as u32;
                            let shared_mem_bytes = (block_size.x / warp_size + (fanout(rp.radix_bits) as u32))
                                * mem::size_of::<usize>() as u32;
                            assert!(
                                shared_mem_bytes <= max_shared_mem_bytes,
                                "Failed to allocate enough shared memory"
                                );

                            unsafe {
                                launch!(
                                    module.[<gpu_block_radix_partition_ $Suffix _ $Suffix>]<<<
                                    grid_size,
                                    block_size,
                                    shared_mem_bytes,
                                    stream
                                    >>>(
                                        device_args.as_device_ptr()
                                       ))?;
                            }
                        }
                    }

                    Ok(())
                }
            }
        }
    }
}

impl_gpu_radix_partition_for_type!(i32, int32);
impl_gpu_radix_partition_for_type!(i64, int64);

#[cfg(test)]
mod tests {
    use super::*;
    use datagen::relation::UniformRelation;
    use numa_gpu::runtime::allocator::{Allocator, MemType};
    use rustacuda::function::{BlockSize, GridSize};
    use rustacuda::memory::LockedBuffer;
    use rustacuda::stream::{Stream, StreamFlags};
    use std::collections::hash_map::{Entry, HashMap};
    use std::error::Error;
    use std::iter;
    use std::ops::RangeInclusive;
    use std::result::Result;

    fn gpu_tuple_loss_or_duplicates_i32(
        tuples: usize,
        algorithm: GpuRadixPartitionAlgorithm,
        radix_bits: u32,
        block_size: BlockSize,
    ) -> Result<(), Box<dyn Error>> {
        const PAYLOAD_RANGE: RangeInclusive<usize> = 1..=10000;

        let _context = rustacuda::quick_init()?;

        let mut data_key: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;
        let mut data_pay: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;

        UniformRelation::gen_primary_key(&mut data_key)?;
        UniformRelation::gen_attr(&mut data_pay, PAYLOAD_RANGE)?;

        let mut original_tuples: HashMap<_, _> = data_key
            .iter()
            .cloned()
            .zip(data_pay.iter().cloned().zip(std::iter::repeat(0)))
            .collect();

        let mut partitioned_relation = PartitionedRelation::new(
            tuples,
            radix_bits,
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
        );

        let mut partitioner = GpuRadixPartitioner::new(
            algorithm,
            radix_bits,
            Allocator::mem_alloc_fn::<u64>(MemType::CudaUniMem),
            GridSize::from(10),
            BlockSize::from(128),
        )?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let data_key = Mem::CudaPinnedMem(data_key);
        let data_pay = Mem::CudaPinnedMem(data_pay);

        partitioner.partition(
            data_key.as_launchable_slice(),
            data_pay.as_launchable_slice(),
            &mut partitioned_relation,
            &stream,
        )?;

        stream.synchronize()?;

        let relation: &[_] = (&partitioned_relation.relation)
            .try_into()
            .expect("Tried to convert device memory into host slice");

        relation.iter().cloned().for_each(|Tuple { key, value }| {
            let entry = original_tuples.entry(key);
            match entry {
                entry @ Entry::Occupied(_) => {
                    let key = *entry.key();
                    entry.and_modify(|(original_value, counter)| {
                        assert_eq!(
                            value, *original_value,
                            "Invalid payload: {}; expected: {}",
                            value, *original_value
                        );
                        assert_eq!(*counter, 0, "Duplicate key: {}", key);
                        *counter = *counter + 1;
                    });
                }
                entry @ Entry::Vacant(_) => {
                    // skip padding entries
                    if *entry.key() != 0 {
                        assert!(false, "Invalid key: {}", entry.key());
                    }
                }
            };
        });

        original_tuples.iter().for_each(|(&key, &(_, counter))| {
            assert_eq!(
                counter, 1,
                "Key {} occurs {} times; expected exactly once",
                key, counter
            );
        });

        Ok(())
    }

    fn gpu_verify_partitions_i32(
        tuples: usize,
        key_range: RangeInclusive<usize>,
        algorithm: GpuRadixPartitionAlgorithm,
        radix_bits: u32,
        block_size: BlockSize,
    ) -> Result<(), Box<dyn Error>> {
        const PAYLOAD_RANGE: RangeInclusive<usize> = 1..=10000;

        let _context = rustacuda::quick_init()?;

        let mut data_key: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;
        let mut data_pay: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;

        UniformRelation::gen_attr(&mut data_key, key_range)?;
        UniformRelation::gen_attr(&mut data_pay, PAYLOAD_RANGE)?;

        let mut partitioned_relation = PartitionedRelation::new(
            tuples,
            radix_bits,
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
        );

        let mut partitioner = GpuRadixPartitioner::new(
            algorithm,
            radix_bits,
            Allocator::mem_alloc_fn::<u64>(MemType::CudaUniMem),
            GridSize::from(1),
            block_size,
        )?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let data_key = Mem::CudaPinnedMem(data_key);
        let data_pay = Mem::CudaPinnedMem(data_pay);

        partitioner.partition(
            data_key.as_launchable_slice(),
            data_pay.as_launchable_slice(),
            &mut partitioned_relation,
            &stream,
        )?;

        stream.synchronize()?;

        let mask = fanout(radix_bits) - 1;
        (0..partitioned_relation.partitions())
            .flat_map(|i| iter::repeat(i).zip(partitioned_relation[i].iter()))
            .for_each(|(i, &tuple)| {
                let dst_partition = (tuple.key) & mask as i32;
                assert_eq!(
                    dst_partition, i as i32,
                    "Wrong partitioning detected: key {} in partition {}; expected partition {}",
                    tuple.key, i, dst_partition
                );
            });

        Ok(())
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_block_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            32 << 20 / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::Block,
            10,
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_block_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            32 << 20 / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::Block,
            10,
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_block_i32_12_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            32 << 20 / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::Block,
            12,
            BlockSize::from(1024),
        )
    }

    #[test]
    fn gpu_verify_partitions_block_i32_12_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            32 << 20 / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::Block,
            12,
            BlockSize::from(1024),
        )
    }
}
