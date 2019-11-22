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
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::launch;
use rustacuda::memory::{DeviceBox, DeviceBuffer, DeviceCopy};
use rustacuda::module::Module;
use rustacuda::stream::Stream;
use std::ffi::CString;
use std::mem;

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

/// Specifies the radix partition algorithm.
#[derive(Copy, Clone, Debug)]
pub enum GpuRadixPartitionAlgorithm {
    /// Chunked radix partition.
    Chunked,
}

#[derive(Debug)]
enum RadixPartitionState {
    Chunked(Mem<u64>),
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

impl GpuRadixPartitionable for i32 {
    fn partition_impl(
        rp: &mut GpuRadixPartitioner,
        partition_attr: LaunchableSlice<i32>,
        payload_attr: LaunchableSlice<i32>,
        partitioned_relation: &mut PartitionedRelation<Tuple<i32, i32>>,
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

        unsafe {
            launch!(
            module.gpu_chunked_radix_partition_int32_int32<<<grid_size, block_size, 0, stream>>>(
                device_args.as_device_ptr()
                )
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::result::Result;
    use std::error::Error;
    use datagen::relation::UniformRelation;
    use numa_gpu::runtime::allocator::{Allocator, MemType};
    use std::collections::hash_map::{Entry, HashMap};
    use std::iter;
    use std::mem::size_of;
    use std::ops::RangeInclusive;
    use rustacuda::function::{GridSize, BlockSize};
    use rustacuda::stream::{Stream, StreamFlags};
    use rustacuda::memory::LockedBuffer;

    #[test]
    fn gpu_verify_partitions() -> Result<(), Box<dyn Error>> {
        const KEY_RANGE: RangeInclusive<usize> = 1..=10000;
        const PAYLOAD_RANGE: RangeInclusive<usize> = 1..=10000;
        const NUMA_NODE: u16 = 0;
        const TUPLES: usize = 32 << 20;
        const RADIX_BITS: u32 = 12;
        const ALGORITHM: GpuRadixPartitionAlgorithm = GpuRadixPartitionAlgorithm::Chunked;

        let _context = rustacuda::quick_init()?;

        let mut data_key: LockedBuffer<i32> = LockedBuffer::new(&0, TUPLES)?;
        let mut data_pay: LockedBuffer<i32> = LockedBuffer::new(&0, TUPLES)?;

        UniformRelation::gen_attr(&mut data_key, KEY_RANGE)?;
        UniformRelation::gen_attr(&mut data_pay, PAYLOAD_RANGE)?;

        let mut partitioned_relation = PartitionedRelation::new(
            TUPLES,
            RADIX_BITS,
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
            );

        let mut partitioner = GpuRadixPartitioner::new(
            ALGORITHM,
            RADIX_BITS,
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

        // let mask = fanout(RADIX_BITS) - 1;
        // (0..partitioned_relation.partitions())
        //     .flat_map(|i| {
        //         iter::repeat(i)
        //             .zip(partitioned_relation[i].iter())
        //     })
        // .for_each(|(i, &tuple)| {
        //     let dst_partition = (tuple.key) & mask as i32;
        //     assert_eq!(
        //         dst_partition, i as i32,
        //         "Wrong partitioning detected: key {} in partition {}; expected partition {}",
        //         tuple.key, i, dst_partition
        //         );
        // });

        Ok(())
    }
}
