/*
 * Copyright 2019-2020 Clemens Lutz, German Research Center for Artificial Intelligence
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

use super::{fanout, Tuple};
use crate::error::{ErrorKind, Result};
use crate::prefix_scan::{GpuPrefixScanState, GpuPrefixSum};
use numa_gpu::runtime::allocator::{Allocator, MemAllocFn, MemType};
use numa_gpu::runtime::memory::{LaunchableMutPtr, LaunchablePtr, LaunchableSlice, Mem};
use rustacuda::context::CurrentContext;
use rustacuda::device::DeviceAttribute;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::{DeviceBox, DeviceBuffer, DeviceCopy};
use rustacuda::module::Module;
use rustacuda::stream::Stream;
use rustacuda::{launch, launch_cooperative};
use std::convert::TryInto;
use std::ffi::CString;
use std::mem;
use std::ops::{Index, IndexMut};

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
    padding_len: u32,
    radix_bits: u32,

    // State
    prefix_scan_state: LaunchableMutPtr<GpuPrefixScanState<u32>>,
    tmp_partition_offsets: LaunchableMutPtr<u32>,
    device_memory_buffers: LaunchableMutPtr<i8>,
    device_memory_buffer_bytes: u64,

    // Outputs
    partition_offsets: LaunchableMutPtr<u64>,
    partitioned_relation: LaunchableMutPtr<Tuple<T, T>>,
}

unsafe impl<T: DeviceCopy> DeviceCopy for RadixPartitionArgs<T> {}

pub trait GpuRadixPartitionable: Sized + DeviceCopy {
    fn partition_impl(
        rp: &mut GpuRadixPartitioner,
        partition_attr: LaunchableSlice<'_, Self>,
        payload_attr: LaunchableSlice<'_, Self>,
        partitioned_relation: &mut PartitionedRelation<Tuple<Self, Self>>,
        stream: &Stream,
    ) -> Result<()>;
}

/// A radix-partitioned relation, optionally with padding in front of each
/// partition.
///
/// The relation supports chunking on a single GPU. E.g. in the `Chunked`
/// algorithm, there is a chunk per thread block. In this case, `chunks` should
/// equal the grid size.
///
/// # Invariants
///
///  - `radix_bits` must match in `GpuRadixPartitioner`.
///  - `chunks` must equal the actual number of chunks computed at runtime
///     (e.g., the grid size).
#[derive(Debug)]
pub struct PartitionedRelation<T: DeviceCopy> {
    pub(super) relation: Mem<T>,
    pub(super) offsets: Mem<u64>,
    pub(super) chunks: u32,
    pub(super) radix_bits: u32,
}

impl<T: DeviceCopy> PartitionedRelation<T> {
    /// Creates a new partitioned relation, and automatically includes the
    /// necessary padding and metadata.
    pub fn new(
        len: usize,
        algorithm: GpuRadixPartitionAlgorithm,
        radix_bits: u32,
        grid_size: &GridSize,
        partition_alloc_fn: MemAllocFn<T>,
        offsets_alloc_fn: MemAllocFn<u64>,
    ) -> Self {
        let chunks: u32 = match algorithm {
            GpuRadixPartitionAlgorithm::Chunked
            | GpuRadixPartitionAlgorithm::ChunkedLASWWC
            | GpuRadixPartitionAlgorithm::ChunkedSSWWC
            | GpuRadixPartitionAlgorithm::ChunkedSSWWCNT
            | GpuRadixPartitionAlgorithm::ChunkedSSWWCv2
            | GpuRadixPartitionAlgorithm::ChunkedHSSWWC
            | GpuRadixPartitionAlgorithm::ChunkedHSSWWCv2
            | GpuRadixPartitionAlgorithm::ChunkedHSSWWCv3 => grid_size.x,
            GpuRadixPartitionAlgorithm::Contiguous => 1,
        };

        let padding_len = 0;
        let num_partitions = fanout(radix_bits);
        let relation_len = len + (num_partitions * chunks as usize) * padding_len;

        let relation = partition_alloc_fn(relation_len);
        let offsets = offsets_alloc_fn(num_partitions * chunks as usize);

        Self {
            relation,
            offsets,
            chunks,
            radix_bits,
        }
    }

    /// Returns the total number of elements in the relation (excluding padding).
    pub fn len(&self) -> usize {
        let num_partitions = fanout(self.radix_bits);

        self.relation.len()
            - (num_partitions * self.chunks() as usize) * self.padding_len() as usize
    }

    /// Returs the number of chunks.
    pub fn chunks(&self) -> u32 {
        self.chunks
    }

    /// Returns the number of partitions.
    pub fn partitions(&self) -> usize {
        fanout(self.radix_bits)
    }

    /// Returns the number of padding elements per partition.
    pub(super) fn padding_len(&self) -> u32 {
        0
    }
}

/// Returns the specified chunk and partition as a subslice of the relation.
impl<T: DeviceCopy> Index<(usize, usize)> for PartitionedRelation<T> {
    type Output = [T];

    fn index(&self, i: (usize, usize)) -> &Self::Output {
        let (offsets, relation): (&[u64], &[T]) =
            match ((&self.offsets).try_into(), (&self.relation).try_into()) {
                (Ok(offsets), Ok(relation)) => (offsets, relation),
                _ => panic!("Trying to dereference device memory!"),
            };

        let ofi = i.0 * self.partitions() + i.1;
        let begin = offsets[ofi] as usize;
        let end = if ofi + 1 < self.offsets.len() {
            offsets[ofi + 1] as usize - self.padding_len() as usize
        } else {
            relation.len()
        };

        &relation[begin..end]
    }
}

/// Returns the specified chunk and partition as a mutable subslice of the
/// relation.
impl<T: DeviceCopy> IndexMut<(usize, usize)> for PartitionedRelation<T> {
    fn index_mut(&mut self, i: (usize, usize)) -> &mut Self::Output {
        let padding_len = self.padding_len();
        let offsets_len = self.offsets.len();
        let partitions = self.partitions();

        let (offsets, relation): (&mut [u64], &mut [T]) = match (
            (&mut self.offsets).try_into(),
            (&mut self.relation).try_into(),
        ) {
            (Ok(offsets), Ok(relation)) => (offsets, relation),
            _ => panic!("Trying to dereference device memory!"),
        };

        let ofi = i.0 * partitions + i.1;
        let begin = offsets[ofi] as usize;
        let end = if ofi + 1 < offsets_len {
            offsets[ofi + 1] as usize - padding_len as usize
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

    /// Chunked radix partitioning with look-ahead software write combining.
    ChunkedLASWWC,

    /// Chunked radix partitioning with shared software write combining.
    ChunkedSSWWC,

    /// Chunked radix partitioning with shared software write combining and non-temporal
    /// loads/stores.
    ChunkedSSWWCNT,

    /// Chunked radix partitioning with shared software write combining, version 2.
    ///
    /// In version 1, a warp can block all other warps by holding locks on more
    /// than one buffer (i.e., leader candidates).
    ///
    /// Version 2 tries to avoid blocking other warps by releasing all locks except
    /// one (i.e., the leader's buffer lock).
    ChunkedSSWWCv2,

    /// Chunked radix partitioning with hierarchical shared software write combining.
    ChunkedHSSWWC,

    /// Chunked radix partitioning with hierarchical shared software write combining, version 2.
    ///
    /// In version 1, a warp can block all other warps by holding locks on more
    /// than one buffer (i.e., leader candidates).
    ///
    /// Version 2 tries to avoid blocking other warps by releasing all locks except
    /// one (i.e., the leader's buffer lock).
    ChunkedHSSWWCv2,

    /// Chunked radix partitioning with hierarchical shared software write combining, version 3.
    ///
    /// Version 3 performs the buffer flush from dmem to memory asynchronously.
    /// This change enables other warps to make progress during the dmem flush, which
    /// is important because the dmem buffer is large (several MBs) and the flush can
    /// take a long time.
    ChunkedHSSWWCv3,

    /// Contiguous radix partition.
    ///
    /// This is the "normal" parallel radix partition algorithm. Each resulting
    /// partition is laid out contiguously in memory.
    Contiguous,
}

#[derive(Debug)]
enum RadixPartitionState {
    Chunked,
    ChunkedLASWWC,
    ChunkedSSWWC(bool),
    ChunkedSSWWCv2,
    ChunkedHSSWWC,
    ChunkedHSSWWCv2,
    ChunkedHSSWWCv3,
    Contiguous(Mem<GpuPrefixScanState<u32>>, Mem<u32>),
}

#[derive(Debug)]
pub struct GpuRadixPartitioner {
    radix_bits: u32,
    log2_num_banks: u32,
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
        _alloc_fn: MemAllocFn<u64>,
        grid_size: &GridSize,
        block_size: &BlockSize,
    ) -> Result<Self> {
        let num_partitions = fanout(radix_bits);
        let log2_num_banks = env!("LOG2_NUM_BANKS")
            .parse::<u32>()
            .expect("Failed to parse \"log2_num_banks\" string to an integer");
        let prefix_scan_state_len = GpuPrefixSum::state_len(grid_size.clone(), block_size.clone())?;
        let tmp_partition_offsets_len =
            num_partitions as usize * grid_size.x as usize * block_size.x as usize;

        let state = match algorithm {
            GpuRadixPartitionAlgorithm::Chunked => RadixPartitionState::Chunked,
            GpuRadixPartitionAlgorithm::ChunkedLASWWC => RadixPartitionState::ChunkedLASWWC,
            GpuRadixPartitionAlgorithm::ChunkedSSWWC => RadixPartitionState::ChunkedSSWWC(false),
            GpuRadixPartitionAlgorithm::ChunkedSSWWCNT => RadixPartitionState::ChunkedSSWWC(true),
            GpuRadixPartitionAlgorithm::ChunkedSSWWCv2 => RadixPartitionState::ChunkedSSWWCv2,
            GpuRadixPartitionAlgorithm::ChunkedHSSWWC => RadixPartitionState::ChunkedHSSWWC,
            GpuRadixPartitionAlgorithm::ChunkedHSSWWCv2 => RadixPartitionState::ChunkedHSSWWCv2,
            GpuRadixPartitionAlgorithm::ChunkedHSSWWCv3 => RadixPartitionState::ChunkedHSSWWCv3,
            GpuRadixPartitionAlgorithm::Contiguous => RadixPartitionState::Contiguous(
                Allocator::alloc_mem(MemType::CudaDevMem, prefix_scan_state_len),
                Allocator::alloc_mem(MemType::CudaDevMem, tmp_partition_offsets_len),
            ),
        };

        let module_path = CString::new(env!("CUDAUTILS_PATH")).map_err(|_| {
            ErrorKind::NulCharError(
                "Failed to load CUDA module, check your CUDAUTILS_PATH".to_string(),
            )
        })?;

        let module = Module::load_from_file(&module_path)?;

        Ok(Self {
            radix_bits,
            log2_num_banks,
            state,
            module,
            grid_size: grid_size.clone(),
            block_size: block_size.clone(),
        })
    }

    /// Radix-partitions a relation by its key attribute.
    ///
    /// See the module-level documentation for details on the algorithm.
    pub fn partition<T: DeviceCopy + GpuRadixPartitionable>(
        &mut self,
        partition_attr: LaunchableSlice<'_, T>,
        payload_attr: LaunchableSlice<'_, T>,
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
                    partition_attr: LaunchableSlice<'_, $Type>,
                    payload_attr: LaunchableSlice<'_, $Type>,
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
                    match rp.state {
                        RadixPartitionState::Chunked
                            | RadixPartitionState::ChunkedLASWWC
                            | RadixPartitionState::ChunkedSSWWC(_)
                            | RadixPartitionState::ChunkedSSWWCv2
                            | RadixPartitionState::ChunkedHSSWWC
                            | RadixPartitionState::ChunkedHSSWWCv2
                            | RadixPartitionState::ChunkedHSSWWCv3
                            => if partitioned_relation.chunks() != rp.grid_size.x {
                                Err(ErrorKind::InvalidArgument(
                                        "PartitionedRelation has mismatching number of chunks".to_string(),
                                        ))?;
                            },
                        RadixPartitionState::Contiguous(_, _)
                            => if partitioned_relation.chunks() != 1 {
                                Err(ErrorKind::InvalidArgument(
                                        "PartitionedRelation has mismatching number of chunks".to_string(),
                                        ))?;
                            }
                    }

                    let data_len = partition_attr.len();
                    let (prefix_scan_state, tmp_partition_offsets) = match rp.state {
                        RadixPartitionState::Chunked
                            | RadixPartitionState::ChunkedLASWWC
                            | RadixPartitionState::ChunkedSSWWC(_)
                            | RadixPartitionState::ChunkedSSWWCv2
                            | RadixPartitionState::ChunkedHSSWWC
                            | RadixPartitionState::ChunkedHSSWWCv2
                            | RadixPartitionState::ChunkedHSSWWCv3
                             => (LaunchableMutPtr::null_mut() ,LaunchableMutPtr::null_mut()),
                        RadixPartitionState::Contiguous(ref mut prefix_scan_state, ref mut offsets)
                            => (prefix_scan_state.as_launchable_mut_ptr(), offsets.as_launchable_mut_ptr()),
                    };

                    let mut args = RadixPartitionArgs {
                        partition_attr_data: partition_attr.as_launchable_ptr(),
                        payload_attr_data: payload_attr.as_launchable_ptr(),
                        data_len,
                        padding_len: partitioned_relation.padding_len(),
                        radix_bits: rp.radix_bits,
                        prefix_scan_state,
                        tmp_partition_offsets,
                        device_memory_buffers: LaunchableMutPtr::null_mut(),
                        device_memory_buffer_bytes: 0,
                        partition_offsets: partitioned_relation.offsets.as_launchable_mut_ptr(),
                        partitioned_relation: partitioned_relation.relation.as_launchable_mut_ptr(),
                    };

                    let module = &rp.module;
                    let grid_size = rp.grid_size.clone();
                    let block_size = rp.block_size.clone();
                    let device = CurrentContext::get_device()?;
                    let _warp_size = device.get_attribute(DeviceAttribute::WarpSize)? as u32;
                    let max_shared_mem_bytes =
                        device.get_attribute(DeviceAttribute::MaxSharedMemPerBlockOptin)? as u32;
                    let fanout_u32 = fanout(rp.radix_bits) as u32;

                    match rp.state {
                        RadixPartitionState::Chunked => {
                            let shared_mem_bytes = (
                                (block_size.x + (block_size.x >> rp.log2_num_banks)) + fanout_u32
                                ) * mem::size_of::<u32>() as u32;
                            assert!(
                                shared_mem_bytes <= max_shared_mem_bytes,
                                "Failed to allocate enough shared memory"
                                );

                            let mut device_args = DeviceBox::new(&args)?;

                            unsafe {
                                launch!(
                                    module.[<gpu_chunked_radix_partition_ $Suffix _ $Suffix>]<<<
                                    grid_size,
                                    block_size,
                                    shared_mem_bytes,
                                    stream
                                    >>>(
                                        device_args.as_device_ptr()
                                       ))?;
                            }
                        },
                        RadixPartitionState::ChunkedLASWWC => {
                            let tuples_per_thread = 5;
                            let shared_mem_bytes =
                                (2 * tuples_per_thread * block_size.x * mem::size_of::<$Type>() as u32)
                                + (fanout_u32 * mem::size_of::<u32>() as u32)
                                + (std::cmp::max(
                                        block_size.x + (block_size.x >> rp.log2_num_banks),
                                        fanout_u32
                                        ) * mem::size_of::<u32>() as u32
                                  );

                            assert!(
                                shared_mem_bytes <= max_shared_mem_bytes,
                                "Failed to allocate enough shared memory"
                                );

                            let name = std::ffi::CString::new(
                                stringify!([<gpu_chunked_laswwc_radix_partition_ $Suffix _ $Suffix>])
                                ).unwrap();
                            let mut function = module.get_function(&name)?;
                            function.set_max_dynamic_shared_size_bytes(max_shared_mem_bytes)?;

                            let mut device_args = DeviceBox::new(&args)?;

                            unsafe {
                                launch!(
                                    function<<<
                                    grid_size,
                                    block_size,
                                    shared_mem_bytes,
                                    stream
                                    >>>(
                                        device_args.as_device_ptr()
                                       ))?;
                            }
                        },
                        RadixPartitionState::ChunkedSSWWC(non_temporal) => {
                            let sswwc_buffer_len =
                                (max_shared_mem_bytes - (
                                    2 * fanout_u32 * mem::size_of::<u32>() as u32
                                )) / (2 * mem::size_of::<$Type>() as u32);

                            assert!(
                                fanout_u32 <= sswwc_buffer_len,
                                "Failed to allocate enough shared memory"
                                );

                            let name = match non_temporal {
                              false => std::ffi::CString::new(
                                    stringify!([<gpu_chunked_sswwc_radix_partition_ $Suffix _ $Suffix>])
                                    ).unwrap(),
                              true => std::ffi::CString::new(
                                    stringify!([<gpu_chunked_sswwc_non_temporal_radix_partition_ $Suffix _ $Suffix>])
                                    ).unwrap(),
                            };
                            let mut function = module.get_function(&name)?;
                            function.set_max_dynamic_shared_size_bytes(max_shared_mem_bytes)?;

                            let mut device_args = DeviceBox::new(&args)?;

                            unsafe {
                                launch!(
                                    function<<<
                                    grid_size,
                                    block_size,
                                    max_shared_mem_bytes,
                                    stream
                                    >>>(
                                        device_args.as_device_ptr(),
                                        max_shared_mem_bytes
                                       ))?;
                            }
                        },
                        RadixPartitionState::ChunkedSSWWCv2 => {
                            let sswwc_buffer_len =
                                (max_shared_mem_bytes - (
                                    2 * fanout_u32 * mem::size_of::<u32>() as u32
                                )) / (2 * mem::size_of::<$Type>() as u32);

                            assert!(
                                fanout_u32 <= sswwc_buffer_len,
                                "Failed to allocate enough shared memory"
                                );

                            let name = std::ffi::CString::new(
                                stringify!([<gpu_chunked_sswwc_radix_partition_v2_ $Suffix _ $Suffix>])
                                ).unwrap();
                            let mut function = module.get_function(&name)?;
                            function.set_max_dynamic_shared_size_bytes(max_shared_mem_bytes)?;

                            let mut device_args = DeviceBox::new(&args)?;

                            unsafe {
                                launch!(
                                    function<<<
                                    grid_size,
                                    block_size,
                                    max_shared_mem_bytes,
                                    stream
                                    >>>(
                                        device_args.as_device_ptr(),
                                        max_shared_mem_bytes
                                       ))?;
                            }
                        },
                        RadixPartitionState::ChunkedHSSWWC => {
                            let dmem_buffer_bytes: u64 = 2 * 1024 * 1024;
                            let global_dmem_buffer_bytes = dmem_buffer_bytes * grid_size.x as u64;
                            let sswwc_buffer_bytes = max_shared_mem_bytes - (
                                    2 * fanout_u32 * mem::size_of::<u32>() as u32
                                );
                            let sswwc_buffer_len =
                                sswwc_buffer_bytes / (2 * mem::size_of::<$Type>() as u32);

                            let tuples_per_buffer = sswwc_buffer_len.next_power_of_two() / 2 / fanout_u32;
                            let tuples_per_dmem_buffer = dmem_buffer_bytes / (fanout as u64) / (2 * mem::size_of::<$Type>() as u64);

                            assert!(
                                tuples_per_buffer >= 1,
                                "Failed to allocate enough shared memory"
                                );
                            assert!(
                                tuples_per_dmem_buffer % (tuples_per_buffer as u64) == 0,
                                "Invalid device memory buffer size! Device memory buffer must be a multiple of shared memory buffer."
                                );

                            let name = std::ffi::CString::new(
                                stringify!([<gpu_chunked_hsswwc_radix_partition_ $Suffix _ $Suffix>])
                                ).unwrap();
                            let mut function = module.get_function(&name)?;
                            function.set_max_dynamic_shared_size_bytes(max_shared_mem_bytes)?;

                            let mut dmem_buffer = Mem::CudaDevMem(unsafe { DeviceBuffer::uninitialized(global_dmem_buffer_bytes as usize)? });
                            args.device_memory_buffers = dmem_buffer.as_launchable_mut_ptr();
                            args.device_memory_buffer_bytes = dmem_buffer_bytes;
                            let mut device_args = DeviceBox::new(&args)?;

                            unsafe {
                                launch!(
                                    function<<<
                                    grid_size,
                                    block_size,
                                    max_shared_mem_bytes,
                                    stream
                                    >>>(
                                        device_args.as_device_ptr(),
                                        max_shared_mem_bytes
                                       ))?;
                            }
                        },
                        RadixPartitionState::ChunkedHSSWWCv2 => {
                            let dmem_buffer_bytes: u64 = 2 * 1024 * 1024;
                            let global_dmem_buffer_bytes = dmem_buffer_bytes * grid_size.x as u64;
                            let sswwc_buffer_bytes = max_shared_mem_bytes - (
                                    2 * fanout_u32 * mem::size_of::<u32>() as u32
                                );
                            let sswwc_buffer_len =
                                sswwc_buffer_bytes / (2 * mem::size_of::<$Type>() as u32);

                            let tuples_per_buffer = sswwc_buffer_len.next_power_of_two() / 2 / fanout_u32;
                            let tuples_per_dmem_buffer = dmem_buffer_bytes / (fanout as u64) / (2 * mem::size_of::<$Type>() as u64);

                            assert!(
                                tuples_per_buffer >= 1,
                                "Failed to allocate enough shared memory"
                                );
                            assert!(
                                tuples_per_dmem_buffer % (tuples_per_buffer as u64) == 0,
                                "Invalid device memory buffer size! Device memory buffer must be a multiple of shared memory buffer."
                                );

                            let name = std::ffi::CString::new(
                                stringify!([<gpu_chunked_hsswwc_radix_partition_v2_ $Suffix _ $Suffix>])
                                ).unwrap();
                            let mut function = module.get_function(&name)?;
                            function.set_max_dynamic_shared_size_bytes(max_shared_mem_bytes)?;

                            let mut dmem_buffer = Mem::CudaDevMem(unsafe { DeviceBuffer::uninitialized(global_dmem_buffer_bytes as usize)? });
                            args.device_memory_buffers = dmem_buffer.as_launchable_mut_ptr();
                            args.device_memory_buffer_bytes = dmem_buffer_bytes;
                            let mut device_args = DeviceBox::new(&args)?;

                            unsafe {
                                launch!(
                                    function<<<
                                    grid_size,
                                    block_size,
                                    max_shared_mem_bytes,
                                    stream
                                    >>>(
                                        device_args.as_device_ptr(),
                                        max_shared_mem_bytes
                                       ))?;
                            }
                        },
                        RadixPartitionState::ChunkedHSSWWCv3 => {
                            let dmem_buffer_bytes: u64 = 2 * 1024 * 1024;
                            let global_dmem_buffer_bytes = dmem_buffer_bytes * grid_size.x as u64;
                            let sswwc_buffer_bytes = max_shared_mem_bytes - (
                                    3 * fanout_u32 * mem::size_of::<u32>() as u32
                                );
                            let sswwc_buffer_len =
                                sswwc_buffer_bytes / (2 * mem::size_of::<$Type>() as u32);

                            let tuples_per_buffer = sswwc_buffer_len.next_power_of_two() / 2 / fanout_u32;
                            let tuples_per_dmem_buffer = dmem_buffer_bytes / (fanout as u64) / (2 * mem::size_of::<$Type>() as u64);

                            assert!(
                                tuples_per_buffer >= 1,
                                "Failed to allocate enough shared memory"
                                );
                            assert!(
                                tuples_per_dmem_buffer % (tuples_per_buffer as u64) == 0,
                                "Invalid device memory buffer size! Device memory buffer must be a multiple of shared memory buffer."
                                );

                            let name = std::ffi::CString::new(
                                stringify!([<gpu_chunked_hsswwc_radix_partition_v3_ $Suffix _ $Suffix>])
                                ).unwrap();
                            let mut function = module.get_function(&name)?;
                            function.set_max_dynamic_shared_size_bytes(max_shared_mem_bytes)?;

                            let mut dmem_buffer = Mem::CudaDevMem(unsafe { DeviceBuffer::uninitialized(global_dmem_buffer_bytes as usize)? });
                            args.device_memory_buffers = dmem_buffer.as_launchable_mut_ptr();
                            args.device_memory_buffer_bytes = dmem_buffer_bytes;
                            let mut device_args = DeviceBox::new(&args)?;

                            unsafe {
                                launch!(
                                    function<<<
                                    grid_size,
                                    block_size,
                                    max_shared_mem_bytes,
                                    stream
                                    >>>(
                                        device_args.as_device_ptr(),
                                        max_shared_mem_bytes
                                       ))?;
                            }
                        }
                        RadixPartitionState::Contiguous(_, _) => {
                            let shared_mem_bytes = (
                                (block_size.x + (block_size.x >> rp.log2_num_banks)) + fanout_u32
                                ) * mem::size_of::<u32>() as u32;
                            assert!(
                                shared_mem_bytes <= max_shared_mem_bytes,
                                "Failed to allocate enough shared memory"
                                );

                            let mut device_args = DeviceBox::new(&args)?;

                            unsafe {
                                launch_cooperative!(
                                    module.[<gpu_contiguous_radix_partition_ $Suffix _ $Suffix>]<<<
                                    grid_size,
                                    block_size,
                                    shared_mem_bytes,
                                    stream
                                    >>>(
                                        device_args.as_device_ptr()
                                       ))?;
                            }
                        },

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
        grid_size: GridSize,
        block_size: BlockSize,
    ) -> Result<(), Box<dyn Error>> {
        const PAYLOAD_RANGE: RangeInclusive<usize> = 1..=10000;
        let _context = rustacuda::quick_init()?;

        let mut data_key: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;
        let mut data_pay: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;

        UniformRelation::gen_primary_key(&mut data_key, None)?;
        UniformRelation::gen_attr(&mut data_pay, PAYLOAD_RANGE)?;

        let mut original_tuples: HashMap<_, _> = data_key
            .iter()
            .cloned()
            .zip(data_pay.iter().cloned().zip(std::iter::repeat(0)))
            .collect();

        let mut partitioned_relation = PartitionedRelation::new(
            tuples,
            algorithm,
            radix_bits,
            &grid_size,
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
        );

        let mut partitioner = GpuRadixPartitioner::new(
            algorithm,
            radix_bits,
            Allocator::mem_alloc_fn::<u64>(MemType::CudaUniMem),
            &grid_size,
            &block_size,
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
        grid_size: GridSize,
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
            algorithm,
            radix_bits,
            &grid_size,
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
        );

        let mut partitioner = GpuRadixPartitioner::new(
            algorithm,
            radix_bits,
            Allocator::mem_alloc_fn::<u64>(MemType::CudaUniMem),
            &grid_size,
            &block_size,
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
        (0..partitioned_relation.chunks())
            .flat_map(|c| iter::repeat(c).zip(0..partitioned_relation.partitions()))
            .flat_map(|(c, p)| iter::repeat((c, p)).zip(partitioned_relation[(c as usize, p)].iter()))
            .for_each(|((c, p), &tuple)| {
                let dst_partition = (tuple.key) as usize & mask;
                assert_eq!(
                    dst_partition, p,
                    "Wrong partitioning detected in chunk {}: key {} in partition {}; expected partition {}",
                    c, tuple.key, p, dst_partition
                );
            });

        Ok(())
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::Chunked,
            2,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::Chunked,
            10,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::Chunked,
            2,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::Chunked,
            10,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_i32_12_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::Chunked,
            12,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_i32_12_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::Chunked,
            12,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            10_usize.pow(6),
            GpuRadixPartitionAlgorithm::Chunked,
            10,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            10_usize.pow(6),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::Chunked,
            10,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_laswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::ChunkedLASWWC,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_laswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::ChunkedLASWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_laswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedLASWWC,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_laswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedLASWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_laswwc_i32_12_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::ChunkedLASWWC,
            12,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_laswwc_i32_12_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedLASWWC,
            12,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_laswwc_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            10_usize.pow(6),
            GpuRadixPartitionAlgorithm::ChunkedLASWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_laswwc_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            10_usize.pow(6),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedLASWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_sswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::ChunkedSSWWC,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_sswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::ChunkedSSWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_sswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedSSWWC,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_sswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedSSWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    // #[test]
    // fn gpu_tuple_loss_or_duplicates_chunked_sswwc_i32_12_bits() -> Result<(), Box<dyn Error>> {
    //     gpu_tuple_loss_or_duplicates_i32(
    //         (32 << 20) / mem::size_of::<i32>(),
    //         GpuRadixPartitionAlgorithm::ChunkedSSWWC,
    //         12,
    //         GridSize::from(1),
    //         BlockSize::from(128),
    //     )
    // }

    // #[test]
    // fn gpu_verify_partitions_chunked_sswwc_i32_12_bits() -> Result<(), Box<dyn Error>> {
    //     gpu_verify_partitions_i32(
    //         (32 << 20) / mem::size_of::<i32>(),
    //         1..=(32 << 20),
    //         GpuRadixPartitionAlgorithm::ChunkedSSWWC,
    //         12,
    //         GridSize::from(1),
    //         BlockSize::from(128),
    //     )
    // }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_sswwc_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            10_usize.pow(6),
            GpuRadixPartitionAlgorithm::ChunkedSSWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_sswwc_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            10_usize.pow(6),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedSSWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_sswwc_non_temporal_i32_10_bits(
    ) -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::ChunkedSSWWCNT,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::ChunkedSSWWCv2,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::ChunkedSSWWCv2,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_sswwc_v2_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedSSWWCv2,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_sswwc_v2_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedSSWWCv2,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    // #[test]
    // fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2_i32_12_bits() -> Result<(), Box<dyn Error>> {
    //     gpu_tuple_loss_or_duplicates_i32(
    //         (32 << 20) / mem::size_of::<i32>(),
    //         GpuRadixPartitionAlgorithm::ChunkedSSWWCv2,
    //         12,
    //         GridSize::from(1),
    //         BlockSize::from(128),
    //     )
    // }

    // #[test]
    // fn gpu_verify_partitions_chunked_sswwc_v2_i32_12_bits() -> Result<(), Box<dyn Error>> {
    //     gpu_verify_partitions_i32(
    //         (32 << 20) / mem::size_of::<i32>(),
    //         1..=(32 << 20),
    //         GpuRadixPartitionAlgorithm::ChunkedSSWWCv2,
    //         12,
    //         GridSize::from(1),
    //         BlockSize::from(128),
    //     )
    // }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            10_usize.pow(6),
            GpuRadixPartitionAlgorithm::ChunkedSSWWCv2,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_sswwc_v2_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            10_usize.pow(6),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedSSWWCv2,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWC,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_hsswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWC,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_hsswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            10_usize.pow(6),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_hsswwc_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            10_usize.pow(6),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v2_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWCv2,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v2_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWCv2,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_hsswwc_v2_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWCv2,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_hsswwc_v2_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWCv2,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v2_non_power_two() -> Result<(), Box<dyn Error>>
    {
        gpu_tuple_loss_or_duplicates_i32(
            10_usize.pow(6),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWCv2,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_hsswwc_v2_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            10_usize.pow(6),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWCv2,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v3_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWCv3,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v3_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWCv3,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_hsswwc_v3_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWCv3,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_hsswwc_v3_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWCv3,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v3_non_power_two() -> Result<(), Box<dyn Error>>
    {
        gpu_tuple_loss_or_duplicates_i32(
            10_usize.pow(6),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWCv3,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_hsswwc_v3_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            10_usize.pow(6),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::ChunkedHSSWWCv3,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_contiguous_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::Contiguous,
            2,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_contiguous_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::Contiguous,
            10,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_contiguous_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::Contiguous,
            2,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_contiguous_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::Contiguous,
            10,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_contiguous_i32_12_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuRadixPartitionAlgorithm::Contiguous,
            12,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_contiguous_i32_12_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::Contiguous,
            12,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_contiguous_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            10_usize.pow(6),
            GpuRadixPartitionAlgorithm::Contiguous,
            10,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_contiguous_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            10_usize.pow(6),
            1..=(32 << 20),
            GpuRadixPartitionAlgorithm::Contiguous,
            10,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }
}
