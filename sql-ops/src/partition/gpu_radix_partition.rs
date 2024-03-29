// Copyright 2019-2022 Clemens Lutz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::cpu_radix_partition::CpuHistogramAlgorithm;
use super::{
    partition_input_chunk, HistogramAlgorithmType, PartitionOffsets, PartitionedRelation,
    RadixBits, RadixPass, Tuple,
};
use crate::constants;
use crate::error::{ErrorKind, Result};
use crate::prefix_scan::{GpuPrefixScanState, GpuPrefixSum};
use numa_gpu::runtime::allocator::{Allocator, MemType};
use numa_gpu::runtime::memory::{
    LaunchableMem, LaunchableMutPtr, LaunchableMutSlice, LaunchablePtr, LaunchableSlice, Mem,
};
use rustacuda::context::CurrentContext;
use rustacuda::device::DeviceAttribute;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::{DeviceBuffer, DeviceCopy};
use rustacuda::stream::Stream;
use rustacuda::{launch, launch_cooperative};
use std::cmp;
use std::ffi;
use std::mem;

/// Arguments to the C/C++ prefix sum function.
///
/// Note that the struct's layout must be kept in sync with its counterpart in
/// C/C++.
#[repr(C)]
#[derive(Clone, Debug)]
struct PrefixSumArgs {
    // Inputs
    partition_attr: LaunchablePtr<ffi::c_void>,
    data_len: usize,
    canonical_chunk_len: usize,
    padding_len: u32,
    radix_bits: u32,
    ignore_bits: u32,

    // State
    prefix_scan_state: LaunchableMutPtr<GpuPrefixScanState<u64>>,
    tmp_partition_offsets: LaunchableMutPtr<u64>,

    // Outputs
    partition_offsets: LaunchableMutPtr<u64>,
}

unsafe impl DeviceCopy for PrefixSumArgs {}

/// Arguments to the C/C++ prefix sum and transform function.
///
/// Note that the struct's layout must be kept in sync with its counterpart in
/// C/C++.
#[repr(C)]
#[derive(Clone, Debug)]
struct PrefixSumAndCopyWithPayloadArgs {
    // Inputs
    src_partition_attr: LaunchablePtr<ffi::c_void>,
    src_payload_attr: LaunchablePtr<ffi::c_void>,
    data_len: usize,
    canonical_chunk_len: usize,
    padding_len: u32,
    radix_bits: u32,
    ignore_bits: u32,

    // State
    prefix_scan_state: LaunchableMutPtr<GpuPrefixScanState<u64>>,
    tmp_partition_offsets: LaunchableMutPtr<u64>,

    // Outputs
    dst_partition_attr: LaunchableMutPtr<ffi::c_void>,
    dst_payload_attr: LaunchableMutPtr<ffi::c_void>,
    partition_offsets: LaunchableMutPtr<u64>,
}

unsafe impl DeviceCopy for PrefixSumAndCopyWithPayloadArgs {}

/// Arguments to the C/C++ prefix sum and transform function.
///
/// Note that the struct's layout must be kept in sync with its counterpart in
/// C/C++.
#[repr(C)]
#[derive(Clone, Debug)]
struct PrefixSumAndTransformArgs {
    // Inputs
    partition_id: u32,
    src_relation: LaunchablePtr<ffi::c_void>,
    src_offsets: LaunchablePtr<u64>,
    src_chunks: u32,
    src_radix_bits: u32,
    data_len: usize,
    padding_len: u32,
    radix_bits: u32,
    ignore_bits: u32,

    // State
    prefix_scan_state: LaunchableMutPtr<GpuPrefixScanState<u64>>,
    tmp_partition_offsets: LaunchableMutPtr<u64>,

    // Outputs
    dst_partition_attr: LaunchableMutPtr<ffi::c_void>,
    dst_payload_attr: LaunchableMutPtr<ffi::c_void>,
    partition_offsets: LaunchableMutPtr<u64>,
}

unsafe impl DeviceCopy for PrefixSumAndTransformArgs {}

/// Arguments to the C/C++ partitioning function.
///
/// Note that the struct's layout must be kept in sync with its counterpart in
/// C/C++.
#[repr(C)]
#[derive(Clone, Debug)]
struct RadixPartitionArgs {
    // Inputs
    partition_attr_data: LaunchablePtr<ffi::c_void>,
    payload_attr_data: LaunchablePtr<ffi::c_void>,
    data_len: usize,
    padding_len: u32,
    radix_bits: u32,
    ignore_bits: u32,
    partition_offsets: LaunchablePtr<u64>,

    // State
    tmp_partition_offsets: LaunchablePtr<u32>,
    l2_cache_buffers: LaunchablePtr<i8>,
    device_memory_buffers: LaunchablePtr<i8>,
    device_memory_buffer_bytes: u64,

    // Outputs
    partitioned_relation: LaunchableMutPtr<ffi::c_void>,
}

unsafe impl DeviceCopy for RadixPartitionArgs {}

pub trait GpuRadixPartitionable: Sized + DeviceCopy {
    fn prefix_sum_impl(
        rp: &mut GpuRadixPartitioner,
        pass: RadixPass,
        partition_attr: LaunchableSlice<'_, Self>,
        partition_offsets: &mut PartitionOffsets<Tuple<Self, Self>>,
        stream: &Stream,
    ) -> Result<()>;

    fn prefix_sum_and_copy_with_payload_impl(
        rp: &mut GpuRadixPartitioner,
        pass: RadixPass,
        src_partition_attr: LaunchableSlice<'_, Self>,
        src_payload_attr: LaunchableSlice<'_, Self>,
        dst_partition_attr: LaunchableMutSlice<'_, Self>,
        dst_payload_attr: LaunchableMutSlice<'_, Self>,
        partition_offsets: &mut PartitionOffsets<Tuple<Self, Self>>,
        stream: &Stream,
    ) -> Result<()>;

    fn prefix_sum_and_transform_impl(
        rp: &mut GpuRadixPartitioner,
        pass: RadixPass,
        partition_id: u32,
        src_relation: &PartitionedRelation<Tuple<Self, Self>>,
        dst_partition_attr: LaunchableMutSlice<'_, Self>,
        dst_payload_attr: LaunchableMutSlice<'_, Self>,
        partition_offsets: &mut PartitionOffsets<Tuple<Self, Self>>,
        stream: &Stream,
    ) -> Result<()>;

    fn allocate_partition_state_impl(rp: &mut GpuRadixPartitioner, pass: RadixPass) -> Result<()>;

    fn partition_impl(
        rp: &mut GpuRadixPartitioner,
        pass: RadixPass,
        partition_attr: LaunchableSlice<'_, Self>,
        payload_attr: LaunchableSlice<'_, Self>,
        partition_offsets: &mut PartitionOffsets<Tuple<Self, Self>>,
        partitioned_relation: &mut PartitionedRelation<Tuple<Self, Self>>,
        stream: &Stream,
    ) -> Result<()>;
}

/// Specifies the histogram algorithm that computes the partition offsets.
#[derive(Copy, Clone, Debug)]
pub enum GpuHistogramAlgorithm {
    /// Chunked partitions, that are computed on the GPU.
    ///
    /// `Chunked` computes a separate set of partitions per thread block. Tuples
    /// of the resulting partitions are thus distributed among all chunks.
    ///
    /// It was originally introduced for NUMA locality by Schuh et al. in "An
    /// Experimental Comparison of Thirteen Relational Equi-Joins in Main
    /// Memory".
    ///
    /// On GPUs, it has two main benefits. First, thread blocks don't
    /// communicate to compute the histogram, and can avoid global
    /// synchronization. Second, the offsets are smaller, and potentially we can
    /// use 32-bit integers instead of 64-bit integers when caching them in
    /// shared memory during the partitioning phase.
    Chunked,

    /// Contiguous partitions, that are computed on the GPU.
    ///
    /// `Contiguous` computes the "normal" partition layout. Each resulting
    /// partition is laid out contiguously in memory.
    ///
    /// Note that this algorithm does not work on pre-`Pascal` GPUs, because it
    /// requires cooperative launch capability to perform grid synchronization.
    Contiguous,
}

impl From<GpuHistogramAlgorithm> for HistogramAlgorithmType {
    fn from(algo: GpuHistogramAlgorithm) -> Self {
        match algo {
            GpuHistogramAlgorithm::Chunked => Self::Chunked,
            GpuHistogramAlgorithm::Contiguous => Self::Contiguous,
        }
    }
}

impl From<CpuHistogramAlgorithm> for GpuHistogramAlgorithm {
    fn from(algo: CpuHistogramAlgorithm) -> Self {
        match algo {
            CpuHistogramAlgorithm::Chunked => Self::Chunked,
            CpuHistogramAlgorithm::ChunkedSimd => Self::Chunked,
        }
    }
}

/// Specifies the radix partition algorithm.
#[derive(Copy, Clone, Debug)]
pub enum GpuRadixPartitionAlgorithm {
    /// Non-caching radix partition.
    ///
    /// This is a standard, parallel radix partition algorithm.
    NC,

    /// Radix partitioning with look-ahead software write combining.
    ///
    /// This algorithm reorders tuples in shared memory before writing them out
    /// to device memory. The purpose is to coalesce as many writes as possible,
    /// which can lead to higher throughput.
    ///
    /// This algorithm was first described by Stehle and Jacobsen in "A Memory
    /// Bandwidth-Efficient Hybrid Radix Sort on GPUs". It is also implemented
    /// by Sioulas et al. for "Hardware-conscious Hash-Joins on GPUs", although
    /// the authors do not mention or cite it in the paper.
    LASWWC,

    /// Radix partitioning with shared software write combining, version 2.
    ///
    /// In version 1, a warp can block all other warps by holding locks on more
    /// than one buffer (i.e., leader candidates).
    ///
    /// Version 2 tries to avoid blocking other warps by releasing all locks except
    /// one (i.e., the leader's buffer lock).
    SSWWCv2,

    /// Radix partitioning with shared software write combining, version 2G.
    ///
    /// Version 2G is exactly the same algorithm as version 2, but stores the
    /// SWWC buffers in device memory instead of in shared memory. The fanout
    /// can thus be scaled higher.
    ///
    /// Ideally, the buffers are retained in the GPU L2 cache. Storing buffers
    /// in the L1 cache is infeasible for the targeted use-case of high fanouts,
    /// as the buffers are too large. To avoid polluting the L2 cache, reading
    /// input and writing output uses streaming load and store instructions.
    SSWWCv2G,

    /// Radix partitioning with hierarchical shared software write combining, version 4.
    ///
    /// Version 4 performs the buffer flush from dmem to memory asynchronously with
    /// double-buffering.
    ///
    /// Double-buffering means that there are `fanout + #warps` dmem buffers. Thus each warp owns a
    /// spare buffer. When the dmem buffer of a partition is
    /// full, the warp that holds the lock exchanges the full dmem buffer for its empty spare
    /// buffer, and releases the lock. Only then does the warp flush the dmem buffer to CPU memory.
    ///
    /// Double-buffering enables all warps to always make progress during the dmem flush, because
    /// there is always a (partially-) empty dmem buffer available.
    HSSWWCv4,
}

#[derive(Debug)]
enum PrefixSumState {
    Chunked,
    Contiguous(Mem<GpuPrefixScanState<u64>>),
}

#[derive(Debug)]
enum RadixPartitionState {
    None,
    SSWWCv2G {
        offsets_buffer: DeviceBuffer<u32>,
        swwc_buffer: DeviceBuffer<i8>,
    },
    HSSWWC {
        device_memory_buffers: DeviceBuffer<i8>,
        buffer_bytes_per_block: usize,
    },
}

#[derive(Debug)]
pub struct GpuRadixPartitioner {
    radix_bits: RadixBits,
    prefix_sum_algorithm: GpuHistogramAlgorithm,
    partition_algorithm: GpuRadixPartitionAlgorithm,
    prefix_sum_state: PrefixSumState,
    partition_state: RadixPartitionState,
    grid_size: GridSize,
    block_size: BlockSize,
    rp_block_size: BlockSize,
    dmem_buffer_bytes: usize,
}

impl GpuRadixPartitioner {
    /// Creates a new CPU radix partitioner.
    pub fn new(
        prefix_sum_algorithm: GpuHistogramAlgorithm,
        partition_algorithm: GpuRadixPartitionAlgorithm,
        radix_bits: RadixBits,
        grid_size: &GridSize,
        block_size: &BlockSize,
        dmem_buffer_bytes: usize,
    ) -> Result<Self> {
        let prefix_scan_state_len = GpuPrefixSum::state_len(grid_size.clone(), block_size.clone())?;

        let prefix_sum_state = match prefix_sum_algorithm {
            GpuHistogramAlgorithm::Chunked => PrefixSumState::Chunked,
            GpuHistogramAlgorithm::Contiguous => PrefixSumState::Contiguous(Allocator::alloc_mem(
                MemType::CudaDevMem,
                prefix_scan_state_len,
            )),
        };

        let partition_state = match partition_algorithm {
            GpuRadixPartitionAlgorithm::SSWWCv2G => unsafe {
                RadixPartitionState::SSWWCv2G {
                    offsets_buffer: DeviceBuffer::uninitialized(0)?,
                    swwc_buffer: DeviceBuffer::uninitialized(0)?,
                }
            },
            GpuRadixPartitionAlgorithm::HSSWWCv4 => RadixPartitionState::HSSWWC {
                device_memory_buffers: unsafe { DeviceBuffer::uninitialized(0)? },
                buffer_bytes_per_block: 0,
            },
            _ => RadixPartitionState::None,
        };

        let rp_block_size = BlockSize::from(cmp::min(
            block_size.x,
            match partition_algorithm {
                GpuRadixPartitionAlgorithm::NC => 1024,
                GpuRadixPartitionAlgorithm::LASWWC => 1024,
                GpuRadixPartitionAlgorithm::SSWWCv2 => 1024,
                GpuRadixPartitionAlgorithm::SSWWCv2G => 1024,
                GpuRadixPartitionAlgorithm::HSSWWCv4 => 512,
            },
        ));

        Ok(Self {
            radix_bits,
            prefix_sum_algorithm,
            partition_algorithm,
            prefix_sum_state,
            partition_state,
            grid_size: grid_size.clone(),
            block_size: block_size.clone(),
            rp_block_size,
            dmem_buffer_bytes,
        })
    }

    /// Computes the prefix sum.
    ///
    /// The prefix sum performs a scan over all partitioning keys. It first
    /// computes a histogram. The prefix sum is computed from this histogram.
    ///
    /// The prefix sum serves two main purposes:
    ///
    /// 1. The prefix sums are used in `partition` as offsets in an array for
    ///    the output partitions.
    /// 2. The prefix sum can also be used to detect skew in the data.
    ///
    /// ## Parallelism
    ///
    /// The function is internally parallelized by the GPU. The function is
    /// *not* thread-safe for multiple callers.
    pub fn prefix_sum<T: DeviceCopy + GpuRadixPartitionable>(
        &mut self,
        pass: RadixPass,
        partition_attr: LaunchableSlice<'_, T>,
        partition_offsets: &mut PartitionOffsets<Tuple<T, T>>,
        stream: &Stream,
    ) -> Result<()> {
        T::prefix_sum_impl(self, pass, partition_attr, partition_offsets, stream)
    }

    /// Computes the prefix sum on a partitioned relation, and copies the data.
    ///
    /// The typical partitioning workflow first calls `prefix_sum`, and then
    /// calls `partition`. However, if performed over an interconnect, this
    /// workflow transfers the data twice.
    ///
    /// With `prefix_sum_and_copy_with_payload`, the data can be copied to GPU
    /// memory and thus the data transfer occurs only once.
    ///
    /// ## Parallelism
    ///
    /// The function is internally parallelized by the GPU. The function is
    /// *not* thread-safe for multiple callers.
    ///
    /// ## Limitations
    ///
    /// Currently only the `Contiguous` histogram algorithm is supported.
    /// The reason is that `prefix_sum_and_copy_with_payload` is typically used
    /// for small relations that fit into GPU memory. Thus the next step in the
    /// workflow is a SQL operator (e.g., join), which only takes a contiguous
    /// relation as input.
    pub fn prefix_sum_and_copy_with_payload<T: DeviceCopy + GpuRadixPartitionable>(
        &mut self,
        pass: RadixPass,
        src_partition_attr: LaunchableSlice<'_, T>,
        src_payload_attr: LaunchableSlice<'_, T>,
        dst_partition_attr: LaunchableMutSlice<'_, T>,
        dst_payload_attr: LaunchableMutSlice<'_, T>,
        partition_offsets: &mut PartitionOffsets<Tuple<T, T>>,
        stream: &Stream,
    ) -> Result<()> {
        T::prefix_sum_and_copy_with_payload_impl(
            self,
            pass,
            src_partition_attr,
            src_payload_attr,
            dst_partition_attr,
            dst_payload_attr,
            partition_offsets,
            stream,
        )
    }

    /// Computes the prefix sum on a partitioned relation, and transforms to a
    /// columnar format.
    ///
    /// ## Layout transformation
    ///
    /// Multi-pass partitioning requires the data in a columnar format. However,
    /// the first partitioning pass stores each partition in a row format.
    ///
    /// `prefix_sum_and_transform` transforms the row format into a column
    /// format, in addition to computing the prefix sum.
    ///
    /// ## Chunk concatenation
    ///
    /// The transform concatenates chunked partitions into contiguous partitions.
    /// `partition` and SQL operators (e.g., join) expect contiguous input. This
    /// design reduces the number of operator variants required from
    /// (layouts * operators) to (layouts + operators).
    ///
    /// ## Parallelism
    ///
    /// The function is internally parallelized by the GPU. The function is
    /// *not* thread-safe for multiple callers.
    ///
    /// ## Limitations
    ///
    /// Currently only the `Contiguous` histogram algorithm is supported. See
    /// `prefix_sum_and_copy_with_payload` for details.
    pub fn prefix_sum_and_transform<T: DeviceCopy + GpuRadixPartitionable>(
        &mut self,
        pass: RadixPass,
        partition_id: u32,
        src_relation: &PartitionedRelation<Tuple<T, T>>,
        dst_partition_attr: LaunchableMutSlice<'_, T>,
        dst_payload_attr: LaunchableMutSlice<'_, T>,
        partition_offsets: &mut PartitionOffsets<Tuple<T, T>>,
        stream: &Stream,
    ) -> Result<()> {
        T::prefix_sum_and_transform_impl(
            self,
            pass,
            partition_id,
            src_relation,
            dst_partition_attr,
            dst_payload_attr,
            partition_offsets,
            stream,
        )
    }

    /// Preallocates the internal state of `partition`
    ///
    /// Some partitioning variants use GPU memory buffers to hold internal state
    /// (e.g., HSSWWC). This state is lazy-allocated by the function and cached
    /// internally between function calls.
    ///
    /// `preallocate_partition_state` allows eager allocation of the state for
    /// optimization purposes. Specifically, it's sometimes possible to overlap
    /// the memory allocation with `prefix_sum` computation.
    pub fn preallocate_partition_state<T: GpuRadixPartitionable>(
        &mut self,
        pass: RadixPass,
    ) -> Result<()> {
        // Pre-load the CUDA module. The module consumes several hundred MB of
        // GPU memory, but is loaded lazily on first use. Thus, the preallocation
        // must also load the module to complete all state.
        //
        // Note that in some configurations, the Triton join fails with an
        // out-of-memory error if the module is not pre-loaded. The reason is
        // that the amount of free memory isn't correct without accounting for
        // the module.
        let _ = *crate::MODULE;

        T::allocate_partition_state_impl(self, pass)
    }

    /// Radix-partitions a relation by its key attribute.
    ///
    /// See the module-level documentation for details on the algorithm.
    ///
    /// ## Post-conditions
    ///
    /// - `partition_offsets` becomes uninitialized due to memory swap. However,
    ///   can be reused for `prefix_sum`.
    pub fn partition<T: DeviceCopy + GpuRadixPartitionable>(
        &mut self,
        pass: RadixPass,
        partition_attr: LaunchableSlice<'_, T>,
        payload_attr: LaunchableSlice<'_, T>,
        partition_offsets: &mut PartitionOffsets<Tuple<T, T>>,
        partitioned_relation: &mut PartitionedRelation<Tuple<T, T>>,
        stream: &Stream,
    ) -> Result<()> {
        T::partition_impl(
            self,
            pass,
            partition_attr,
            payload_attr,
            partition_offsets,
            partitioned_relation,
            stream,
        )
    }
}

macro_rules! impl_gpu_radix_partition_for_type {
    ($Type:ty, $Suffix:expr) => {
        impl GpuRadixPartitionable for $Type {
            paste::item! {
                fn prefix_sum_impl(
                    rp: &mut GpuRadixPartitioner,
                    pass: RadixPass,
                    partition_attr: LaunchableSlice<'_, $Type>,
                    partition_offsets: &mut PartitionOffsets<Tuple<$Type, $Type>>,
                    stream: &Stream,
                    ) -> Result<()> {

                    let device = CurrentContext::get_device()?;
                    let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
                    let radix_bits = rp
                        .radix_bits
                        .pass_radix_bits(pass)
                        .ok_or_else(||
                                ErrorKind::InvalidArgument(
                                    "The requested partitioning pass is not specified".to_string()
                                    ))?;

                    if partition_offsets.radix_bits() != radix_bits {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionedRelation has mismatching radix bits".to_string(),
                                ))?;
                    }
                    match rp.prefix_sum_algorithm {
                        GpuHistogramAlgorithm::Chunked
                            => if partition_offsets.num_chunks() != rp.grid_size.x {
                                Err(ErrorKind::InvalidArgument(
                                        "PartitionedRelation has mismatching number of chunks".to_string(),
                                        ))?;
                            },
                        GpuHistogramAlgorithm::Contiguous
                            => {
                                if partition_offsets.num_chunks() != 1 {
                                    Err(ErrorKind::InvalidArgument(
                                            "PartitionedRelation has mismatching number of chunks".to_string(),
                                            ))?;
                                }
                                if sm_count < partition_offsets.num_chunks() {
                                    Err(ErrorKind::InvalidArgument(
                                            "The Contiguous algorithm requires \
                                            all threads to run simultaneously. Try \
                                            decreasing the grid size.".to_string(),
                                            ))?;
                                }
                            }
                    }
                    if (partition_attr.len() + (rp.grid_size.x as usize) - 1)
                        / (rp.grid_size.x as usize) >= std::u32::MAX as usize {
                            let msg = "Relation is too large and causes an integer overflow. Try using more chunks by setting a higher CUDA grid size";
                            Err(ErrorKind::IntegerOverflow(msg.to_string(),))?
                    }

                    partition_offsets.set_data_len(partition_attr.len());

                    let module = *crate::MODULE;
                    let max_shared_mem_bytes =
                        device.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlockOptin)? as u32;
                    let fanout_u32 = rp.radix_bits.pass_fanout(pass).unwrap();
                    let ignore_bits = rp.radix_bits.pass_ignore_bits(pass);
                    let grid_size = rp.grid_size.clone();
                    let block_size = rp.block_size.clone();
                    let canonical_chunk_len = partition_input_chunk::input_chunk_size::<$Type>(partition_attr.len(), partition_offsets.num_chunks())?;

                    let tmp_partition_offsets = if let Some(ref mut local_offsets) = partition_offsets.local_offsets {
                        local_offsets.as_launchable_mut_ptr()
                    } else {
                        LaunchableMutPtr::null_mut()
                    };

                    let mut args = PrefixSumArgs {
                        partition_attr: partition_attr.as_launchable_ptr().as_void(),
                        data_len: partition_attr.len(),
                        canonical_chunk_len,
                        padding_len: partition_offsets.padding_len(),
                        radix_bits,
                        ignore_bits,
                        prefix_scan_state: LaunchableMutPtr::null_mut(),
                        tmp_partition_offsets,
                        partition_offsets: partition_offsets.offsets.as_launchable_mut_ptr(),
                    };

                    match rp.prefix_sum_state {
                        PrefixSumState::Chunked => {
                            let shared_mem_bytes = (
                                (block_size.x + (block_size.x >> constants::LOG2_NUM_BANKS)) + fanout_u32
                                ) * mem::size_of::<u32>() as u32;
                            assert!(
                                shared_mem_bytes <= max_shared_mem_bytes,
                                "Failed to allocate enough shared memory"
                                );

                            let name = std::ffi::CString::new(
                                stringify!([<gpu_chunked_prefix_sum_ $Suffix>])
                                ).unwrap();
                            let mut function = module.get_function(&name)?;
                            function.set_max_dynamic_shared_size_bytes(shared_mem_bytes)?;

                            unsafe {
                                launch!(
                                    function<<<
                                    grid_size.clone(),
                                    block_size.clone(),
                                    shared_mem_bytes,
                                    stream
                                    >>>(
                                        args.clone()
                                       ))?;
                            }
                        },
                        PrefixSumState::Contiguous(ref mut prefix_scan_state) => {
                            let shared_mem_bytes = (
                                (block_size.x + (block_size.x >> constants::LOG2_NUM_BANKS)) + fanout_u32
                                ) * mem::size_of::<u64>() as u32;
                            assert!(
                                shared_mem_bytes <= max_shared_mem_bytes,
                                "Failed to allocate enough shared memory"
                                );

                            args.prefix_scan_state = prefix_scan_state.as_launchable_mut_ptr();

                            unsafe {
                                launch_cooperative!(
                                    module.[<gpu_contiguous_prefix_sum_ $Suffix>]<<<
                                    grid_size.clone(),
                                    block_size.clone(),
                                    shared_mem_bytes,
                                    stream
                                    >>>(
                                        args.clone()
                                       ))?;
                            }
                        }
                    }

                    Ok(())
                }

                fn prefix_sum_and_copy_with_payload_impl(
                    rp: &mut GpuRadixPartitioner,
                    pass: RadixPass,
                    src_partition_attr: LaunchableSlice<'_, $Type>,
                    src_payload_attr: LaunchableSlice<'_, $Type>,
                    mut dst_partition_attr: LaunchableMutSlice<'_, $Type>,
                    mut dst_payload_attr: LaunchableMutSlice<'_, $Type>,
                    partition_offsets: &mut PartitionOffsets<Tuple<$Type, $Type>>,
                    stream: &Stream,
                    ) -> Result<()> {

                    let device = CurrentContext::get_device()?;
                    let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
                    let radix_bits = rp
                        .radix_bits
                        .pass_radix_bits(pass)
                        .ok_or_else(||
                                ErrorKind::InvalidArgument(
                                    "The requested partitioning pass is not specified".to_string()
                                    ))?;

                    if partition_offsets.radix_bits() != radix_bits {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionOffsets has mismatching radix bits".to_string(),
                                ))?;
                    }
                    match rp.prefix_sum_algorithm {
                        GpuHistogramAlgorithm::Chunked
                            => {
                                Err(ErrorKind::InvalidArgument(
                                        "Unsupported histogram algorithm, try using Contiguous instead".to_string(),
                                        ))?;
                            },
                        GpuHistogramAlgorithm::Contiguous
                            => {
                                if partition_offsets.num_chunks() != 1 {
                                    Err(ErrorKind::InvalidArgument(
                                            "PartitionOffsets has mismatching number of chunks".to_string(),
                                            ))?;
                                }
                                if sm_count < partition_offsets.num_chunks() {
                                    Err(ErrorKind::InvalidArgument(
                                            "The Contiguous algorithm requires \
                                            all threads to run simultaneously. Try \
                                            decreasing the grid size.".to_string(),
                                            ))?;
                                }
                            }
                    }

                    // Only checking for contiguous partitioned relations, as
                    // chunked aren't supported.
                    if src_partition_attr.len() >= std::u32::MAX as usize {
                            let msg = "Relation is too large and causes an integer overflow.";
                            Err(ErrorKind::IntegerOverflow(msg.to_string(),))?
                    }

                    partition_offsets.set_data_len(src_partition_attr.len());

                    let module = *crate::MODULE;
                    let max_shared_mem_bytes =
                        device.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlockOptin)? as u32;
                    let fanout_u32 = rp.radix_bits.pass_fanout(pass).unwrap();
                    let ignore_bits = rp.radix_bits.pass_ignore_bits(pass);
                    let grid_size = rp.grid_size.clone();
                    let block_size = rp.block_size.clone();
                    let canonical_chunk_len = partition_input_chunk::input_chunk_size::<$Type>(
                        src_partition_attr.len(),
                        partition_offsets.num_chunks()
                        )?;

                    let tmp_partition_offsets = if let Some(ref mut local_offsets) = partition_offsets.local_offsets {
                        local_offsets.as_launchable_mut_ptr()
                    } else {
                        LaunchableMutPtr::null_mut()
                    };

                    let mut args = PrefixSumAndCopyWithPayloadArgs {
                        src_partition_attr: src_partition_attr.as_launchable_ptr().as_void(),
                        src_payload_attr: src_payload_attr.as_launchable_ptr().as_void(),
                        data_len: src_partition_attr.len(),
                        canonical_chunk_len,
                        padding_len: partition_offsets.padding_len(),
                        radix_bits,
                        ignore_bits,
                        prefix_scan_state: LaunchableMutPtr::null_mut(),
                        tmp_partition_offsets,
                        dst_partition_attr: dst_partition_attr.as_launchable_mut_ptr().as_void(),
                        dst_payload_attr: dst_payload_attr.as_launchable_mut_ptr().as_void(),
                        partition_offsets: partition_offsets.offsets.as_launchable_mut_ptr(),
                    };

                    match rp.prefix_sum_state {
                        PrefixSumState::Chunked => unimplemented!(),
                        PrefixSumState::Contiguous(ref mut prefix_scan_state) => {
                            let shared_mem_bytes = (
                                (block_size.x + (block_size.x >> constants::LOG2_NUM_BANKS)) + fanout_u32
                                ) * mem::size_of::<u64>() as u32;
                            assert!(
                                shared_mem_bytes <= max_shared_mem_bytes,
                                "Failed to allocate enough shared memory"
                                );

                            let name = std::ffi::CString::new(
                                stringify!([<gpu_contiguous_prefix_sum_and_copy_with_payload_ $Suffix _ $Suffix>])
                                ).unwrap();
                            let mut function = module.get_function(&name)?;
                            function.set_max_dynamic_shared_size_bytes(shared_mem_bytes)?;

                            args.prefix_scan_state = prefix_scan_state.as_launchable_mut_ptr();

                            unsafe {
                                launch_cooperative!(
                                    function<<<
                                    grid_size.clone(),
                                    block_size.clone(),
                                    shared_mem_bytes,
                                    stream
                                    >>>(
                                        args.clone()
                                       ))?;
                            }
                        }
                    }

                    Ok(())
                }

                fn prefix_sum_and_transform_impl(
                    rp: &mut GpuRadixPartitioner,
                    pass: RadixPass,
                    partition_id: u32,
                    src_relation: &PartitionedRelation<Tuple<$Type, $Type>>,
                    mut dst_partition_attr: LaunchableMutSlice<'_, $Type>,
                    mut dst_payload_attr: LaunchableMutSlice<'_, $Type>,
                    partition_offsets: &mut PartitionOffsets<Tuple<$Type, $Type>>,
                    stream: &Stream,
                    ) -> Result<()> {

                    let device = CurrentContext::get_device()?;
                    let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
                    let radix_bits = rp
                        .radix_bits
                        .pass_radix_bits(pass)
                        .ok_or_else(||
                                ErrorKind::InvalidArgument(
                                    "The requested partitioning pass is not specified".to_string()
                                    ))?;

                    if partition_id >= src_relation.fanout() {
                        Err(ErrorKind::InvalidArgument(
                                "Invalid partition ID".to_string(),
                                ))?;
                    }
                    if partition_offsets.radix_bits() != radix_bits {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionedRelation has mismatching radix bits".to_string(),
                                ))?;
                    }
                    match rp.prefix_sum_algorithm {
                        GpuHistogramAlgorithm::Chunked
                            => {
                                Err(ErrorKind::InvalidArgument(
                                        "Unsupported histogram algorithm, try using Contiguous instead".to_string(),
                                        ))?;
                            },
                        GpuHistogramAlgorithm::Contiguous
                            => {
                                if partition_offsets.num_chunks() != 1 {
                                    Err(ErrorKind::InvalidArgument(
                                            "PartitionedRelation has mismatching number of chunks".to_string(),
                                            ))?;
                                }
                                if sm_count < partition_offsets.num_chunks() {
                                    Err(ErrorKind::InvalidArgument(
                                            "The Contiguous algorithm requires \
                                            all threads to run simultaneously. Try \
                                            decreasing the grid size.".to_string(),
                                            ))?;
                                }
                            }
                    }

                    // Only checking for contiguous partitioned relations, as
                    // chunked aren't supported.
                    if src_relation.padded_len() >= std::u32::MAX as usize {
                            let msg = "Relation is too large and causes an integer overflow.";
                            Err(ErrorKind::IntegerOverflow(msg.to_string(),))?
                    }

                    partition_offsets.set_data_len(dst_partition_attr.len());

                    let module = *crate::MODULE;
                    let max_shared_mem_bytes =
                        device.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlockOptin)? as u32;
                    let fanout_u32 = rp.radix_bits.pass_fanout(pass).unwrap();
                    let ignore_bits = rp.radix_bits.pass_ignore_bits(pass);
                    let grid_size = rp.grid_size.clone();
                    let block_size = rp.block_size.clone();

                    let tmp_partition_offsets = if let Some(ref mut local_offsets) = partition_offsets.local_offsets {
                        local_offsets.as_launchable_mut_ptr()
                    } else {
                        LaunchableMutPtr::null_mut()
                    };

                    let mut args = PrefixSumAndTransformArgs {
                        partition_id,
                        src_relation: src_relation.relation.as_launchable_ptr().as_void(),
                        src_offsets: src_relation.offsets.as_launchable_ptr(),
                        src_chunks: src_relation.num_chunks(),
                        src_radix_bits: src_relation.radix_bits(),
                        data_len: src_relation.padded_len(),
                        padding_len: partition_offsets.padding_len(),
                        radix_bits,
                        ignore_bits,
                        prefix_scan_state: LaunchableMutPtr::null_mut(),
                        tmp_partition_offsets,
                        dst_partition_attr: dst_partition_attr.as_launchable_mut_ptr().as_void(),
                        dst_payload_attr: dst_payload_attr.as_launchable_mut_ptr().as_void(),
                        partition_offsets: partition_offsets.offsets.as_launchable_mut_ptr(),
                    };

                    match rp.prefix_sum_state {
                        PrefixSumState::Chunked => unimplemented!(),
                        PrefixSumState::Contiguous(ref mut prefix_scan_state) => {
                            let shared_mem_bytes = (
                                (block_size.x + (block_size.x >> constants::LOG2_NUM_BANKS))
                                + fanout_u32
                                + 2 * src_relation.num_chunks()
                                ) * mem::size_of::<u64>() as u32;
                            assert!(
                                shared_mem_bytes <= max_shared_mem_bytes,
                                "Failed to allocate enough shared memory"
                                );

                            let name = std::ffi::CString::new(
                                stringify!([<gpu_contiguous_prefix_sum_and_transform_ $Suffix _ $Suffix>])
                                ).unwrap();
                            let mut function = module.get_function(&name)?;
                            function.set_max_dynamic_shared_size_bytes(shared_mem_bytes)?;

                            args.prefix_scan_state = prefix_scan_state.as_launchable_mut_ptr();

                            unsafe {
                                launch_cooperative!(
                                    function<<<
                                    grid_size.clone(),
                                    block_size.clone(),
                                    shared_mem_bytes,
                                    stream
                                    >>>(
                                        args.clone()
                                       ))?;
                            }
                        }
                    }

                    Ok(())
                }
            }

            fn allocate_partition_state_impl(
                    rp: &mut GpuRadixPartitioner,
                    pass: RadixPass,
                ) -> Result<()> {

                    let grid_size = rp.grid_size.clone();
                    let rp_block_size = rp.rp_block_size.clone();
                    let device = CurrentContext::get_device()?;
                    let warp_size = device.get_attribute(DeviceAttribute::WarpSize)? as u32;
                    let fanout_u32 = rp.radix_bits.pass_fanout(pass).unwrap();

                    let partition_state = mem::replace(&mut rp.partition_state, RadixPartitionState::None);
                    let partition_state = match partition_state {
                        RadixPartitionState::SSWWCv2G { mut offsets_buffer, mut swwc_buffer } => {
                            let swwc_bytes = grid_size.x as usize
                                * fanout_u32 as usize
                                * constants::GPU_CACHE_LINE_SIZE as usize;
                            let offsets_len = grid_size.x as usize
                                * fanout_u32 as usize * 3;

                            if swwc_buffer.len() < swwc_bytes {
                                DeviceBuffer::drop(swwc_buffer).map_err(|(e, _)| e)?;
                                swwc_buffer = unsafe { DeviceBuffer::uninitialized(swwc_bytes)? };
                            };

                            if offsets_buffer.len() < offsets_len {
                                DeviceBuffer::drop(offsets_buffer).map_err(|(e, _)| e)?;
                                offsets_buffer = unsafe { DeviceBuffer::uninitialized(offsets_len)? };
                            };

                            let state = RadixPartitionState::SSWWCv2G { offsets_buffer, swwc_buffer };

                            state
                        },
                        state @ _ => state,
                    };
                    rp.partition_state = partition_state;

                    let dmem_buffer_bytes_per_block = match rp.partition_algorithm {
                        GpuRadixPartitionAlgorithm::HSSWWCv4 => {
                            let warps_per_block = rp_block_size.x / warp_size;
                            rp.dmem_buffer_bytes * (fanout_u32 + warps_per_block) as usize
                        },
                        _ => 0
                    };

                    if let RadixPartitionState::HSSWWC { ref device_memory_buffers, .. } = rp.partition_state {
                        let global_dmem_buffer_bytes = dmem_buffer_bytes_per_block as usize * grid_size.x as usize;

                        if device_memory_buffers.len() < global_dmem_buffer_bytes {
                            let new_buffer = unsafe { DeviceBuffer::uninitialized(global_dmem_buffer_bytes)? };
                            rp.partition_state = RadixPartitionState::HSSWWC{
                                device_memory_buffers: new_buffer,
                                buffer_bytes_per_block: dmem_buffer_bytes_per_block
                            };
                        }
                    };

                Ok(())
            }

            paste::item! {
                fn partition_impl(
                    rp: &mut GpuRadixPartitioner,
                    pass: RadixPass,
                    partition_attr: LaunchableSlice<'_, $Type>,
                    payload_attr: LaunchableSlice<'_, $Type>,
                    partition_offsets: &mut PartitionOffsets<Tuple<$Type, $Type>>,
                    partitioned_relation: &mut PartitionedRelation<Tuple<$Type, $Type>>,
                    stream: &Stream,
                    ) -> Result<()> {

                    let radix_bits = rp
                        .radix_bits
                        .pass_radix_bits(pass)
                        .ok_or_else(||
                                ErrorKind::InvalidArgument(
                                    "The requested partitioning pass is not specified".to_string()
                                    ))?;
                    if partition_attr.len() != payload_attr.len() {
                        Err(ErrorKind::InvalidArgument(
                                "Partition and payload attributes have different sizes".to_string(),
                                ))?;
                    }
                    if let Some(len) = partition_offsets.len() {
                        if partitioned_relation.len() != len {
                            Err(ErrorKind::InvalidArgument(
                                    "PartitionOffsets and PartitionedRelation have mismatching lengths".to_string(),
                                    ))?;
                        }
                    }
                    else {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionOffsets has no length".to_string()
                                ))?
                    }
                    if partitioned_relation.radix_bits() != radix_bits {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionedRelation has mismatching radix bits".to_string(),
                                ))?;
                    }
                    if (partition_attr.len() + (rp.grid_size.x as usize) - 1)
                        / (rp.grid_size.x as usize) >= std::u32::MAX as usize {
                            let msg = "Relation is too large and causes an integer overflow. Try using more chunks by setting a higher CUDA grid size";
                            Err(ErrorKind::IntegerOverflow(msg.to_string(),))?
                    }
                    if (partition_offsets.num_chunks() != partitioned_relation.num_chunks()) {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionOffsets and PartitionedRelation have mismatching chunks".to_string(),
                                ))?;
                    }
                    if (partition_offsets.radix_bits() != radix_bits) {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionOffsets has mismatching radix bits".to_string(),
                                ))?;
                    }
                    if partitioned_relation.offsets.mem_type() != partition_offsets.offsets.mem_type() {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionedRelation offsets and PartitionOffsets have mismatching memory::Mem types".to_string()
                                ))?;
                    }

                    let module = *crate::MODULE;
                    let grid_size = rp.grid_size.clone();
                    let rp_block_size = rp.rp_block_size.clone();
                    let device = CurrentContext::get_device()?;
                    let max_shared_mem_bytes =
                        device.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlockOptin)? as u32;
                    let fanout_u32 = rp.radix_bits.pass_fanout(pass).unwrap();
                    let ignore_bits = rp.radix_bits.pass_ignore_bits(pass);
                    let data_len = partition_attr.len();

                    Self::allocate_partition_state_impl(rp, pass)?;
                    let (tmp_partition_offsets, l2_cache_buffers, device_memory_buffers, device_memory_buffer_bytes) = match &rp.partition_state {
                        RadixPartitionState::SSWWCv2G { offsets_buffer, swwc_buffer } => {
                            let offsets_ptr = offsets_buffer.as_launchable_ptr();
                            let swwc_ptr = swwc_buffer.as_launchable_ptr();

                            (offsets_ptr, swwc_ptr, LaunchablePtr::null(), 0)
                        },
                        RadixPartitionState::HSSWWC{device_memory_buffers, buffer_bytes_per_block} => {
                            (
                                LaunchablePtr::null(),
                                LaunchablePtr::null(),
                                device_memory_buffers.as_launchable_ptr(),
                                *buffer_bytes_per_block as u64
                            )
                        }
                        _ => (LaunchablePtr::null(), LaunchablePtr::null(), LaunchablePtr::null(), 0),
                    };

                    let partition_offsets_ptr = if let Some(ref local_offsets) = partition_offsets.local_offsets {
                        local_offsets.as_launchable_ptr()
                    } else {
                        partition_offsets.offsets.as_launchable_ptr()
                    };

                    // Swap the offsets of PartitionedRelation with PartitionedOffsets.  This
                    // avoids copying the offsets, as the optimal copy strategy is non-trivial to
                    // determine (copy using CPU vs. GPU) as it depends on the memory type and NUMA
                    // node.
                    mem::swap(&mut partitioned_relation.offsets, &mut partition_offsets.offsets);

                    let args = RadixPartitionArgs {
                        partition_attr_data: partition_attr.as_launchable_ptr().as_void(),
                        payload_attr_data: payload_attr.as_launchable_ptr().as_void(),
                        data_len,
                        padding_len: partitioned_relation.padding_len(),
                        radix_bits,
                        ignore_bits,
                        partition_offsets: partition_offsets_ptr,
                        tmp_partition_offsets,
                        l2_cache_buffers,
                        device_memory_buffers,
                        device_memory_buffer_bytes,
                        partitioned_relation: partitioned_relation.relation.as_launchable_mut_ptr().as_void(),
                    };

                    match rp.partition_algorithm {
                        GpuRadixPartitionAlgorithm::NC => {
                            let shared_mem_bytes = fanout_u32 * mem::size_of::<u32>() as u32;
                            assert!(
                                shared_mem_bytes <= max_shared_mem_bytes,
                                "Failed to allocate enough shared memory"
                                );

                            unsafe {
                                launch!(
                                    module.[<gpu_chunked_radix_partition_ $Suffix _ $Suffix>]<<<
                                    grid_size,
                                    rp_block_size,
                                    shared_mem_bytes,
                                    stream
                                    >>>(
                                    args.clone()
                                       ))?;
                            }
                        },
                        GpuRadixPartitionAlgorithm::LASWWC => {
                            let name = std::ffi::CString::new(
                                stringify!([<gpu_chunked_laswwc_radix_partition_ $Suffix _ $Suffix>])
                                ).unwrap();
                            let mut function = module.get_function(&name)?;
                            function.set_max_dynamic_shared_size_bytes(max_shared_mem_bytes)?;

                            unsafe {
                                launch!(
                                    function<<<
                                    grid_size,
                                    rp_block_size,
                                    max_shared_mem_bytes,
                                    stream
                                    >>>(
                                        args.clone(),
                                        max_shared_mem_bytes
                                       ))?;
                            }
                        },
                        GpuRadixPartitionAlgorithm::SSWWCv2 => {
                            let name = std::ffi::CString::new(
                                stringify!([<gpu_chunked_sswwc_radix_partition_v2_ $Suffix _ $Suffix>])
                                ).unwrap();
                            let mut function = module.get_function(&name)?;
                            function.set_max_dynamic_shared_size_bytes(max_shared_mem_bytes)?;

                            unsafe {
                                launch!(
                                    function<<<
                                    grid_size,
                                    rp_block_size,
                                    max_shared_mem_bytes,
                                    stream
                                    >>>(
                                        args.clone(),
                                        max_shared_mem_bytes
                                       ))?;
                            }
                        },
                        GpuRadixPartitionAlgorithm::SSWWCv2G => {
                            let name = std::ffi::CString::new(
                                stringify!([<gpu_chunked_sswwc_radix_partition_v2g_ $Suffix _ $Suffix>])
                                ).unwrap();
                            let function = module.get_function(&name)?;

                            unsafe {
                                launch!(
                                    function<<<
                                    grid_size,
                                    rp_block_size,
                                    0,
                                    stream
                                    >>>(
                                        args.clone()
                                       ))?;
                            }
                        },
                        GpuRadixPartitionAlgorithm::HSSWWCv4 => {
                            let name = std::ffi::CString::new(
                                stringify!([<gpu_chunked_hsswwc_radix_partition_v4_ $Suffix _ $Suffix>])
                                ).unwrap();
                            let mut function = module.get_function(&name)?;
                            function.set_max_dynamic_shared_size_bytes(max_shared_mem_bytes)?;

                            unsafe {
                                launch!(
                                    function<<<
                                    grid_size,
                                    rp_block_size,
                                    max_shared_mem_bytes,
                                    stream
                                    >>>(
                                        args.clone(),
                                        max_shared_mem_bytes
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
