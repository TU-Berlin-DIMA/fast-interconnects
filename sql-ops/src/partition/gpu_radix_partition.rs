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
use std::cmp;
use std::convert::TryInto;
use std::ffi::{self, CString};
use std::mem;
use std::ops::{Index, IndexMut};

extern "C" {
    fn cpu_chunked_prefix_sum_int32(args: *mut PrefixSumArgs, num_chunks: u32);
    fn cpu_chunked_prefix_sum_int64(args: *mut PrefixSumArgs, num_chunks: u32);
}

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
    padding_len: u32,
    radix_bits: u32,

    // State
    prefix_scan_state: LaunchableMutPtr<GpuPrefixScanState<u64>>,
    tmp_partition_offsets: LaunchableMutPtr<u64>,

    // Outputs
    partition_offsets: LaunchableMutPtr<u64>,
}

unsafe impl DeviceCopy for PrefixSumArgs {}

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
    partition_offsets: LaunchablePtr<u64>,

    // State
    device_memory_buffers: LaunchableMutPtr<i8>,
    device_memory_buffer_bytes: u64,

    // Outputs
    partitioned_relation: LaunchableMutPtr<ffi::c_void>,
}

unsafe impl DeviceCopy for RadixPartitionArgs {}

pub trait GpuRadixPartitionable: Sized + DeviceCopy {
    fn prefix_sum_impl(
        rp: &mut GpuRadixPartitioner,
        partition_attr: LaunchableSlice<'_, Self>,
        partition_offsets: &mut PartitionOffsets<Tuple<Self, Self>>,
        stream: &Stream,
    ) -> Result<()>;

    fn partition_impl(
        rp: &mut GpuRadixPartitioner,
        partition_attr: LaunchableSlice<'_, Self>,
        payload_attr: LaunchableSlice<'_, Self>,
        partition_offsets: PartitionOffsets<Tuple<Self, Self>>,
        partitioned_relation: &mut PartitionedRelation<Tuple<Self, Self>>,
        stream: &Stream,
    ) -> Result<()>;
}

/// Partition offsets for an array of chunked partitions.
///
/// The offsets describe a `PartitionedRelation`. The offsets reference
/// partitions within an array. Optionally, each partition includes padding.
///
/// # Layout
///
/// The layout looks as follows (C = chunk, P = partition):
///
/// ```
/// C0.P0 | C0.P1 | ... | C0.PN | C1.P0 | C1.P1 | ... | C1.PN | ... | CM.PN
/// ```
///
/// Note that no restrictions are placed on the data layout (i.e., offsets may
/// be numerically unordered).
///
/// # Invariants
///
///  - `radix_bits` must match in `GpuRadixPartitioner`
///  - `max_chunks` must equal the maximum number of chunks computed at runtime
///     (e.g., the grid size)
#[derive(Debug)]
pub struct PartitionOffsets<T: DeviceCopy> {
    pub offsets: Mem<u64>,
    chunks: u32,
    radix_bits: u32,
    phantom_data: std::marker::PhantomData<T>,
}

impl<T: DeviceCopy> PartitionOffsets<T> {
    // Creates a new partition offsets array.
    pub fn new(
        histogram_algorithm: GpuHistogramAlgorithm,
        max_chunks: u32,
        radix_bits: u32,
        alloc_fn: MemAllocFn<u64>,
    ) -> Self {
        let chunks: u32 = match histogram_algorithm {
            GpuHistogramAlgorithm::CpuChunked => max_chunks,
            GpuHistogramAlgorithm::GpuChunked => max_chunks,
            GpuHistogramAlgorithm::GpuContiguous => 1,
        };

        let num_partitions = fanout(radix_bits);
        let offsets = alloc_fn(num_partitions * chunks as usize);

        Self {
            offsets,
            chunks,
            radix_bits,
            phantom_data: std::marker::PhantomData,
        }
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
    pub fn padding_len(&self) -> u32 {
        super::GPU_PADDING_BYTES / mem::size_of::<T>() as u32
    }
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
///  - `max_chunks` must equal the maximum number of chunks computed at runtime
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
        histogram_algorithm: GpuHistogramAlgorithm,
        radix_bits: u32,
        max_chunks: u32,
        partition_alloc_fn: MemAllocFn<T>,
        offsets_alloc_fn: MemAllocFn<u64>,
    ) -> Self {
        let chunks: u32 = match histogram_algorithm {
            GpuHistogramAlgorithm::CpuChunked => max_chunks,
            GpuHistogramAlgorithm::GpuChunked => max_chunks,
            GpuHistogramAlgorithm::GpuContiguous => 1,
        };

        let padding_len = super::GPU_PADDING_BYTES / mem::size_of::<T>() as u32;
        let num_partitions = fanout(radix_bits);
        let relation_len = len + (num_partitions * chunks as usize) * padding_len as usize;

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
        super::GPU_PADDING_BYTES / mem::size_of::<T>() as u32
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

/// Specifies the histogram algorithm that computes the partition offsets.
#[derive(Copy, Clone, Debug)]
pub enum GpuHistogramAlgorithm {
    /// Chunked partitions, that are computed on the CPU.
    ///
    /// `CpuChunked` computes the exact same result as `GpuChunked`. However,
    /// `CpuChunked` computes the histogram on the CPU instead of on the GPU.
    /// This saves us from transferring the keys twice over the interconnect,
    /// once for the histogram and once for partitioning.
    ///
    /// The reasoning is that the required histograms are cheap to compute. The
    /// target size at maximum 2^12 buckets, which is only 32 KiB and should fit
    /// into the CPU's L1 cache. This upper bound for buckets is given by the
    /// GPU hardware and partitioning algorithm.
    CpuChunked,

    /// Chunked partitions, that are computed on the GPU.
    ///
    /// `GpuChunked` computes a separate set of partitions per thread block. Tuples
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
    GpuChunked,

    /// Contiguous partitions, that are computed on the GPU.
    ///
    /// `GpuContiguous` computes the "normal" partition layout. Each resulting
    /// partition is laid out contiguously in memory.
    ///
    /// Note that this algorithm does not work on pre-`Pascal` GPUs, because it
    /// requires cooperative launch capability to perform grid synchronization.
    GpuContiguous,
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

    /// Radix partitioning with shared software write combining.
    ///
    /// This algorithm shares the software write-combine buffers within a thread
    /// block. The buffers are cached in shared memory. To share the buffers,
    /// the thread block synchronizes access to each buffer via a lock.
    SSWWC,

    /// Radix partitioning with shared software write combining and non-temporal
    /// loads/stores.
    SSWWCNT,

    /// Radix partitioning with shared software write combining, version 2.
    ///
    /// In version 1, a warp can block all other warps by holding locks on more
    /// than one buffer (i.e., leader candidates).
    ///
    /// Version 2 tries to avoid blocking other warps by releasing all locks except
    /// one (i.e., the leader's buffer lock).
    SSWWCv2,

    /// Radix partitioning with hierarchical shared software write combining.
    ///
    /// This algorithm adds a second level of software write-combine buffers in
    /// device memory. The purpose is to more efficiently transfer data over a
    /// GPU interconnect (e.g., NVLink). Larger buffers amortize the overheads
    /// such as TLB misses over more tuples, which can lead to higher throughput.
    HSSWWC,

    /// Radix partitioning with hierarchical shared software write combining, version 2.
    ///
    /// In version 1, a warp can block all other warps by holding locks on more
    /// than one buffer (i.e., leader candidates).
    ///
    /// Version 2 tries to avoid blocking other warps by releasing all locks except
    /// one (i.e., the leader's buffer lock).
    HSSWWCv2,

    /// Radix partitioning with hierarchical shared software write combining, version 3.
    ///
    /// Version 3 performs the buffer flush from dmem to memory asynchronously.
    /// This change enables other warps to make progress during the dmem flush, which
    /// is important because the dmem buffer is large (several MBs) and the flush can
    /// take a long time.
    HSSWWCv3,

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
    CpuChunked,
    GpuChunked,
    GpuContiguous(Mem<GpuPrefixScanState<u64>>, Mem<u64>),
}

#[derive(Debug)]
pub struct GpuRadixPartitioner {
    radix_bits: u32,
    log2_num_banks: u32,
    prefix_sum_algorithm: GpuHistogramAlgorithm,
    partition_algorithm: GpuRadixPartitionAlgorithm,
    prefix_sum_state: PrefixSumState,
    module: Module,
    grid_size: GridSize,
    block_size: BlockSize,
    dmem_buffer_bytes: usize,
    prefix_sum_args: Option<DeviceBox<PrefixSumArgs>>,
    radix_partition_args: Option<DeviceBox<RadixPartitionArgs>>,
}

impl GpuRadixPartitioner {
    /// Creates a new CPU radix partitioner.
    pub fn new(
        prefix_sum_algorithm: GpuHistogramAlgorithm,
        partition_algorithm: GpuRadixPartitionAlgorithm,
        radix_bits: u32,
        grid_size: &GridSize,
        block_size: &BlockSize,
        dmem_buffer_bytes: usize,
    ) -> Result<Self> {
        let num_partitions = fanout(radix_bits);
        let log2_num_banks = env!("LOG2_NUM_BANKS")
            .parse::<u32>()
            .expect("Failed to parse \"log2_num_banks\" string to an integer");
        let prefix_scan_state_len = GpuPrefixSum::state_len(grid_size.clone(), block_size.clone())?;
        let tmp_partition_offsets_len = num_partitions as usize * grid_size.x as usize;

        let prefix_sum_state = match prefix_sum_algorithm {
            GpuHistogramAlgorithm::CpuChunked => PrefixSumState::CpuChunked,
            GpuHistogramAlgorithm::GpuChunked => PrefixSumState::GpuChunked,
            GpuHistogramAlgorithm::GpuContiguous => PrefixSumState::GpuContiguous(
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
            prefix_sum_algorithm,
            partition_algorithm,
            prefix_sum_state,
            module,
            grid_size: grid_size.clone(),
            block_size: block_size.clone(),
            dmem_buffer_bytes,
            prefix_sum_args: None,
            radix_partition_args: None,
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
    /// The parallelism of this function depends on the `GpuHistogramAlgorithm`.
    /// GPU algorithms are internally parallelized. CPU algorithms are sequential
    /// and can be externally parallelized by the caller.
    pub fn prefix_sum<T: DeviceCopy + GpuRadixPartitionable>(
        &mut self,
        partition_attr: LaunchableSlice<'_, T>,
        partition_offsets: &mut PartitionOffsets<Tuple<T, T>>,
        stream: &Stream,
    ) -> Result<()> {
        T::prefix_sum_impl(self, partition_attr, partition_offsets, stream)
    }

    /// Radix-partitions a relation by its key attribute.
    ///
    /// See the module-level documentation for details on the algorithm.
    pub fn partition<T: DeviceCopy + GpuRadixPartitionable>(
        &mut self,
        partition_attr: LaunchableSlice<'_, T>,
        payload_attr: LaunchableSlice<'_, T>,
        partition_offsets: PartitionOffsets<Tuple<T, T>>,
        partitioned_relation: &mut PartitionedRelation<Tuple<T, T>>,
        stream: &Stream,
    ) -> Result<()> {
        T::partition_impl(
            self,
            partition_attr,
            payload_attr,
            partition_offsets,
            partitioned_relation,
            stream,
        )
    }

    // FIXME: Add helper function that transforms data and partitioned_relation
    //        into a slice per chunk. This is needed to parallelize prefix_sum
    //        on the CPU.
}

macro_rules! impl_gpu_radix_partition_for_type {
    ($Type:ty, $Suffix:expr) => {
        impl GpuRadixPartitionable for $Type {
            paste::item! {
                fn prefix_sum_impl(
                    rp: &mut GpuRadixPartitioner,
                    partition_attr: LaunchableSlice<'_, $Type>,
                    partition_offsets: &mut PartitionOffsets<Tuple<$Type, $Type>>,
                    stream: &Stream,
                    ) -> Result<()> {

                    if partition_offsets.radix_bits != rp.radix_bits {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionedRelation has mismatching radix bits".to_string(),
                                ))?;
                    }
                    match rp.prefix_sum_algorithm {
                        GpuHistogramAlgorithm::CpuChunked
                            => if partition_offsets.chunks() != rp.grid_size.x {
                                Err(ErrorKind::InvalidArgument(
                                        "PartitionedRelation has mismatching number of chunks".to_string(),
                                        ))?;
                            },
                        GpuHistogramAlgorithm::GpuChunked
                            => if partition_offsets.chunks() != rp.grid_size.x {
                                Err(ErrorKind::InvalidArgument(
                                        "PartitionedRelation has mismatching number of chunks".to_string(),
                                        ))?;
                            },
                        GpuHistogramAlgorithm::GpuContiguous
                            => if partition_offsets.chunks() != 1 {
                                Err(ErrorKind::InvalidArgument(
                                        "PartitionedRelation has mismatching number of chunks".to_string(),
                                        ))?;
                            }
                    }
                    if (partition_attr.len() + (rp.grid_size.x as usize) - 1)
                        / (rp.grid_size.x as usize) >= std::u32::MAX as usize {
                            let msg = "Relation is too large and causes an integer overflow. Try using more chunks by setting a higher CUDA grid size";
                            Err(ErrorKind::IntegerOverflow(msg.to_string(),))?
                    }

                    let module = &rp.module;
                    let device = CurrentContext::get_device()?;
                    let max_shared_mem_bytes =
                        device.get_attribute(DeviceAttribute::MaxSharedMemPerBlockOptin)? as u32;
                    let fanout_u32 = fanout(rp.radix_bits) as u32;
                    let grid_size = rp.grid_size.clone();
                    let block_size = rp.block_size.clone();

                    let mut args = PrefixSumArgs {
                        partition_attr: partition_attr.as_launchable_ptr().as_void(),
                        data_len: partition_attr.len(),
                        padding_len: partition_offsets.padding_len(),
                        radix_bits: rp.radix_bits,
                        prefix_scan_state: LaunchableMutPtr::null_mut(),
                        tmp_partition_offsets: LaunchableMutPtr::null_mut(),
                        partition_offsets: partition_offsets.offsets.as_launchable_mut_ptr(),
                    };

                    // FIXME: Remove internal chunking in CPU prefix_sum function
                    match rp.prefix_sum_state {
                        PrefixSumState::CpuChunked => {
                            unsafe {
                                [<cpu_chunked_prefix_sum_ $Suffix>](&mut args, grid_size.x);
                            }
                        },
                        PrefixSumState::GpuChunked => {
                            let shared_mem_bytes = (
                                (block_size.x + (block_size.x >> rp.log2_num_banks)) + fanout_u32
                                ) * mem::size_of::<u32>() as u32;
                            assert!(
                                shared_mem_bytes <= max_shared_mem_bytes,
                                "Failed to allocate enough shared memory"
                                );

                            unsafe {
                                launch!(
                                    module.[<gpu_chunked_prefix_sum_ $Suffix>]<<<
                                    grid_size.clone(),
                                    block_size.clone(),
                                    shared_mem_bytes,
                                    stream
                                    >>>(
                                        args.clone()
                                       ))?;
                            }
                        },
                        PrefixSumState::GpuContiguous(ref mut prefix_scan_state, ref mut tmp_partition_offsets) => {
                            let shared_mem_bytes = (
                                (block_size.x + (block_size.x >> rp.log2_num_banks)) + fanout_u32
                                ) * mem::size_of::<u64>() as u32;
                            assert!(
                                shared_mem_bytes <= max_shared_mem_bytes,
                                "Failed to allocate enough shared memory"
                                );

                            args.prefix_scan_state = prefix_scan_state.as_launchable_mut_ptr();
                            args.tmp_partition_offsets = tmp_partition_offsets.as_launchable_mut_ptr();

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
            }

            paste::item! {
                fn partition_impl(
                    rp: &mut GpuRadixPartitioner,
                    partition_attr: LaunchableSlice<'_, $Type>,
                    payload_attr: LaunchableSlice<'_, $Type>,
                    partition_offsets: PartitionOffsets<Tuple<$Type, $Type>>,
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
                    if (partition_attr.len() + (rp.grid_size.x as usize) - 1)
                        / (rp.grid_size.x as usize) >= std::u32::MAX as usize {
                            let msg = "Relation is too large and causes an integer overflow. Try using more chunks by setting a higher CUDA grid size";
                            Err(ErrorKind::IntegerOverflow(msg.to_string(),))?
                    }
                    if (partition_offsets.chunks != partitioned_relation.chunks) {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionOffsets and PartitionedRelation have mismatching chunks".to_string(),
                                ))?;
                    }
                    if (partition_offsets.radix_bits != rp.radix_bits) {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionOffsets has mismatching radix bits".to_string(),
                                ))?;
                    }

                    let module = &rp.module;
                    let grid_size = rp.grid_size.clone();
                    let device = CurrentContext::get_device()?;
                    let _warp_size = device.get_attribute(DeviceAttribute::WarpSize)? as u32;
                    let max_shared_mem_bytes =
                        device.get_attribute(DeviceAttribute::MaxSharedMemPerBlockOptin)? as u32;
                    let fanout_u32 = fanout(rp.radix_bits) as u32;
                    let data_len = partition_attr.len();

                    let block_size = rp.block_size.clone();
                    let rp_block_size: u32 = cmp::min(
                        block_size.x,
                        match rp.partition_algorithm {
                            GpuRadixPartitionAlgorithm::NC => 1024,
                            GpuRadixPartitionAlgorithm::LASWWC => 1024,
                            GpuRadixPartitionAlgorithm::SSWWC => 1024,
                            GpuRadixPartitionAlgorithm::SSWWCNT => 1024,
                            GpuRadixPartitionAlgorithm::SSWWCv2 => 1024,
                            GpuRadixPartitionAlgorithm::HSSWWC => 512,
                            GpuRadixPartitionAlgorithm::HSSWWCv2 => 512,
                            GpuRadixPartitionAlgorithm::HSSWWCv3 => 512,
                            GpuRadixPartitionAlgorithm::HSSWWCv4 => 512,
                        });

                    let (mut dmem_buffer, device_memory_buffer_bytes) = match rp.partition_algorithm {
                        GpuRadixPartitionAlgorithm::HSSWWC
                            | GpuRadixPartitionAlgorithm::HSSWWCv2
                            | GpuRadixPartitionAlgorithm::HSSWWCv3
                            | GpuRadixPartitionAlgorithm::HSSWWCv4 => {
                                let dmem_buffer_bytes: u64 = rp.dmem_buffer_bytes as u64;
                                let global_dmem_buffer_bytes = dmem_buffer_bytes * grid_size.x as u64;
                                (
                                    Some(Mem::CudaDevMem(unsafe { DeviceBuffer::uninitialized(global_dmem_buffer_bytes as usize)? })),
                                    dmem_buffer_bytes
                                )
                            }
                        _ => (None, 0)
                    };

                    let args = RadixPartitionArgs {
                        partition_attr_data: partition_attr.as_launchable_ptr().as_void(),
                        payload_attr_data: payload_attr.as_launchable_ptr().as_void(),
                        data_len,
                        padding_len: partitioned_relation.padding_len(),
                        radix_bits: rp.radix_bits,
                        partition_offsets: partition_offsets.offsets.as_launchable_ptr(),
                        device_memory_buffers: dmem_buffer
                            .as_mut()
                            .map_or(
                                LaunchableMutPtr::null_mut(),
                                |b| b.as_launchable_mut_ptr()
                                ),
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
                        GpuRadixPartitionAlgorithm::SSWWC => {
                            let name = std::ffi::CString::new(
                                    stringify!([<gpu_chunked_sswwc_radix_partition_ $Suffix _ $Suffix>])
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
                        GpuRadixPartitionAlgorithm::SSWWCNT => {
                            let name = std::ffi::CString::new(
                                    stringify!([<gpu_chunked_sswwc_non_temporal_radix_partition_ $Suffix _ $Suffix>])
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
                        GpuRadixPartitionAlgorithm::HSSWWC => {
                            let name = std::ffi::CString::new(
                                stringify!([<gpu_chunked_hsswwc_radix_partition_ $Suffix _ $Suffix>])
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
                        GpuRadixPartitionAlgorithm::HSSWWCv2 => {
                            let name = std::ffi::CString::new(
                                stringify!([<gpu_chunked_hsswwc_radix_partition_v2_ $Suffix _ $Suffix>])
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
                        GpuRadixPartitionAlgorithm::HSSWWCv3 => {
                            let name = std::ffi::CString::new(
                                stringify!([<gpu_chunked_hsswwc_radix_partition_v3_ $Suffix _ $Suffix>])
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

                    // Move ownership of offsets to PartitionedRelation.
                    // PartitionOffsets will be destroyed.
                    partitioned_relation.offsets = partition_offsets.offsets;

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
    use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemType};
    use numa_gpu::runtime::memory::Mem;
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
        histogram_algorithm: GpuHistogramAlgorithm,
        partition_algorithm: GpuRadixPartitionAlgorithm,
        radix_bits: u32,
        grid_size: GridSize,
        block_size: BlockSize,
    ) -> Result<(), Box<dyn Error>> {
        const PAYLOAD_RANGE: RangeInclusive<usize> = 1..=10000;
        const DMEM_BUFFER_BYTES: usize = 2 * 1024 * 1024;
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

        // Ensure that the allocated memory is zeroed
        let alloc_fn = Box::new(|len: usize| {
            let mut mem = Allocator::alloc_deref_mem(DerefMemType::CudaUniMem, len);
            mem.iter_mut().for_each(|x| *x = Default::default());
            mem.into()
        });

        let mut partition_offsets = PartitionOffsets::new(
            histogram_algorithm,
            grid_size.x,
            radix_bits,
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
        );

        let mut partitioned_relation = PartitionedRelation::new(
            tuples,
            histogram_algorithm,
            radix_bits,
            grid_size.x,
            alloc_fn,
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
        );

        let mut partitioner = GpuRadixPartitioner::new(
            histogram_algorithm,
            partition_algorithm,
            radix_bits,
            &grid_size,
            &block_size,
            DMEM_BUFFER_BYTES,
        )?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let data_key = Mem::CudaPinnedMem(data_key);
        let data_pay = Mem::CudaPinnedMem(data_pay);

        partitioner.prefix_sum(
            data_key.as_launchable_slice(),
            &mut partition_offsets,
            &stream,
        )?;

        partitioner.partition(
            data_key.as_launchable_slice(),
            data_pay.as_launchable_slice(),
            partition_offsets,
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
        histogram_algorithm: GpuHistogramAlgorithm,
        partition_algorithm: GpuRadixPartitionAlgorithm,
        radix_bits: u32,
        grid_size: GridSize,
        block_size: BlockSize,
    ) -> Result<(), Box<dyn Error>> {
        const PAYLOAD_RANGE: RangeInclusive<usize> = 1..=10000;
        const DMEM_BUFFER_BYTES: usize = 2 * 1024 * 1024;

        let _context = rustacuda::quick_init()?;

        let mut data_key: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;
        let mut data_pay: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;

        UniformRelation::gen_attr(&mut data_key, key_range)?;
        UniformRelation::gen_attr(&mut data_pay, PAYLOAD_RANGE)?;

        let mut partition_offsets = PartitionOffsets::new(
            histogram_algorithm,
            grid_size.x,
            radix_bits,
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
        );

        let mut partitioned_relation = PartitionedRelation::new(
            tuples,
            histogram_algorithm,
            radix_bits,
            grid_size.x,
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
        );

        let mut partitioner = GpuRadixPartitioner::new(
            histogram_algorithm,
            partition_algorithm,
            radix_bits,
            &grid_size,
            &block_size,
            DMEM_BUFFER_BYTES,
        )?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let data_key = Mem::CudaPinnedMem(data_key);
        let data_pay = Mem::CudaPinnedMem(data_pay);

        partitioner.prefix_sum(
            data_key.as_launchable_slice(),
            &mut partition_offsets,
            &stream,
        )?;

        partitioner.partition(
            data_key.as_launchable_slice(),
            data_pay.as_launchable_slice(),
            partition_offsets,
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
    fn gpu_tuple_loss_or_duplicates_cpu_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::CpuChunked,
            GpuRadixPartitionAlgorithm::NC,
            2,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_cpu_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuHistogramAlgorithm::CpuChunked,
            GpuRadixPartitionAlgorithm::NC,
            2,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_small_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            100,
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::NC,
            2,
            GridSize::from(4),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_small_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            100,
            1..=(32 << 20),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::NC,
            2,
            GridSize::from(4),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::NC,
            2,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::NC,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::NC,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::NC,
            10,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_i32_12_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::NC,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::NC,
            12,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            10_usize.pow(6),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::NC,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::NC,
            10,
            GridSize::from(10),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_contiguous_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuContiguous,
            GpuRadixPartitionAlgorithm::NC,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_contiguous_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuContiguous,
            GpuRadixPartitionAlgorithm::NC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_contiguous_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuHistogramAlgorithm::GpuContiguous,
            GpuRadixPartitionAlgorithm::NC,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_contiguous_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuHistogramAlgorithm::GpuContiguous,
            GpuRadixPartitionAlgorithm::NC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_contiguous_i32_12_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuContiguous,
            GpuRadixPartitionAlgorithm::NC,
            12,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_contiguous_i32_12_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuHistogramAlgorithm::GpuContiguous,
            GpuRadixPartitionAlgorithm::NC,
            12,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_contiguous_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            10_usize.pow(6),
            GpuHistogramAlgorithm::GpuContiguous,
            GpuRadixPartitionAlgorithm::NC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_contiguous_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            10_usize.pow(6),
            1..=(32 << 20),
            GpuHistogramAlgorithm::GpuContiguous,
            GpuRadixPartitionAlgorithm::NC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_laswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::LASWWC,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_laswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::LASWWC,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::LASWWC,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::LASWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_laswwc_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            10_usize.pow(6),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::LASWWC,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::LASWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_sswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::SSWWC,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_sswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::SSWWC,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::SSWWC,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::SSWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_sswwc_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            10_usize.pow(6),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::SSWWC,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::SSWWC,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::SSWWCNT,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::SSWWCv2,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::SSWWCv2,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::SSWWCv2,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::SSWWCv2,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            10_usize.pow(6),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::SSWWCv2,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::SSWWCv2,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWC,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWC,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWC,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            10_usize.pow(6),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWC,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWC,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v2_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv2,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v2_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv2,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv2,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv2,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv2,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv2,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v3_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv3,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v3_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv3,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv3,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv3,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv3,
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
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv3,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v4_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv4,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v4_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_tuple_loss_or_duplicates_i32(
            (32 << 20) / mem::size_of::<i32>(),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv4,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_hsswwc_v4_i32_2_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv4,
            2,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_hsswwc_v4_i32_10_bits() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            (32 << 20) / mem::size_of::<i32>(),
            1..=(32 << 20),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv4,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v4_non_power_two() -> Result<(), Box<dyn Error>>
    {
        gpu_tuple_loss_or_duplicates_i32(
            10_usize.pow(6),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv4,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }

    #[test]
    fn gpu_verify_partitions_chunked_hsswwc_v4_non_power_two() -> Result<(), Box<dyn Error>> {
        gpu_verify_partitions_i32(
            10_usize.pow(6),
            1..=(32 << 20),
            GpuHistogramAlgorithm::GpuChunked,
            GpuRadixPartitionAlgorithm::HSSWWCv4,
            10,
            GridSize::from(1),
            BlockSize::from(128),
        )
    }
}
