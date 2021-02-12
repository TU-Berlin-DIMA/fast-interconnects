/*
 * Copyright 2019-2021 Clemens Lutz, German Research Center for Artificial Intelligence
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

//! Radix partition operators for CPU and GPU.
//!
//! # Overview
//!
//! CPU and GPU partitioning operators are compatible. The devices can cooperate
//! to partition a relation in parallel. Note that this only holds for equivalent
//! operator algorithms, i.e., `chunked_radix_partition_swwc` cannot be combined
//! with `chunked_radix_partition`.
//!
//! Provided is one radix partitioning algorithm, chunked radix partitioning with
//! software write-combine buffering (SWWC). This algorithm is described by Schuh
//! et al. in Section 6 of "An Experimental Comparison of Thirteen Relational
//! Equi-Joins in Main Memory".
//!
//! # Thread-safety
//!
//! The radix partitioning operators are designed to be thread-safe. Although the
//! input data can be shared between threads, threads should typically work on
//! disjuct input partitions for correct results. In contrast, each thread must
//! have exclusive ownership of its output and intermediate state buffers.
//!
//! # Padding
//!
//! It is important to note that partitions are padded to, at minimum, the
//! cache-line size. This is necessary for SWWC buffering because cache-lines
//! are written back to memory as a whole. However, partition offsets are not
//! naturally aligned, because partitions can have any size. Therefore,
//! all partitions are padded in front by, at minimum, the length of a cache-line.
//! The cache-alignment is also necessary for non-temporal SIMD writes, which
//! must be aligned to their SIMD vector length.
//!
//! # Copyright notes
//!
//! The C/C++ CPU code is based on [code kindly published by Cagri Balkesen and
//! Claude Barthels][mchj]. As such, we adhere to their copyright and license
//! (MIT) in derived code. Modifications are licensed under Apache License 2.0.
//!
//! [mchj]: https://www.systems.ethz.ch/sites/default/files/file/PublishedCode/multicore-distributed-hashjoins-0_1.zip

use super::{
    fanout, HistogramAlgorithmType, PartitionOffsetsMutSlice, PartitionedRelationMutSlice,
    RadixPartitionInputChunk, Tuple,
};
use crate::constants;
use crate::error::{ErrorKind, Result};
use numa_gpu::runtime::allocator::{Allocator, DerefMemAllocFn, DerefMemType};
use numa_gpu::runtime::memory::DerefMem;
use numa_gpu::utils::CachePadded;
use rustacuda::memory::DeviceCopy;
use std::ffi::c_void;
use std::{mem, ptr};

extern "C" {
    fn cpu_swwc_buffer_bytes() -> usize;
    fn cpu_chunked_prefix_sum_int32(args: *mut PrefixSumArgs, chunk_id: u32, num_chunks: u32);
    fn cpu_chunked_prefix_sum_int64(args: *mut PrefixSumArgs, chunk_id: u32, num_chunks: u32);
    #[cfg(target_arch = "powerpc64")]
    fn cpu_chunked_prefix_sum_simd_int32(args: *mut PrefixSumArgs, chunk_id: u32, num_chunks: u32);
    #[cfg(target_arch = "powerpc64")]
    fn cpu_chunked_prefix_sum_simd_int64(args: *mut PrefixSumArgs, chunk_id: u32, num_chunks: u32);
    fn cpu_chunked_radix_partition_int32_int32(args: *mut RadixPartitionArgs);
    fn cpu_chunked_radix_partition_int64_int64(args: *mut RadixPartitionArgs);
    fn cpu_chunked_radix_partition_swwc_int32_int32(args: *mut RadixPartitionArgs);
    fn cpu_chunked_radix_partition_swwc_int64_int64(args: *mut RadixPartitionArgs);
    #[cfg(target_arch = "powerpc64")]
    fn cpu_chunked_radix_partition_swwc_simd_int32_int32(args: *mut RadixPartitionArgs);
    #[cfg(target_arch = "powerpc64")]
    fn cpu_chunked_radix_partition_swwc_simd_int64_int64(args: *mut RadixPartitionArgs);
}

/// Arguments to the C/C++ prefix sum function.
///
/// Note that the struct's layout must be kept in sync with its counterpart in
/// C/C++.
#[repr(C)]
#[derive(Clone, Debug)]
struct PrefixSumArgs {
    // Inputs
    partition_attr: *const c_void,
    data_len: usize,
    canonical_chunk_len: usize,
    padding_len: u32,
    radix_bits: u32,
    ignore_bits: u32,

    // State
    tmp_partition_offsets: *mut u32,

    // Outputs
    partition_offsets: *mut u64,
}

/// Arguments to the C/C++ partitioning function.
///
/// Note that the struct's layout must be kept in sync with its counterpart in
/// C/C++.
#[repr(C)]
#[derive(Debug)]
struct RadixPartitionArgs {
    // Inputs
    partition_attr_data: *const c_void,
    payload_attr_data: *const c_void,
    data_len: usize,
    padding_len: usize, // FIXME: u32
    radix_bits: u32,
    ignore_bits: u32,
    partition_offsets: *const u64,

    // State
    tmp_partition_offsets: *mut u64,
    write_combine_buffer: *mut c_void,

    // Outputs
    partitioned_relation: *mut c_void,
}

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
/// * The `radix_bits` must match in `PartitionedRelation` and `CpuRadixPartitioner`.
///
/// * The backing memory must be aligned to the cache-line size of the machine.
///   Hint: `DerefMem::Numa` alignes to the page size, which is a multiple of
///   the cache-line size
#[derive(Debug)]
struct WriteCombineBuffer {
    raw_memory: DerefMem<u64>,
}

impl WriteCombineBuffer {
    /// Creates a new set of SWWC buffers.
    fn new(radix_bits: u32, alloc_fn: DerefMemAllocFn<u64>) -> Self {
        let buffer_bytes = unsafe { cpu_swwc_buffer_bytes() };
        assert!(buffer_bytes as u32 <= constants::PADDING_BYTES, "Partition padding is too small for the SWWC buffers; padding must be at least {} bytes", buffer_bytes);

        let align_bytes = mem::align_of::<CachePadded<u64>>();
        let bytes = buffer_bytes * fanout(radix_bits) as usize + align_bytes;
        let raw_memory = alloc_fn(bytes / mem::size_of::<u64>());

        Self { raw_memory }
    }

    /// Returns a mutable slice to the aligned buffers
    fn as_mut_slice(&mut self) -> &mut [u64] {
        let (_, buffers, _) = unsafe {
            self.raw_memory
                .as_mut_slice()
                .align_to_mut::<CachePadded<u64>>()
        };

        let buffer_bytes = unsafe { cpu_swwc_buffer_bytes() };
        let len = buffer_bytes / mem::size_of::<u64>();

        let buffers_slice =
            unsafe { std::slice::from_raw_parts_mut(buffers.as_mut_ptr() as *mut u64, len) };
        buffers_slice
    }

    /// Computes the number of tuples per SWWC buffer.
    ///
    /// Note that `WriteCombineBuffer` contains one SWWC buffer per
    fn tuples_per_buffer<T: Sized>() -> u32 {
        let buffer_bytes = unsafe { cpu_swwc_buffer_bytes() };
        (buffer_bytes / mem::size_of::<T>()) as u32
    }
}

/// Specifies that the implementing type can be used as partitioning key in
/// `CpuRadixPartitioner`.
///
/// `CpuRadixPartitionable` is a trait for which specialized implementations
/// exist for each implementing type (currently i32 and i64). Specialization is
/// necessary because each type requires a different C++ function to be called.
///
/// See `CudaHashJoinable` for more details on the design decision.
pub trait CpuRadixPartitionable: Sized + DeviceCopy {
    fn prefix_sum_impl(
        rp: &mut CpuRadixPartitioner,
        partition_attr: RadixPartitionInputChunk<'_, Self>,
        partition_offsets: PartitionOffsetsMutSlice<'_, Tuple<Self, Self>>,
    ) -> Result<()>;

    fn partition_impl(
        rp: &mut CpuRadixPartitioner,
        partition_attr: RadixPartitionInputChunk<'_, Self>,
        payload_attr: RadixPartitionInputChunk<'_, Self>,
        partition_offsets: PartitionOffsetsMutSlice<Tuple<Self, Self>>,
        partitioned_relation: PartitionedRelationMutSlice<Tuple<Self, Self>>,
    ) -> Result<()>;
}

/// Specifies the histogram algorithm that computes the partition offsets.
#[derive(Copy, Clone, Debug)]
pub enum CpuHistogramAlgorithm {
    /// `Chunked` computes a separate set of partitions per thread block. Tuples of the resulting
    /// partitions are thus distributed among all chunks.
    ///
    /// It was originally introduced for NUMA locality by Schuh et al. in "An Experimental
    /// Comparison of Thirteen Relational Equi-Joins in Main Memory".
    Chunked,

    /// Prefix sum with SIMD optimizations.
    ///
    /// This is the same algorithm as `Chunked`, but uses SIMD load instructions. Also, loops are
    /// manually unrolled to 64 bytes.
    ///
    /// # Limitations
    ///
    /// Currently only implemented for PPC64le.
    ChunkedSimd,
}

impl From<CpuHistogramAlgorithm> for HistogramAlgorithmType {
    fn from(algorithm: CpuHistogramAlgorithm) -> Self {
        match algorithm {
            CpuHistogramAlgorithm::Chunked => Self::Chunked,
            CpuHistogramAlgorithm::ChunkedSimd => Self::Chunked,
        }
    }
}

/// Specifies the radix partition algorithm.
#[derive(Copy, Clone, Debug)]
pub enum CpuRadixPartitionAlgorithm {
    /// Non-caching radix partition.
    ///
    /// This is a standard, parallel radix partition algorithm.
    NC,

    /// Radix partition with software write-combining.
    ///
    /// This algorithm uses software-write combine buffers to avoid TLB misses.
    /// The buffers are flushed using non-temporal SIMD stores on x86-64. In
    /// contrast, PPC64le uses regular SIMD stores, as non-temporal hints
    /// don't actually enforce streaming, but cause additional overhead.
    Swwc,

    /// Radix partition with software write-combining and SIMD optimizations.
    ///
    /// This is the same algorithm as `Swwc`, but uses SIMD loads in addition to
    /// the SIMD buffer flush. Also, loops are manually unrolled to 64 bytes.
    ///
    /// # Limitations
    ///
    /// Currently only implemented for PPC64le.
    SwwcSimd,
}

#[derive(Debug)]
enum PrefixSumState {
    Chunked(DerefMem<u32>),
    ChunkedSimd(DerefMem<u32>),
}

/// Mutable internal state of the partition functions.
///
/// The state is reusable as long as the radix bits remain unchanged between
/// runs.
#[derive(Debug)]
enum RadixPartitionState {
    NC(DerefMem<u64>),
    Swwc(WriteCombineBuffer),
    SwwcSimd(WriteCombineBuffer),
}

/// A CPU radix partitioner that provides partitioning functions.
#[derive(Debug)]
pub struct CpuRadixPartitioner {
    radix_bits: u32,
    prefix_sum_state: PrefixSumState,
    radix_partition_state: RadixPartitionState,
}

impl CpuRadixPartitioner {
    /// Creates a new CPU radix partitioner.
    pub fn new(
        prefix_sum_algorithm: CpuHistogramAlgorithm,
        partition_algorithm: CpuRadixPartitionAlgorithm,
        radix_bits: u32,
        state_mem_type: DerefMemType,
    ) -> Self {
        let num_partitions = fanout(radix_bits) as usize;
        let vec_len = 4;
        let unroll_len = 4;

        let prefix_sum_state = match prefix_sum_algorithm {
            CpuHistogramAlgorithm::Chunked => PrefixSumState::Chunked(Allocator::alloc_deref_mem(
                state_mem_type.clone(),
                num_partitions,
            )),
            CpuHistogramAlgorithm::ChunkedSimd => {
                PrefixSumState::ChunkedSimd(Allocator::alloc_deref_mem(
                    state_mem_type.clone(),
                    num_partitions * vec_len * unroll_len,
                ))
            }
        };

        let radix_partition_state = match partition_algorithm {
            CpuRadixPartitionAlgorithm::NC => RadixPartitionState::NC(Allocator::alloc_deref_mem(
                state_mem_type.clone(),
                num_partitions,
            )),
            CpuRadixPartitionAlgorithm::Swwc => RadixPartitionState::Swwc(WriteCombineBuffer::new(
                radix_bits,
                Allocator::deref_mem_alloc_fn(state_mem_type.clone()),
            )),
            CpuRadixPartitionAlgorithm::SwwcSimd => {
                RadixPartitionState::SwwcSimd(WriteCombineBuffer::new(
                    radix_bits,
                    Allocator::deref_mem_alloc_fn(state_mem_type.clone()),
                ))
            }
        };

        Self {
            radix_bits,
            prefix_sum_state,
            radix_partition_state,
        }
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
    /// The function is thread-safe, and meant to be externally parallelized by
    /// the caller.
    pub fn prefix_sum<T: DeviceCopy + CpuRadixPartitionable>(
        &mut self,
        partition_attr: RadixPartitionInputChunk<'_, T>,
        partition_offsets: PartitionOffsetsMutSlice<'_, Tuple<T, T>>,
    ) -> Result<()> {
        T::prefix_sum_impl(self, partition_attr, partition_offsets)
    }

    /// Radix-partitions a relation by its key attribute.
    ///
    /// See the module-level documentation for details on the algorithm.
    pub fn partition<T: DeviceCopy + CpuRadixPartitionable>(
        &mut self,
        partition_attr: RadixPartitionInputChunk<'_, T>,
        payload_attr: RadixPartitionInputChunk<'_, T>,
        partition_offsets: PartitionOffsetsMutSlice<Tuple<T, T>>,
        partitioned_relation: PartitionedRelationMutSlice<Tuple<T, T>>,
    ) -> Result<()> {
        T::partition_impl(
            self,
            partition_attr,
            payload_attr,
            partition_offsets,
            partitioned_relation,
        )
    }
}

macro_rules! impl_cpu_radix_partition_for_type {
    ($Type:ty, $Suffix:expr) => {
        impl CpuRadixPartitionable for $Type {
            paste::item! {
                fn prefix_sum_impl(
                    rp: &mut CpuRadixPartitioner,
                    partition_attr: RadixPartitionInputChunk<'_, Self>,
                    mut partition_offsets: PartitionOffsetsMutSlice<'_, Tuple<Self, Self>>,
                    ) -> Result<()> {

                    let radix_bits = rp.radix_bits;
                    if partition_offsets.radix_bits != radix_bits {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionedRelation has mismatching radix bits".to_string(),
                                ))?;
                    }
                    if partition_attr.chunk_id != partition_offsets.chunk_id {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionOffsets has mismatching chunk ID".to_string(),
                                ))?;
                    }
                    if partition_attr.num_chunks != partition_offsets.chunks {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionOffsets has mismatching number of chunks".to_string(),
                                ))?;
                    }
                    if partition_offsets.padding_len() < WriteCombineBuffer::tuples_per_buffer::<Tuple<Self, Self>>() {
                        Err(ErrorKind::InvalidArgument(
                                "Padding is too small; should be at least the SWWC buffer size".to_string(),
                                ))?;
                    }

                    partition_offsets.set_data_len(partition_attr.total_data_len);

                    let (prefix_sum_fn, tmp_partition_offsets):
                        (
                            unsafe extern "C" fn(*mut PrefixSumArgs, u32, u32),
                            *mut u32
                        ) = match rp.prefix_sum_state
                    {
                        PrefixSumState::Chunked(ref mut state) =>
                            (
                                [<cpu_chunked_prefix_sum_ $Suffix>],
                                state.as_mut_ptr(),
                            ),
                        #[cfg(target_arch = "powerpc64")]
                        PrefixSumState::ChunkedSimd(ref mut state) =>
                            (
                                [<cpu_chunked_prefix_sum_simd_ $Suffix>],
                                state.as_mut_ptr(),
                            ),
                        #[cfg(not(target_arch = "powerpc64"))]
                        PrefixSumState::ChunkedSimd(_) =>
                            unimplemented!()
                    };

                    let mut args = PrefixSumArgs {
                        partition_attr: partition_attr.data.as_ptr() as *const c_void,
                        data_len: partition_attr.data.len(),
                        canonical_chunk_len: partition_attr.canonical_chunk_len,
                        padding_len: partition_offsets.padding_len(),
                        radix_bits,
                        ignore_bits: 0,
                        tmp_partition_offsets,
                        partition_offsets: partition_offsets.offsets.as_mut_ptr(),
                    };

                    unsafe {
                        prefix_sum_fn(&mut args, partition_offsets.chunk_id, partition_offsets.chunks);
                    }

                    Ok(())

                }

                fn partition_impl(
                    rp: &mut CpuRadixPartitioner,
                    partition_attr: RadixPartitionInputChunk<'_, Self>,
                    payload_attr: RadixPartitionInputChunk<'_, Self>,
                    partition_offsets: PartitionOffsetsMutSlice<Tuple<Self, Self>>,
                    mut partitioned_relation: PartitionedRelationMutSlice<Tuple<Self, Self>>,
                    ) -> Result<()>
                {
                    if partition_attr.data.len() != payload_attr.data.len() {
                        Err(ErrorKind::InvalidArgument(
                                "Partition and payload attributes have different sizes"
                                .to_string()
                            ))?;
                    }
                    if partitioned_relation.radix_bits != rp.radix_bits {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionedRelation has mismatching radix bits"
                                .to_string()
                            ))?;
                    }
                    if (partition_offsets.radix_bits != rp.radix_bits) {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionOffsets has mismatching radix bits".to_string(),
                                ))?;
                    }
                    if (partition_offsets.chunks != partitioned_relation.chunks) {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionOffsets and PartitionedRelation have mismatching chunks".to_string(),
                                ))?;
                    }
                    if (partition_offsets.padding_len() != partitioned_relation.padding_len()) {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionOffsets and PartitionedRelation have mismatching padding".to_string(),
                                ))?;
                    }

                    let data_len = partition_attr.data.len();
                    let (partition_fn, tmp_partition_offsets, write_combine_buffer):
                        (
                            unsafe extern "C" fn(*mut RadixPartitionArgs),
                            *mut u64,
                            *mut c_void,
                        ) = match rp.radix_partition_state
                    {
                        RadixPartitionState::NC(ref mut offsets) =>
                            (
                                [<cpu_chunked_radix_partition_ $Suffix _ $Suffix>],
                                offsets.as_mut_ptr(),
                                ptr::null_mut(),
                            ),
                        RadixPartitionState::Swwc(ref mut swwc) =>
                            (
                                [<cpu_chunked_radix_partition_swwc_ $Suffix _ $Suffix>],
                                ptr::null_mut(),
                                swwc.as_mut_slice().as_mut_ptr() as *mut c_void,
                            ),
                        #[cfg(target_arch = "powerpc64")]
                        RadixPartitionState::SwwcSimd(ref mut swwc) =>
                            (
                                [<cpu_chunked_radix_partition_swwc_simd_ $Suffix _ $Suffix>],
                                ptr::null_mut(),
                                swwc.as_mut_slice().as_mut_ptr() as *mut c_void,
                            ),
                        #[cfg(not(target_arch = "powerpc64"))]
                        RadixPartitionState::SwwcSimd(ref mut _swwc) =>
                            unimplemented!(),
                    };

                    let mut args = RadixPartitionArgs {
                        partition_attr_data: partition_attr.data.as_ptr() as *const c_void,
                        payload_attr_data: payload_attr.data.as_ptr() as *const c_void,
                        data_len,
                        padding_len: partitioned_relation.padding_len() as usize,
                        radix_bits: rp.radix_bits,
                        ignore_bits: 0,
                        partition_offsets: partition_offsets.offsets.as_ptr(),
                        tmp_partition_offsets,
                        write_combine_buffer,
                        partitioned_relation: partitioned_relation.relation
                            .as_mut_ptr() as *mut c_void,
                    };

                    unsafe {
                        partition_fn(
                            &mut args as *mut RadixPartitionArgs
                        );
                    }

                    // Copy offsets to PartitionedRelation.
                    unsafe {
                        partitioned_relation.offsets
                            .as_mut_slice()
                            .copy_from_slice(partition_offsets.offsets.as_slice());
                    }

                    Ok(())
                }
            }
        }
    };
}

impl_cpu_radix_partition_for_type!(i32, int32);
impl_cpu_radix_partition_for_type!(i64, int64);
