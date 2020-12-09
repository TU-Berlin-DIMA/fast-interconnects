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

use super::{fanout, RadixBits, RadixPass, Tuple};
use crate::error::{ErrorKind, Result};
use crate::prefix_scan::{GpuPrefixScanState, GpuPrefixSum};
use numa_gpu::runtime::allocator::{Allocator, MemAllocFn, MemType};
use numa_gpu::runtime::memory::{
    LaunchableMem, LaunchableMutPtr, LaunchableMutSlice, LaunchablePtr, LaunchableSlice, Mem,
};
use rustacuda::context::CurrentContext;
use rustacuda::device::DeviceAttribute;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::{DeviceBuffer, DeviceCopy};
use rustacuda::module::Module;
use rustacuda::stream::Stream;
use rustacuda::{launch, launch_cooperative};
use std::cmp;
use std::convert::TryInto;
use std::ffi::{self, CString};
use std::mem;
use std::ops::{Index, IndexMut};
use std::slice::{Chunks, ChunksMut};

extern "C" {
    fn cpu_chunked_prefix_sum_int32(args: *mut PrefixSumArgs, chunk_id: u32, num_chunks: u32);
    fn cpu_chunked_prefix_sum_int64(args: *mut PrefixSumArgs, chunk_id: u32, num_chunks: u32);
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
    device_memory_buffers: LaunchableMutPtr<i8>,
    device_memory_buffer_bytes: u64,

    // Outputs
    partitioned_relation: LaunchableMutPtr<ffi::c_void>,
}

unsafe impl DeviceCopy for RadixPartitionArgs {}

pub trait RadixPartitionInputChunkable: Sized {
    type Out;

    fn input_chunks<'a, Key>(
        &'a self,
        radix_partitioner: &GpuRadixPartitioner,
    ) -> Result<Vec<RadixPartitionInputChunk<'a, Self::Out>>>;
}

pub trait GpuRadixPartitionable: Sized + DeviceCopy {
    fn cpu_prefix_sum_impl(
        rp: &GpuRadixPartitioner,
        pass: RadixPass,
        partition_attr: &RadixPartitionInputChunk<'_, Self>,
        partition_offsets: &mut PartitionOffsetsMutSlice<'_, Tuple<Self, Self>>,
    ) -> Result<()>;

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

    fn partition_impl(
        rp: &mut GpuRadixPartitioner,
        pass: RadixPass,
        partition_attr: LaunchableSlice<'_, Self>,
        payload_attr: LaunchableSlice<'_, Self>,
        partition_offsets: PartitionOffsets<Tuple<Self, Self>>,
        partitioned_relation: &mut PartitionedRelation<Tuple<Self, Self>>,
        stream: &Stream,
    ) -> Result<()>;
}

/// A reference to a chunk of input data.
///
/// Effectively a slice with additional metadata specifying the referenced chunk.
#[derive(Debug)]
pub struct RadixPartitionInputChunk<'a, T: Sized> {
    data: &'a [T],
    canonical_chunk_len: usize,
    chunk_id: u32,
    num_chunks: u32,
}

impl<T> RadixPartitionInputChunkable for &[T] {
    type Out = T;

    fn input_chunks<Key>(
        &self,
        radix_partitioner: &GpuRadixPartitioner,
    ) -> Result<Vec<RadixPartitionInputChunk<'_, Self::Out>>> {
        let num_chunks = radix_partitioner.grid_size.x;
        let canonical_chunk_len = radix_partitioner.input_chunk_size::<Key>(self.len())?;

        let chunks = (0..num_chunks)
            .map(|chunk_id| {
                let offset = canonical_chunk_len * chunk_id as usize;
                let actual_chunk_len = if chunk_id + 1 == num_chunks {
                    self.len() - offset
                } else {
                    canonical_chunk_len
                };
                let data = &self[offset..(offset + actual_chunk_len)];

                RadixPartitionInputChunk {
                    data,
                    canonical_chunk_len,
                    chunk_id,
                    num_chunks,
                }
            })
            .collect();

        Ok(chunks)
    }
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
/// ```ignore
/// C0.P0 | C0.P1 | ... | C0.PN | C1.P0 | C1.P1 | ... | C1.PN | ... | CM.PN
/// ```
///
/// Note that no restrictions are placed on the data layout (i.e., offsets may
/// be numerically unordered).
///
/// # `offsets` vs. `local_offsets`
///
/// Chunked prefix sums inherently have local offsets for each chunk. In
/// contrast, contiguous histograms only have one set of "global" offsets.
///
/// The partitioning operator requires a set of local offsets for each chunk.
/// Thus, the contiguous prefix sum variants output `local_offsets` in addition
/// to the normal `offsets`.
///
/// # Invariants
///
///  - `radix_bits` must match in `GpuRadixPartitioner`
///  - `max_chunks` must equal the maximum number of chunks computed at runtime
///     (e.g., the grid size)
#[derive(Debug)]
pub struct PartitionOffsets<T: DeviceCopy> {
    pub offsets: Mem<u64>,
    pub local_offsets: Option<Mem<u64>>,
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

        let num_partitions = fanout(radix_bits) as usize;
        let offsets = alloc_fn(num_partitions * chunks as usize);

        let local_offsets = match histogram_algorithm {
            GpuHistogramAlgorithm::GpuContiguous => {
                Some(alloc_fn(num_partitions * max_chunks as usize))
            }
            _ => None,
        };

        Self {
            offsets,
            local_offsets,
            chunks,
            radix_bits,
            phantom_data: std::marker::PhantomData,
        }
    }

    /// Returs the number of chunks.
    pub fn chunks(&self) -> u32 {
        self.chunks
    }

    /// Returns an iterator over the chunks contained inside the offsets.
    ///
    /// Chunks are non-overlapping and can safely be used for parallel
    /// processing.
    pub fn chunks_mut(&mut self) -> PartitionOffsetsChunksMut<'_, T> {
        PartitionOffsetsChunksMut::new(self)
    }

    /// Returns the number of partitions.
    pub fn partitions(&self) -> u32 {
        fanout(self.radix_bits)
    }

    /// Returns the number of padding elements per partition.
    pub fn padding_len(&self) -> u32 {
        super::GPU_PADDING_BYTES / mem::size_of::<T>() as u32
    }
}

/// An iterator that generates `PartitionOffsetsMutSlice`.
#[derive(Debug)]
pub struct PartitionOffsetsChunksMut<'a, T: DeviceCopy> {
    // Note: unsafe slices, must convert back to LaunchableSlice
    offsets_chunks: ChunksMut<'a, u64>,
    chunk_id: u32,
    chunks: u32,
    radix_bits: u32,
    phantom_data: std::marker::PhantomData<T>,
}

impl<'a, T: DeviceCopy> PartitionOffsetsChunksMut<'a, T> {
    fn new(offsets: &'a mut PartitionOffsets<T>) -> Self {
        unsafe {
            let num_partitions = fanout(offsets.radix_bits) as usize;
            let offsets_chunks = offsets
                .offsets
                .as_launchable_mut_slice()
                .as_mut_slice()
                .chunks_mut(num_partitions);

            Self {
                offsets_chunks,
                chunk_id: 0,
                chunks: offsets.chunks,
                radix_bits: offsets.radix_bits,
                phantom_data: std::marker::PhantomData,
            }
        }
    }
}

impl<'a, T: DeviceCopy> Iterator for PartitionOffsetsChunksMut<'a, T> {
    type Item = PartitionOffsetsMutSlice<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.offsets_chunks.next().and_then(|o| {
            let chunk_id = self.chunk_id;
            self.chunk_id = self.chunk_id + 1;

            Some(PartitionOffsetsMutSlice {
                offsets: o.as_launchable_mut_slice(),
                chunk_id,
                chunks: self.chunks,
                radix_bits: self.radix_bits,
                phantom_data: std::marker::PhantomData,
            })
        })
    }
}

/// A mutable slice that references the `PartitionOffsets` of one partition.
///
/// Effectively a mutable slice containing additional metadata about the chunk.
/// The purpose is to allow thread-safe writes to `PartitionOffsets`.
#[derive(Debug)]
pub struct PartitionOffsetsMutSlice<'a, T: DeviceCopy> {
    pub(crate) offsets: LaunchableMutSlice<'a, u64>,
    pub(crate) chunk_id: u32,
    pub(crate) chunks: u32,
    pub(crate) radix_bits: u32,
    phantom_data: std::marker::PhantomData<T>,
}

impl<'a, T: DeviceCopy> PartitionOffsetsMutSlice<'a, T> {
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
    pub(crate) relation: Mem<T>,
    pub(crate) offsets: Mem<u64>,
    pub(crate) chunks: u32,
    pub(crate) radix_bits: u32,
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
        let num_partitions = fanout(radix_bits) as usize;
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
        let num_partitions = fanout(self.radix_bits) as usize;

        self.relation.len()
            - (num_partitions * self.chunks() as usize) * self.padding_len() as usize
    }

    /// Returs the number of chunks.
    pub fn chunks(&self) -> u32 {
        self.chunks
    }

    /// Returns an iterator over the chunks contained inside the relation.
    ///
    /// Chunks are non-overlapping and can safely be used for parallel
    /// processing.
    pub fn chunks_mut(&mut self) -> PartitionedRelationChunksMut<'_, T> {
        PartitionedRelationChunksMut::new(self)
    }

    /// Returns the number of partitions.
    pub fn partitions(&self) -> u32 {
        fanout(self.radix_bits)
    }

    /// Returns the length of the requested partition.
    ///
    /// If the offsets are accessible by the CPU (i.e., in DerefMem), then the
    /// length is returned. Otherwise, the function returns an error.
    pub fn partition_len(&self, partition_id: u32) -> Result<usize> {
        let fanout = self.partitions();
        if partition_id >= fanout {
            Err(ErrorKind::InvalidArgument(
                "Invalid partition ID".to_string(),
            ))?;
        }

        let offsets: &[u64] = match (&self.offsets).try_into() {
            Ok(offsets) => offsets,
            _ => Err(ErrorKind::RuntimeError(
                "Trying to dereference device memory!".to_string(),
            ))?,
        };
        let padding_len = self.padding_len() as usize;

        let len = (0..self.chunks)
            .map(|chunk_id| {
                let ofi = chunk_id as usize * fanout as usize + partition_id as usize;
                let begin = offsets[ofi] as usize;
                let end = if ofi + 1 < offsets.len() {
                    offsets[ofi + 1] as usize - padding_len
                } else {
                    self.relation.len()
                };
                end - begin
            })
            .sum();

        Ok(len)
    }

    /// Returns the number of padding elements per partition.
    pub(crate) fn padding_len(&self) -> u32 {
        super::GPU_PADDING_BYTES / mem::size_of::<T>() as u32
    }

    /// Returns the internal representation of the relation data as a slice.
    ///
    /// This function is intended for unit testing. Use the methods provided by
    /// the `Index` trait or `chunks_mut()` instead if possible.
    ///
    /// The function is unsafe because:
    /// - the internal representation may change
    /// - padding may contain uninitialized memory.
    pub unsafe fn as_raw_relation_slice(&self) -> Result<&[T]> {
        let relation: &[_] = (&self.relation).try_into().map_err(|_| {
            ErrorKind::Msg("Tried to convert device memory into host slice".to_string())
        })?;

        Ok(relation)
    }
}

/// Returns the specified chunk and partition as a subslice of the relation.
impl<T: DeviceCopy> Index<(u32, u32)> for PartitionedRelation<T> {
    type Output = [T];

    fn index(&self, (chunk_id, partition_id): (u32, u32)) -> &Self::Output {
        let fanout = self.partitions();
        if partition_id >= fanout {
            panic!("Invalid partition ID".to_string(),);
        }
        if chunk_id >= self.chunks {
            panic!("Invalid chunk ID".to_string(),);
        }

        let (offsets, relation): (&[u64], &[T]) =
            match ((&self.offsets).try_into(), (&self.relation).try_into()) {
                (Ok(offsets), Ok(relation)) => (offsets, relation),
                _ => panic!("Trying to dereference device memory!"),
            };

        let ofi = (chunk_id * fanout + partition_id) as usize;
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
impl<T: DeviceCopy> IndexMut<(u32, u32)> for PartitionedRelation<T> {
    fn index_mut(&mut self, (chunk_id, partition_id): (u32, u32)) -> &mut Self::Output {
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

        let ofi = (chunk_id * partitions + partition_id) as usize;
        let begin = offsets[ofi] as usize;
        let end = if ofi + 1 < offsets_len {
            offsets[ofi + 1] as usize - padding_len as usize
        } else {
            relation.len()
        };

        &mut relation[begin..end]
    }
}

/// An iterator that generates `PartitionedRelationMutSlice`.
#[derive(Debug)]
pub struct PartitionedRelationChunksMut<'a, T: DeviceCopy> {
    // Note: unsafe slices, must convert back to LaunchableMutSlice
    relation_chunks: ChunksMut<'a, T>,
    // Note: unsafe slices, must convert back to LaunchableSlice
    offsets_chunks: Chunks<'a, u64>,
    chunk_id: u32,
    chunks: u32,
    radix_bits: u32,
}

impl<'a, T: DeviceCopy> PartitionedRelationChunksMut<'a, T> {
    /// Creates a new chunk iterator for a `PartitionedRelation`.
    fn new(rel: &'a mut PartitionedRelation<T>) -> Self {
        unsafe {
            let rel_chunks = rel.chunks as usize;
            let rel_chunk_size = (rel.relation.len() + rel_chunks - 1) / rel_chunks;
            let off_chunk_size = rel.offsets.len() / rel_chunks;

            let relation_chunks = rel
                .relation
                .as_launchable_mut_slice()
                .as_mut_slice()
                .chunks_mut(rel_chunk_size);
            let offsets_chunks = rel
                .offsets
                .as_launchable_slice()
                .as_slice()
                .chunks(off_chunk_size);

            Self {
                relation_chunks,
                offsets_chunks,
                chunk_id: 0,
                chunks: rel.chunks,
                radix_bits: rel.radix_bits,
            }
        }
    }
}

impl<'a, T: DeviceCopy> Iterator for PartitionedRelationChunksMut<'a, T> {
    type Item = PartitionedRelationMutSlice<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<PartitionedRelationMutSlice<'a, T>> {
        let zipped = self
            .relation_chunks
            .next()
            .and_then(|rel| self.offsets_chunks.next().and_then(|off| Some((rel, off))));
        zipped.and_then(|(r, o)| {
            let chunk_id = self.chunk_id;
            self.chunk_id = self.chunk_id + 1;

            Some(PartitionedRelationMutSlice {
                relation: r.as_launchable_mut_slice(),
                offsets: o.as_launchable_slice(),
                chunk_id,
                chunks: self.chunks,
                radix_bits: self.radix_bits,
            })
        })
    }
}

/// A mutable slice that references part of a `PartitionedRelation`.
///
/// Effectively a mutable slice containing additional metadata about the chunk.
/// The purpose is to allow thread-safe writes to a `PartitionedRelation`.
#[derive(Debug)]
pub struct PartitionedRelationMutSlice<'a, T> {
    pub relation: LaunchableMutSlice<'a, T>,
    pub offsets: LaunchableSlice<'a, u64>,
    pub chunk_id: u32,
    pub chunks: u32,
    pub radix_bits: u32,
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
    GpuContiguous(Mem<GpuPrefixScanState<u64>>),
}

#[derive(Debug)]
pub struct GpuRadixPartitioner {
    radix_bits: RadixBits,
    log2_num_banks: u32,
    prefix_sum_algorithm: GpuHistogramAlgorithm,
    partition_algorithm: GpuRadixPartitionAlgorithm,
    prefix_sum_state: PrefixSumState,
    module: Module,
    grid_size: GridSize,
    block_size: BlockSize,
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
        let log2_num_banks = env!("LOG2_NUM_BANKS")
            .parse::<u32>()
            .expect("Failed to parse \"log2_num_banks\" string to an integer");
        let prefix_scan_state_len = GpuPrefixSum::state_len(grid_size.clone(), block_size.clone())?;

        let prefix_sum_state = match prefix_sum_algorithm {
            GpuHistogramAlgorithm::CpuChunked => PrefixSumState::CpuChunked,
            GpuHistogramAlgorithm::GpuChunked => PrefixSumState::GpuChunked,
            GpuHistogramAlgorithm::GpuContiguous => PrefixSumState::GpuContiguous(
                Allocator::alloc_mem(MemType::CudaDevMem, prefix_scan_state_len),
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
    /// Currently only the `GpuContiguous` histogram algorithm is supported.
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
    /// Currently only the `GpuContiguous` histogram algorithm is supported. See
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

    /// Computes the prefix sum on the CPU.
    ///
    /// The purpose is to avoid transferring the key attribute twice the GPU
    /// across the interconnect. Although the CPU may be slower at shuffling the
    /// data, computing a histogram is light-weight and should be close to the
    /// memory bandwidth. See Polychroniou et al. "A comprehensive study of
    /// main-memory partitioning and its application to large-scale comparison-
    /// and radix-sort:
    ///
    /// ## Result
    ///
    /// The result is identical to `prefix_sum` on the GPU.
    ///
    /// ## Parallelism
    ///
    /// The function is sequential and is meant to be externally parallelized by
    /// the caller. Parallel calls to the function are thread-safe. One
    /// `GpuRadixPartitioner` instance should be shared among all threads.
    pub fn cpu_prefix_sum<T: DeviceCopy + GpuRadixPartitionable>(
        &self,
        pass: RadixPass,
        partition_attr: &RadixPartitionInputChunk<'_, T>,
        partition_offsets: &mut PartitionOffsetsMutSlice<'_, Tuple<T, T>>,
    ) -> Result<()> {
        T::cpu_prefix_sum_impl(self, pass, partition_attr, partition_offsets)
    }

    /// Radix-partitions a relation by its key attribute.
    ///
    /// See the module-level documentation for details on the algorithm.
    pub fn partition<T: DeviceCopy + GpuRadixPartitionable>(
        &mut self,
        pass: RadixPass,
        partition_attr: LaunchableSlice<'_, T>,
        payload_attr: LaunchableSlice<'_, T>,
        partition_offsets: PartitionOffsets<Tuple<T, T>>,
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

    /// Returns the reference chunk size with which input should be partitioned.
    pub fn input_chunk_size<Key>(&self, len: usize) -> Result<usize> {
        let num_chunks = self.grid_size.x as usize;
        let input_align_mask = !(super::ALIGN_BYTES as usize / mem::size_of::<Key>() - 1);
        let chunk_len = ((len + num_chunks - 1) / num_chunks) & input_align_mask;

        if chunk_len >= std::u32::MAX as usize {
            let msg = "Relation is too large and causes an integer overflow. Try using more chunks by setting a higher CUDA grid size";
            Err(ErrorKind::IntegerOverflow(msg.to_string()))?
        };

        Ok(chunk_len)
    }
}

macro_rules! impl_gpu_radix_partition_for_type {
    ($Type:ty, $Suffix:expr) => {
        impl GpuRadixPartitionable for $Type {
            paste::item! {
                fn cpu_prefix_sum_impl(
                    rp: &GpuRadixPartitioner,
                    pass: RadixPass,
                    partition_attr: &RadixPartitionInputChunk<'_, $Type>,
                    partition_offsets: &mut PartitionOffsetsMutSlice<'_, Tuple<$Type, $Type>>,
                    ) -> Result<()> {

                    let radix_bits = rp
                        .radix_bits
                        .pass_radix_bits(pass)
                        .ok_or_else(||
                                ErrorKind::InvalidArgument(
                                    "The requested partitioning pass is not specified".to_string()
                                    ))?;
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
                        dbg!(partition_attr.num_chunks);
                        dbg!(partition_offsets.chunks);
                        Err(ErrorKind::InvalidArgument(
                                "PartitionOffsets has mismatching number of chunks".to_string(),
                                ))?;
                    }
                    match rp.prefix_sum_algorithm {
                        GpuHistogramAlgorithm::CpuChunked => {},
                        _ => {
                                Err(ErrorKind::InvalidArgument(
                                        "Unsuppored option, use prefix_sum() instead".to_string(),
                                        ))?;
                            },
                    }

                    let mut args = PrefixSumArgs {
                        partition_attr: partition_attr.data.as_launchable_slice().as_launchable_ptr().as_void(),
                        data_len: partition_attr.data.len(),
                        canonical_chunk_len: partition_attr.canonical_chunk_len,
                        padding_len: partition_offsets.padding_len(),
                        radix_bits,
                        ignore_bits: rp.radix_bits.pass_ignore_bits(pass),
                        prefix_scan_state: LaunchableMutPtr::null_mut(),
                        tmp_partition_offsets: LaunchableMutPtr::null_mut(),
                        partition_offsets: partition_offsets.offsets.as_launchable_mut_ptr(),
                    };

                    match rp.prefix_sum_state {
                        PrefixSumState::CpuChunked => {
                            unsafe {
                                [<cpu_chunked_prefix_sum_ $Suffix>](&mut args, partition_offsets.chunk_id, rp.grid_size.x);
                            }
                        },
                        _ => unreachable!(),
                    }

                    Ok(())
                }

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

                    if partition_offsets.radix_bits != radix_bits {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionedRelation has mismatching radix bits".to_string(),
                                ))?;
                    }
                    match rp.prefix_sum_algorithm {
                        GpuHistogramAlgorithm::CpuChunked
                            => {
                                Err(ErrorKind::InvalidArgument(
                                        "Unsupported option, use cpu_prefix_sum() instead".to_string(),
                                        ))?;
                            },
                        GpuHistogramAlgorithm::GpuChunked
                            => if partition_offsets.chunks() != rp.grid_size.x {
                                Err(ErrorKind::InvalidArgument(
                                        "PartitionedRelation has mismatching number of chunks".to_string(),
                                        ))?;
                            },
                        GpuHistogramAlgorithm::GpuContiguous
                            => {
                                if partition_offsets.chunks() != 1 {
                                    Err(ErrorKind::InvalidArgument(
                                            "PartitionedRelation has mismatching number of chunks".to_string(),
                                            ))?;
                                }
                                if sm_count < partition_offsets.chunks() {
                                    Err(ErrorKind::InvalidArgument(
                                            "The GpuContiguous algorithm requires \
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

                    let module = &rp.module;
                    let max_shared_mem_bytes =
                        device.get_attribute(DeviceAttribute::MaxSharedMemPerBlockOptin)? as u32;
                    let fanout_u32 = rp.radix_bits.pass_fanout(pass).unwrap();
                    let ignore_bits = rp.radix_bits.pass_ignore_bits(pass);
                    let grid_size = rp.grid_size.clone();
                    let block_size = rp.block_size.clone();
                    let canonical_chunk_len = rp.input_chunk_size::<$Type>(partition_attr.len())?;

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
                        PrefixSumState::CpuChunked => unreachable!(),
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
                        PrefixSumState::GpuContiguous(ref mut prefix_scan_state) => {
                            let shared_mem_bytes = (
                                (block_size.x + (block_size.x >> rp.log2_num_banks)) + fanout_u32
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

                    if partition_offsets.radix_bits != radix_bits {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionOffsets has mismatching radix bits".to_string(),
                                ))?;
                    }
                    match rp.prefix_sum_algorithm {
                        GpuHistogramAlgorithm::CpuChunked | GpuHistogramAlgorithm::GpuChunked
                            => {
                                Err(ErrorKind::InvalidArgument(
                                        "Unsupported histogram algorithm, try using GpuContiguous instead".to_string(),
                                        ))?;
                            },
                        GpuHistogramAlgorithm::GpuContiguous
                            => {
                                if partition_offsets.chunks() != 1 {
                                    Err(ErrorKind::InvalidArgument(
                                            "PartitionOffsets has mismatching number of chunks".to_string(),
                                            ))?;
                                }
                                if sm_count < partition_offsets.chunks() {
                                    Err(ErrorKind::InvalidArgument(
                                            "The GpuContiguous algorithm requires \
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

                    let module = &rp.module;
                    let max_shared_mem_bytes =
                        device.get_attribute(DeviceAttribute::MaxSharedMemPerBlockOptin)? as u32;
                    let fanout_u32 = rp.radix_bits.pass_fanout(pass).unwrap();
                    let ignore_bits = rp.radix_bits.pass_ignore_bits(pass);
                    let grid_size = rp.grid_size.clone();
                    let block_size = rp.block_size.clone();
                    let canonical_chunk_len = rp.input_chunk_size::<$Type>(src_partition_attr.len())?;

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
                        PrefixSumState::CpuChunked => unreachable!(),
                        PrefixSumState::GpuChunked => unreachable!(),
                        PrefixSumState::GpuContiguous(ref mut prefix_scan_state) => {
                            let shared_mem_bytes = (
                                (block_size.x + (block_size.x >> rp.log2_num_banks)) + fanout_u32
                                ) * mem::size_of::<u64>() as u32;
                            assert!(
                                shared_mem_bytes <= max_shared_mem_bytes,
                                "Failed to allocate enough shared memory"
                                );

                            args.prefix_scan_state = prefix_scan_state.as_launchable_mut_ptr();

                            unsafe {
                                launch_cooperative!(
                                    module.[<gpu_contiguous_prefix_sum_and_copy_with_payload_ $Suffix _ $Suffix>]<<<
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

                    if partition_id >= src_relation.partitions() {
                        Err(ErrorKind::InvalidArgument(
                                "Invalid partition ID".to_string(),
                                ))?;
                    }
                    if partition_offsets.radix_bits != radix_bits {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionedRelation has mismatching radix bits".to_string(),
                                ))?;
                    }
                    match rp.prefix_sum_algorithm {
                        GpuHistogramAlgorithm::CpuChunked | GpuHistogramAlgorithm::GpuChunked
                            => {
                                Err(ErrorKind::InvalidArgument(
                                        "Unsupported histogram algorithm, try using GpuContiguous instead".to_string(),
                                        ))?;
                            },
                        GpuHistogramAlgorithm::GpuContiguous
                            => {
                                if partition_offsets.chunks() != 1 {
                                    Err(ErrorKind::InvalidArgument(
                                            "PartitionedRelation has mismatching number of chunks".to_string(),
                                            ))?;
                                }
                                if sm_count < partition_offsets.chunks() {
                                    Err(ErrorKind::InvalidArgument(
                                            "The GpuContiguous algorithm requires \
                                            all threads to run simultaneously. Try \
                                            decreasing the grid size.".to_string(),
                                            ))?;
                                }
                            }
                    }

                    // Only checking for contiguous partitioned relations, as
                    // chunked aren't supported.
                    if src_relation.relation.len() >= std::u32::MAX as usize {
                            let msg = "Relation is too large and causes an integer overflow.";
                            Err(ErrorKind::IntegerOverflow(msg.to_string(),))?
                    }

                    let module = &rp.module;
                    let max_shared_mem_bytes =
                        device.get_attribute(DeviceAttribute::MaxSharedMemPerBlockOptin)? as u32;
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
                        src_chunks: src_relation.chunks,
                        src_radix_bits: src_relation.radix_bits,
                        data_len: src_relation.relation.len(),
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
                        PrefixSumState::CpuChunked => unreachable!(),
                        PrefixSumState::GpuChunked => unreachable!(),
                        PrefixSumState::GpuContiguous(ref mut prefix_scan_state) => {
                            let shared_mem_bytes = (
                                (block_size.x + (block_size.x >> rp.log2_num_banks))
                                + fanout_u32
                                + 2 * src_relation.chunks
                                ) * mem::size_of::<u64>() as u32;
                            assert!(
                                shared_mem_bytes <= max_shared_mem_bytes,
                                "Failed to allocate enough shared memory"
                                );

                            args.prefix_scan_state = prefix_scan_state.as_launchable_mut_ptr();

                            unsafe {
                                launch_cooperative!(
                                    module.[<gpu_contiguous_prefix_sum_and_transform_ $Suffix _ $Suffix>]<<<
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
                    pass: RadixPass,
                    partition_attr: LaunchableSlice<'_, $Type>,
                    payload_attr: LaunchableSlice<'_, $Type>,
                    partition_offsets: PartitionOffsets<Tuple<$Type, $Type>>,
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
                    if partitioned_relation.radix_bits != radix_bits {
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
                    if (partition_offsets.radix_bits != radix_bits) {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionOffsets has mismatching radix bits".to_string(),
                                ))?;
                    }

                    let module = &rp.module;
                    let grid_size = rp.grid_size.clone();
                    let device = CurrentContext::get_device()?;
                    let warp_size = device.get_attribute(DeviceAttribute::WarpSize)? as u32;
                    let max_shared_mem_bytes =
                        device.get_attribute(DeviceAttribute::MaxSharedMemPerBlockOptin)? as u32;
                    let fanout_u32 = rp.radix_bits.pass_fanout(pass).unwrap();
                    let ignore_bits = rp.radix_bits.pass_ignore_bits(pass);
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

                    let dmem_buffer_bytes_per_block = match rp.partition_algorithm {
                        GpuRadixPartitionAlgorithm::HSSWWC
                            | GpuRadixPartitionAlgorithm::HSSWWCv2
                            | GpuRadixPartitionAlgorithm::HSSWWCv3 => {
                                Some(rp.dmem_buffer_bytes as u64 * fanout_u32 as u64)
                            },
                        GpuRadixPartitionAlgorithm::HSSWWCv4 => {
                            let warps_per_block = block_size.x / warp_size;
                            Some(rp.dmem_buffer_bytes as u64 * (fanout_u32 + warps_per_block) as u64)
                        },
                        _ => None
                    };

                    let mut dmem_buffer = if let Some(b) = dmem_buffer_bytes_per_block {
                        let global_dmem_buffer_bytes = b * grid_size.x as u64;
                        Some(Mem::CudaDevMem(unsafe { DeviceBuffer::uninitialized(global_dmem_buffer_bytes as usize)? }))
                    } else {
                        None
                    };

                    let partition_offsets_ptr = if let Some(ref local_offsets) = partition_offsets.local_offsets {
                        local_offsets.as_launchable_ptr()
                    } else {
                        partition_offsets.offsets.as_launchable_ptr()
                    };

                    let args = RadixPartitionArgs {
                        partition_attr_data: partition_attr.as_launchable_ptr().as_void(),
                        payload_attr_data: payload_attr.as_launchable_ptr().as_void(),
                        data_len,
                        padding_len: partitioned_relation.padding_len(),
                        radix_bits,
                        ignore_bits,
                        partition_offsets: partition_offsets_ptr,
                        device_memory_buffers: dmem_buffer
                            .as_mut()
                            .map_or(
                                LaunchableMutPtr::null_mut(),
                                |b| b.as_launchable_mut_ptr()
                                ),
                        device_memory_buffer_bytes: dmem_buffer_bytes_per_block.unwrap_or(0),
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
