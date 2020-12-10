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

use super::fanout;
use super::gpu_radix_partition::GpuHistogramAlgorithm;
use crate::error::{ErrorKind, Result};
use numa_gpu::runtime::allocator::MemAllocFn;
use numa_gpu::runtime::memory::{LaunchableMem, LaunchableMutSlice, LaunchableSlice, Mem};
use rustacuda::memory::DeviceCopy;
use std::convert::TryInto;
use std::mem;
use std::ops::{Index, IndexMut};
use std::slice::{Chunks, ChunksMut};

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
    /// Creates a new partition offsets array.
    // TODO: remove dependency on GpuHistogramAlgorithm
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
    pub fn fanout(&self) -> u32 {
        fanout(self.radix_bits)
    }

    /// Returns the number of radix bits.
    pub fn radix_bits(&self) -> u32 {
        self.radix_bits
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
            - (num_partitions * self.num_chunks() as usize) * self.padding_len() as usize
    }

    /// Returs the number of chunks.
    pub fn num_chunks(&self) -> u32 {
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
    pub fn fanout(&self) -> u32 {
        fanout(self.radix_bits)
    }

    /// Returns the length of the requested partition.
    ///
    /// If the offsets are accessible by the CPU (i.e., in DerefMem), then the
    /// length is returned. Otherwise, the function returns an error.
    pub fn partition_len(&self, partition_id: u32) -> Result<usize> {
        let fanout = self.fanout();
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
        let fanout = self.fanout();
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
        let partitions = self.fanout();

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
