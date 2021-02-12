/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2021 Clemens Lutz, German Research Center for Artificial Intelligence
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use crate::constants;
use crate::error::{ErrorKind, Result};
use numa_gpu::runtime::memory::DerefMem;
use rustacuda::memory::DeviceCopy;
use std::mem;

/// Returns the reference chunk size with which input should be partitioned.
///
/// Note that this is an internal method.
pub(super) fn input_chunk_size<Key>(data_len: usize, num_chunks: u32) -> Result<usize> {
    let num_chunks_usize = num_chunks as usize;
    let input_align_mask = !(constants::ALIGN_BYTES as usize / mem::size_of::<Key>() - 1);
    let chunk_len = ((data_len + num_chunks_usize - 1) / num_chunks_usize) & input_align_mask;

    if chunk_len >= std::u32::MAX as usize {
        let msg = "Relation is too large and causes an integer overflow. Try using more chunks by setting a higher CUDA grid size";
        Err(ErrorKind::IntegerOverflow(msg.to_string()))?
    };

    Ok(chunk_len)
}

pub trait RadixPartitionInputChunkable {
    type Out;

    /// Splits the input into equally sized chunks.
    ///
    /// If necessary, the last chunk is shortened to not exceed the data length.
    fn input_chunks<'a, Key>(
        &'a self,
        num_chunks: u32,
    ) -> Result<Vec<RadixPartitionInputChunk<'a, Self::Out>>>;
}

/// A reference to a chunk of input data.
///
/// Effectively a slice with additional metadata specifying the referenced chunk.
#[derive(Debug)]
pub struct RadixPartitionInputChunk<'a, T: Sized> {
    pub data: &'a [T],
    pub canonical_chunk_len: usize,
    pub chunk_id: u32,
    pub num_chunks: u32,
    pub total_data_len: usize,
}

impl<T: Sized> RadixPartitionInputChunkable for [T] {
    type Out = T;

    fn input_chunks<Key>(
        &self,
        num_chunks: u32,
    ) -> Result<Vec<RadixPartitionInputChunk<'_, Self::Out>>> {
        let canonical_chunk_len = input_chunk_size::<Key>(self.len(), num_chunks)?;

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
                    total_data_len: self.len(),
                }
            })
            .collect();

        Ok(chunks)
    }
}

impl<T: Sized + DeviceCopy> RadixPartitionInputChunkable for DerefMem<T> {
    type Out = T;

    fn input_chunks<Key>(
        &self,
        num_chunks: u32,
    ) -> Result<Vec<RadixPartitionInputChunk<'_, Self::Out>>> {
        self.as_slice().input_chunks::<Key>(num_chunks)
    }
}
