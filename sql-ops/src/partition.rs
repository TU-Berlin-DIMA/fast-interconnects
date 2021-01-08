/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2021 Clemens Lutz, German Research Center for Artificial Intelligence
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use crate::error::{Error, ErrorKind};
use rustacuda::memory::DeviceCopy;
use std::convert::TryFrom;

pub mod cpu_radix_partition;
pub mod gpu_radix_partition;
mod partition_input_chunk;
mod partitioned_relation;

// Export structs
pub use partition_input_chunk::{RadixPartitionInputChunk, RadixPartitionInputChunkable};
pub use partitioned_relation::{
    PartitionOffsets, PartitionOffsetsChunksMut, PartitionOffsetsMutSlice, PartitionedRelation,
    PartitionedRelationChunksMut, PartitionedRelationMutSlice,
};

/// Histogram algorithm type
#[derive(Copy, Clone, Debug)]
pub enum HistogramAlgorithmType {
    /// A class of algorithms that split partitions over multiple chunks.
    ///
    /// Typically, each chunk is the result of a thread.
    Chunked,

    /// A class of algorithms that output contiguous partitions.
    ///
    /// The result is one single, contiguous sequence of partitions.  Thus, in multi-threaded
    /// implementations, threads cooperate to make the output contiguous.
    Contiguous,
}

/// Compute the fanout (i.e., the number of partitions) from the number of radix
/// bits.
fn fanout(radix_bits: u32) -> u32 {
    1 << radix_bits
}

/// A radix pass
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RadixPass {
    First = 0,
    Second,
    Thrird,
}

/// The number of radix bits per pass.
#[derive(Copy, Clone, Debug)]
pub struct RadixBits {
    pass_bits: [Option<u32>; 3],
}

impl RadixBits {
    /// Returns a new `RadixBits` object
    pub fn new(first_bits: Option<u32>, second_bits: Option<u32>, third_bits: Option<u32>) -> Self {
        let pass_bits = [first_bits, second_bits, third_bits];

        Self { pass_bits }
    }

    /// Total number of radix bits.
    pub fn radix_bits(&self) -> u32 {
        self.pass_bits
            .iter()
            .fold(0_u32, |sum, item| sum + item.unwrap_or(0))
    }

    /// Total fanout.
    pub fn fanout(&self) -> u32 {
        fanout(self.radix_bits())
    }

    /// Number of radix bits per pass.
    pub fn pass_radix_bits(&self, pass: RadixPass) -> Option<u32> {
        self.pass_bits[pass as usize]
    }

    /// Fanout per pass.
    pub fn pass_fanout(&self, pass: RadixPass) -> Option<u32> {
        let radix_bits = self.pass_bits[pass as usize];
        radix_bits.map(|bits| fanout(bits))
    }

    /// Number of radix bits by all earlier passes.
    ///
    /// For example, the first pass has 6 radix bits and the second pass has 6
    /// radix bits. Then `pass_ignore_bits(First)` equals 0, and
    /// `pass_ignore_bits(Second)` equals 6.
    pub fn pass_ignore_bits(&self, pass: RadixPass) -> u32 {
        self.pass_bits
            .iter()
            .take(pass as usize)
            .fold(0_u32, |sum, item| sum + item.unwrap_or(0))
    }
}

impl From<u32> for RadixBits {
    fn from(first_bits: u32) -> Self {
        Self::new(Some(first_bits), None, None)
    }
}

impl TryFrom<&[u32]> for RadixBits {
    type Error = Error;

    fn try_from(v: &[u32]) -> Result<Self, Self::Error> {
        if v.len() == 0 || v.len() > 3 {
            Err(ErrorKind::InvalidArgument(
                "At least one and at most three sets of radix bits required".to_string(),
            ))?;
        }

        Ok(Self::new(
            v.get(0).copied(),
            v.get(1).copied(),
            v.get(2).copied(),
        ))
    }
}

/// A key-value tuple.
///
/// The partitioned relation is stored as a collection of `Tuple<K, V>`.
///
/// Note that the struct's layout must be kept in sync with its counterpart in
/// C/C++.
#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct Tuple<Key: Sized, Value: Sized> {
    pub key: Key,
    pub value: Value,
}

unsafe impl<K, V> DeviceCopy for Tuple<K, V>
where
    K: DeviceCopy,
    V: DeviceCopy,
{
}
