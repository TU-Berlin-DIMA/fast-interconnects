/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2020 Clemens Lutz, German Research Center for Artificial Intelligence
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use rustacuda::memory::DeviceCopy;

pub mod cpu_radix_partition;
pub mod gpu_radix_partition;

/// Defines the alignment of each partition in bytes.
///
/// Typically, alignment should be a multiple of the cache line size. Reasons
/// for this size are:
///
/// 1. Non-temporal store instructions
/// 2. Vector load and store intructions
/// 3. Coalesced loads and stores on GPUs
const ALIGN_BYTES: u32 = 128;

/// Defines the padding bytes between partitions.
///
/// Padding is necessary for partitioning algorithms to align writes. Aligned writes have fixed
/// length and may overwrite the padding space in front of their partition.  For this reason,
/// also the first partition includes padding in front.
///
/// Note that the padding length must be equal to or larger than the alignment.
const GPU_PADDING_BYTES: u32 = 128;

/// Compute the fanout (i.e., the number of partitions) from the number of radix
/// bits.
fn fanout(radix_bits: u32) -> usize {
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
    pub fn fanout(&self) -> usize {
        fanout(self.radix_bits())
    }

    /// Number of radix bits per pass.
    pub fn pass_radix_bits(&self, pass: RadixPass) -> Option<u32> {
        self.pass_bits[pass as usize]
    }

    /// Fanout per pass.
    pub fn pass_fanout(&self, pass: RadixPass) -> Option<usize> {
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
