/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019 Clemens Lutz, German Research Center for Artificial Intelligence
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

//! Definitions of hashing schemes for hash tables.

/// Specifies the hashing scheme using in hash table insert and probe operations.
#[derive(Clone, Copy, Debug)]
pub enum HashingScheme {
    /// Perfect hashing scheme.
    ///
    /// Perfect hashing assumes that build-side join keys are unique and in a
    /// contiguous range, i.e., k \in [0,N-1]. Probe-side keys are allowed to be
    /// non-unique and outside of the range.
    Perfect,

    /// Linear probing scheme.
    ///
    /// Linear probing makes no assumptions about the join key distribution.
    LinearProbing,

    /// Bucket chaining scheme.
    ///
    /// Bucket chaining makes no assumptions about the join key distribution.
    ///
    /// ## Optimizations
    ///
    /// - vectorized loads
    /// - static hash table entry assignment per thread
    /// - key compression (not implemented)
    /// - materialization using coalesced writes (not implemented)
    BucketChaining,
}
