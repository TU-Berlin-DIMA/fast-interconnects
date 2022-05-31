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
