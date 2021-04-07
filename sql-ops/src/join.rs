/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

pub mod cuda_radix_join;
mod hashing_scheme;
pub mod no_partitioning_join;

pub use hashing_scheme::HashingScheme;

/// A hash table entry in the C/C++ implementation.
///
/// Note that the struct's layout must be kept in sync with its counterpart in
/// C/C++.
pub type HtEntry<K, V> = crate::partition::Tuple<K, V>;
