// Copyright 2018-2022 Clemens Lutz
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

pub mod cuda_radix_join;
mod hashing_scheme;
pub mod no_partitioning_join;

pub use hashing_scheme::HashingScheme;

/// A hash table entry in the C/C++ implementation.
///
/// Note that the struct's layout must be kept in sync with its counterpart in
/// C/C++.
pub type HtEntry<K, V> = crate::partition::Tuple<K, V>;
