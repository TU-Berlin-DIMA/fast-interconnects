// Copyright 2020-2022 Clemens Lutz
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

/// Cache pad a value to avoid false sharing between threads.
///
/// Pads the value to 128 bytes, because Intel Sandy Bridge and later pre-fetch
/// two 64-byte cache lines, and IBM POWER processors have 128-byte cache lines.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
#[repr(align(128))]
pub struct CachePadded<T> {
    pub value: T,
}
