// Copyright 2021-2022 Clemens Lutz
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

use super::{Benchmark, ItemBytes, MemoryOperation, TileSize};
use crate::types::*;
use crate::ArgPageType;
use serde_derive::Serialize;

#[derive(Clone, Debug, Default, Serialize)]
pub(super) struct DataPoint {
    pub hostname: String,
    pub device_type: String,
    pub device_codename: Option<String>,
    pub benchmark: Option<Benchmark>,
    pub memory_operation: Option<MemoryOperation>,
    pub cpu_node: Option<u16>,
    pub memory_type: Option<BareMemType>,
    pub memory_node: Option<u16>,
    pub page_type: Option<ArgPageType>,
    pub warm_up: bool,
    pub range_bytes: usize,
    pub item_bytes: Option<ItemBytes>,
    pub tile_size: Option<TileSize>,
    pub threads: Option<ThreadCount>,
    pub grid_size: Option<Grid>,
    pub block_size: Option<Block>,
    pub warp_aligned: Option<bool>,
    pub throttle_reasons: Option<String>,
    pub clock_rate_mhz: Option<u32>,
    pub memory_accesses: u64,
    pub cycles: Cycles,
    pub ns: u64,
}
