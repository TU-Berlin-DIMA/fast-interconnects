/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use super::{Benchmark, ItemBytes, MemoryOperation};
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
    pub threads: Option<ThreadCount>,
    pub grid_size: Option<Grid>,
    pub block_size: Option<Block>,
    pub ilp: Option<Ilp>,
    pub throttle_reasons: Option<String>,
    pub clock_rate_mhz: Option<u32>,
    pub memory_accesses: u64,
    pub cycles: Cycles,
    pub ns: u64,
}
