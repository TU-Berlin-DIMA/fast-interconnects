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

use crate::error::Result;
use crate::{ArgDeviceType, ArgMemType, ArgPageType, CmdTlbLatency};
use serde_derive::Serialize;

#[derive(Clone, Debug, Default, Serialize)]
pub struct DataPoint {
    pub hostname: String,
    pub device_type: Option<ArgDeviceType>,
    pub device_codename: Option<String>,
    pub device_id: Option<u16>,
    pub memory_type: Option<ArgMemType>,
    pub memory_location: Option<u16>,
    pub page_type: Option<ArgPageType>,
    pub range_bytes: Option<usize>,
    pub stride_bytes: Option<usize>,
    pub threads: Option<u32>,
    pub grid_size: Option<u32>,
    pub block_size: Option<u32>,
    pub iotlb_flush: Option<bool>,
    pub throttle_reasons: Option<String>,
    pub clock_rate_mhz: Option<u32>,
    pub cycle_counter_overhead_cycles: Option<u32>,
    pub stride_id: Option<usize>,
    pub tlb_status: Option<String>,
    pub index_bytes: Option<i64>,
    pub cycles: Option<u32>,
    pub ns: Option<u32>,
}

impl DataPoint {
    pub fn from_cmd_options(cmd: &CmdTlbLatency) -> Result<DataPoint> {
        let hostname = hostname::get()
            .map_err(|_| "Couldn't get hostname")?
            .into_string()
            .map_err(|_| "Couldn't convert hostname into UTF-8 string")?;

        let dp = DataPoint {
            hostname,
            device_type: Some(ArgDeviceType::GPU),
            device_id: Some(cmd.device_id),
            memory_type: Some(cmd.mem_type),
            memory_location: Some(cmd.mem_location),
            page_type: Some(cmd.page_type),
            ..DataPoint::default()
        };

        Ok(dp)
    }
}
