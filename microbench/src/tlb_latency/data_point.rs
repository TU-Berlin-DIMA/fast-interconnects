/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2020 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use crate::error::Result;
use crate::{ArgDeviceType, ArgMemType, CmdTlbLatency};
use serde_derive::Serialize;

#[derive(Clone, Debug, Default, Serialize)]
pub struct DataPoint {
    pub hostname: String,
    pub device_type: Option<ArgDeviceType>,
    pub device_codename: Option<String>,
    pub device_id: Option<u16>,
    pub memory_type: Option<ArgMemType>,
    pub memory_location: Option<u16>,
    pub huge_pages: Option<bool>,
    pub range_bytes: Option<usize>,
    pub stride_bytes: Option<usize>,
    pub threads: Option<u32>,
    pub grid_size: Option<u32>,
    pub block_size: Option<u32>,
    pub throttle_reasons: Option<String>,
    pub clock_rate_mhz: Option<u32>,
    pub cycle_counter_overhead_cycles: Option<u32>,
    pub index_bytes: Option<u64>,
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
            huge_pages: cmd.huge_pages,
            ..DataPoint::default()
        };

        Ok(dp)
    }
}
