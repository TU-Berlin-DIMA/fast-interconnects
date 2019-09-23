/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use super::hash_join_bench::HashJoinBench;
use crate::types::*;
use crate::CmdOpt;
use numa_gpu::error::Result;
use numa_gpu::runtime::hw_info::cpu_codename;
use rustacuda::device::Device;
use rustacuda::memory::DeviceCopy;
use serde::Serializer;
use serde_derive::Serialize;
use std::mem::size_of;
use std::string::ToString;
use std::time::Duration;

#[derive(Clone, Debug, Default, Serialize)]
pub struct DataPoint {
    pub data_set: Option<String>,
    pub hostname: String,
    pub execution_method: Option<ArgExecutionMethod>,
    #[serde(serialize_with = "serialize_vec")]
    pub device_codename: Option<Vec<String>>,
    pub transfer_strategy: Option<ArgTransferStrategy>,
    pub chunk_bytes: Option<usize>,
    pub threads: Option<usize>,
    pub hashing_scheme: Option<ArgHashingScheme>,
    pub hash_table_memory_type: Option<ArgMemType>,
    #[serde(serialize_with = "serialize_vec")]
    pub hash_table_memory_location: Option<Vec<u16>>,
    #[serde(serialize_with = "serialize_vec")]
    pub hash_table_proportions: Option<Vec<usize>>,
    pub hash_table_bytes: Option<usize>,
    pub tuple_bytes: Option<ArgTupleBytes>,
    pub relation_memory_type: Option<ArgMemType>,
    pub inner_relation_memory_location: Option<u16>,
    pub outer_relation_memory_location: Option<u16>,
    pub build_tuples: Option<usize>,
    pub build_bytes: Option<usize>,
    pub probe_tuples: Option<usize>,
    pub probe_bytes: Option<usize>,
    pub warm_up: Option<bool>,
    pub build_ns: Option<f64>,
    pub probe_ns: Option<f64>,
    pub build_warm_up_ns: Option<f64>,
    pub probe_warm_up_ns: Option<f64>,
    pub build_copy_ns: Option<f64>,
    pub probe_copy_ns: Option<f64>,
    pub build_compute_ns: Option<f64>,
    pub probe_compute_ns: Option<f64>,
    pub build_cool_down_ns: Option<f64>,
    pub probe_cool_down_ns: Option<f64>,
    pub hash_table_malloc_ns: Option<f64>,
    pub relation_malloc_ns: Option<f64>,
    pub relation_gen_ns: Option<f64>,
}

impl DataPoint {
    pub fn new() -> Result<DataPoint> {
        let hostname = hostname::get_hostname().ok_or_else(|| "Couldn't get hostname")?;

        let dp = DataPoint {
            hostname,
            ..DataPoint::default()
        };

        Ok(dp)
    }

    pub fn fill_from_cmd_options(&self, cmd: &CmdOpt) -> Result<DataPoint> {
        let mut sorted_ht_location = cmd.hash_table_location.clone();
        sorted_ht_location.sort();

        let mut ht_prop_loc: Vec<_> = cmd
            .hash_table_proportions
            .iter()
            .zip(cmd.hash_table_location.iter())
            .collect();
        ht_prop_loc.sort_by_key(|(_, &key)| key);
        let sorted_ht_proportions: Vec<_> = ht_prop_loc.iter().map(|(&val, _)| val).collect();

        // Get device information
        let dev_codename_str = match cmd.execution_method {
            ArgExecutionMethod::Cpu => vec![cpu_codename()],
            ArgExecutionMethod::Gpu | ArgExecutionMethod::GpuStream => {
                let device = Device::get_device(cmd.device_id.into())?;
                vec![device.name()?]
            }
            ArgExecutionMethod::Het => {
                let device = Device::get_device(cmd.device_id.into())?;
                vec![cpu_codename(), device.name()?]
            }
        };

        let dp = DataPoint {
            data_set: Some(cmd.data_set.to_string()),
            execution_method: Some(cmd.execution_method),
            device_codename: Some(dev_codename_str),
            transfer_strategy: if cmd.execution_method == ArgExecutionMethod::GpuStream {
                Some(cmd.transfer_strategy)
            } else {
                None
            },
            chunk_bytes: if cmd.execution_method == ArgExecutionMethod::GpuStream {
                Some(cmd.chunk_bytes)
            } else {
                None
            },
            threads: if cmd.execution_method == ArgExecutionMethod::Cpu {
                Some(cmd.threads)
            } else {
                None
            },
            hashing_scheme: Some(cmd.hashing_scheme),
            hash_table_memory_type: Some(cmd.hash_table_mem_type),
            hash_table_memory_location: Some(sorted_ht_location),
            hash_table_proportions: Some(sorted_ht_proportions),
            tuple_bytes: Some(cmd.tuple_bytes),
            relation_memory_type: Some(cmd.mem_type),
            inner_relation_memory_location: Some(cmd.inner_rel_location),
            outer_relation_memory_location: Some(cmd.outer_rel_location),
            ..self.clone()
        };

        Ok(dp)
    }

    pub fn fill_from_hash_join_bench<T: DeviceCopy>(&self, hjb: &HashJoinBench<T>) -> DataPoint {
        DataPoint {
            hash_table_bytes: Some(hjb.hash_table_len * size_of::<T>()),
            build_tuples: Some(hjb.build_relation_key.len()),
            build_bytes: Some(
                (hjb.build_relation_key.len() + hjb.build_relation_payload.len()) * size_of::<T>(),
            ),
            probe_tuples: Some(hjb.probe_relation_key.len()),
            probe_bytes: Some(
                (hjb.probe_relation_key.len() + hjb.probe_relation_payload.len()) * size_of::<T>(),
            ),
            ..self.clone()
        }
    }

    pub fn set_init_time(&self, malloc: Duration, data_gen: Duration) -> DataPoint {
        DataPoint {
            relation_malloc_ns: Some(malloc.as_nanos() as f64),
            relation_gen_ns: Some(data_gen.as_nanos() as f64),
            ..self.clone()
        }
    }
}

/// Serialize `Option<Vec<T>>` by converting it into a `String`.
///
/// This is necessary because the `csv` crate does not support nesting `Vec`
/// instead of flattening it.
fn serialize_vec<S, T>(option: &Option<Vec<T>>, ser: S) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
    T: ToString,
{
    if let Some(vec) = option {
        let record = vec
            .iter()
            .enumerate()
            .map(|(i, e)| {
                if i == 0 {
                    e.to_string()
                } else {
                    ",".to_owned() + &e.to_string()
                }
            })
            .collect::<String>();
        ser.serialize_str(&record)
    } else {
        ser.serialize_none()
    }
}
