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

use super::hash_join_bench::HashJoinBench;
use crate::types::*;
use crate::CmdOpt;
use data_store::join_data::JoinData;
use numa_gpu::error::Result;
use numa_gpu::runtime::hw_info::cpu_codename;
use numa_gpu::runtime::nvtx::RangeId;
use rustacuda::device::Device;
use rustacuda::function::{BlockSize, GridSize};
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
    pub cpu_morsel_bytes: Option<usize>,
    pub gpu_morsel_bytes: Option<usize>,
    pub threads: Option<usize>,
    pub grid_size: Option<u32>,
    pub block_size: Option<u32>,
    pub hashing_scheme: Option<ArgHashingScheme>,
    pub hash_table_memory_type: Option<ArgMemType>,
    #[serde(serialize_with = "serialize_vec")]
    pub hash_table_memory_location: Option<Vec<u16>>,
    #[serde(serialize_with = "serialize_vec")]
    pub hash_table_proportions: Option<Vec<usize>>,
    pub hash_table_tuples: Option<usize>,
    pub cached_hash_table_tuples: Option<usize>,
    pub tuple_bytes: Option<ArgTupleBytes>,
    pub relation_memory_type: Option<ArgMemType>,
    pub page_type: Option<ArgPageType>,
    pub inner_relation_memory_location: Option<u16>,
    pub outer_relation_memory_location: Option<u16>,
    pub build_tuples: Option<usize>,
    pub build_bytes: Option<usize>,
    pub probe_tuples: Option<usize>,
    pub probe_bytes: Option<usize>,
    pub data_distribution: Option<ArgDataDistribution>,
    pub zipf_exponent: Option<f64>,
    pub join_selectivity: Option<f64>,
    pub warm_up: Option<bool>,
    pub nvtx_run_id: Option<RangeId>,
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

    pub(crate) fn fill_from_cmd_options(&self, cmd: &CmdOpt) -> Result<DataPoint> {
        // Get device information
        let dev_codename_str = match cmd.execution_method {
            ArgExecutionMethod::Cpu => vec![cpu_codename()?],
            ArgExecutionMethod::Gpu | ArgExecutionMethod::GpuStream => {
                let device = Device::get_device(cmd.device_id.into())?;
                vec![device.name()?]
            }
            ArgExecutionMethod::Het | ArgExecutionMethod::GpuBuildHetProbe => {
                let device = Device::get_device(cmd.device_id.into())?;
                vec![cpu_codename()?, device.name()?]
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
            cpu_morsel_bytes: if cmd.execution_method == ArgExecutionMethod::Het
                || cmd.execution_method == ArgExecutionMethod::GpuBuildHetProbe
            {
                Some(cmd.cpu_morsel_bytes)
            } else {
                None
            },
            gpu_morsel_bytes: if cmd.execution_method == ArgExecutionMethod::GpuStream
                || cmd.execution_method == ArgExecutionMethod::Het
                || cmd.execution_method == ArgExecutionMethod::GpuBuildHetProbe
            {
                Some(cmd.gpu_morsel_bytes)
            } else {
                None
            },
            threads: if cmd.execution_method != ArgExecutionMethod::Gpu
                && cmd.execution_method != ArgExecutionMethod::GpuStream
            {
                Some(cmd.threads)
            } else {
                None
            },
            hashing_scheme: Some(cmd.hashing_scheme),
            hash_table_memory_type: Some(cmd.hash_table_mem_type),
            hash_table_memory_location: Some(cmd.hash_table_location.clone()),
            hash_table_proportions: Some(cmd.hash_table_proportions.clone()),
            tuple_bytes: Some(cmd.tuple_bytes),
            relation_memory_type: Some(cmd.mem_type),
            page_type: Some(cmd.page_type),
            inner_relation_memory_location: Some(cmd.inner_rel_location),
            outer_relation_memory_location: Some(cmd.outer_rel_location),
            data_distribution: Some(cmd.data_distribution),
            zipf_exponent: if cmd.data_distribution == ArgDataDistribution::Zipf {
                cmd.zipf_exponent
            } else {
                None
            },
            join_selectivity: Some(cmd.selectivity as f64 / 100.0),
            ..self.clone()
        };

        Ok(dp)
    }

    pub fn fill_from_hash_join_bench<T>(&self, hjb: &HashJoinBench<T>) -> DataPoint {
        DataPoint {
            hash_table_tuples: Some(hjb.hash_table_len),
            ..self.clone()
        }
    }

    pub fn fill_from_join_data<T: DeviceCopy>(&self, join_data: &JoinData<T>) -> DataPoint {
        DataPoint {
            build_tuples: Some(join_data.build_relation_key.len()),
            build_bytes: Some(
                (join_data.build_relation_key.len() + join_data.build_relation_payload.len())
                    * size_of::<T>(),
            ),
            probe_tuples: Some(join_data.probe_relation_key.len()),
            probe_bytes: Some(
                (join_data.probe_relation_key.len() + join_data.probe_relation_payload.len())
                    * size_of::<T>(),
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

    pub fn set_gpu_threads(&self, grid_size: &GridSize, block_size: &BlockSize) -> DataPoint {
        DataPoint {
            grid_size: Some(grid_size.x),
            block_size: Some(block_size.x),
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
