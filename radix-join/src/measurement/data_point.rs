/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use crate::types::*;
use data_store::join_data::JoinData;
use numa_gpu::error::Result;
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
    pub histogram_algorithm: Option<ArgHistogramAlgorithm>,
    pub partition_algorithm: Option<ArgRadixPartitionAlgorithm>,
    pub partition_algorithm_2nd: Option<ArgRadixPartitionAlgorithm>,
    pub execution_method: Option<ArgExecutionMethod>,
    #[serde(serialize_with = "serialize_vec")]
    pub device_codename: Option<Vec<String>>,
    pub dmem_buffer_size: Option<usize>,
    pub threads: Option<usize>,
    pub radix_bits_fst: Option<u32>,
    pub radix_bits_snd: Option<u32>,
    pub radix_bits_trd: Option<u32>,
    pub hashing_scheme: Option<ArgHashingScheme>,
    pub partitions_memory_type: Option<ArgMemType>,
    #[serde(serialize_with = "serialize_vec")]
    pub partitions_memory_location: Option<Vec<u16>>,
    #[serde(serialize_with = "serialize_vec")]
    pub partitions_proportions: Option<Vec<usize>>,
    pub state_memory_type: Option<ArgMemType>,
    pub state_memory_location: Option<u16>,
    pub tuple_bytes: Option<ArgTupleBytes>,
    pub relation_memory_type: Option<ArgMemType>,
    pub page_type: Option<ArgPageType>,
    pub inner_relation_memory_location: Option<u16>,
    pub outer_relation_memory_location: Option<u16>,
    pub build_tuples: Option<usize>,
    pub build_bytes: Option<usize>,
    pub probe_tuples: Option<usize>,
    pub probe_bytes: Option<usize>,
    pub cached_build_tuples: Option<usize>,
    pub cached_probe_tuples: Option<usize>,
    pub data_distribution: Option<ArgDataDistribution>,
    pub zipf_exponent: Option<f64>,
    pub join_selectivity: Option<f64>,
    pub warm_up: Option<bool>,
    pub prefix_sum_ns: Option<f64>,
    pub partition_ns: Option<f64>,
    pub join_ns: Option<f64>,
    pub partitions_malloc_ns: Option<f64>,
    pub state_malloc_ns: Option<f64>,
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
