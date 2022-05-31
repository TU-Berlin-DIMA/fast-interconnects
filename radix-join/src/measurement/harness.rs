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

use super::data_point::DataPoint;
use crate::error::Result;
use numa_gpu::runtime::nvtx::Range;
use std::ffi::CString;
use std::path::PathBuf;

#[derive(Debug, Default)]
pub struct RadixJoinPoint {
    pub partitions_malloc_ns: Option<f64>,
    pub state_malloc_ns: Option<f64>,
    pub prefix_sum_ns: Option<f64>,
    pub partition_ns: Option<f64>,
    pub join_ns: Option<f64>,
    pub cached_build_tuples: Option<usize>,
    pub cached_probe_tuples: Option<usize>,
}

pub fn measure(
    _name: &str,
    repeat: u32,
    out_file_name: Option<PathBuf>,
    template: DataPoint,
    mut func: Box<dyn FnMut() -> Result<RadixJoinPoint>>,
) -> Result<()> {
    let measurements = (0..repeat)
        .zip(std::iter::once(true).chain(std::iter::repeat(false)))
        .map(|(run, warm_up)| {
            let range_message =
                CString::new(format!("Measurement run {}", run)).expect("Failed to format string");

            let range = Range::new(&range_message);
            let result = func();
            let run_id = range.end();

            result.map(|p| DataPoint {
                warm_up: Some(warm_up),
                nvtx_run_id: Some(run_id),
                cached_build_tuples: p.cached_build_tuples,
                cached_probe_tuples: p.cached_probe_tuples,
                relation_malloc_ns: if warm_up {
                    template.relation_malloc_ns
                } else {
                    None
                },
                relation_gen_ns: if warm_up {
                    template.relation_gen_ns
                } else {
                    None
                },
                prefix_sum_ns: p.prefix_sum_ns,
                partition_ns: p.partition_ns,
                join_ns: p.join_ns,
                partitions_malloc_ns: p.partitions_malloc_ns,
                state_malloc_ns: p.state_malloc_ns,
                ..template.clone()
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if let Some(ofn) = out_file_name {
        let csv_file = std::fs::File::create(ofn)?;
        let mut csv = csv::Writer::from_writer(csv_file);
        measurements.iter().try_for_each(|row| csv.serialize(row))?;
    }

    Ok(())
}
