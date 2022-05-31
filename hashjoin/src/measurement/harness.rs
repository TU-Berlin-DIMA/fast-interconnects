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
use super::hash_join_bench::HashJoinPoint;
use crate::error::Result;
use error_chain::ensure;
use numa_gpu::runtime::nvtx::Range;
use std::ffi::CString;
use std::path::PathBuf;

pub fn measure(
    _name: &str,
    repeat: u32,
    out_file_name: Option<PathBuf>,
    template: DataPoint,
    mut func: Box<dyn FnMut() -> Result<HashJoinPoint>>,
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
                hash_table_malloc_ns: p.hash_table_malloc_ns,
                build_ns: p.build_ns,
                probe_ns: p.probe_ns,
                build_warm_up_ns: p.build_warm_up_ns,
                probe_warm_up_ns: p.probe_warm_up_ns,
                build_copy_ns: p.build_copy_ns,
                probe_copy_ns: p.probe_copy_ns,
                build_compute_ns: p.build_compute_ns,
                probe_compute_ns: p.probe_compute_ns,
                build_cool_down_ns: p.build_cool_down_ns,
                probe_cool_down_ns: p.probe_cool_down_ns,
                cached_hash_table_tuples: p.cached_hash_table_tuples,
                ..template.clone()
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if let Some(ofn) = out_file_name {
        let csv_file = std::fs::File::create(ofn)?;
        let mut csv = csv::Writer::from_writer(csv_file);
        ensure!(
            measurements
                .iter()
                .try_for_each(|row| csv.serialize(row))
                .is_ok(),
            "Couldn't write serialized measurements"
        );
    }

    Ok(())
}
