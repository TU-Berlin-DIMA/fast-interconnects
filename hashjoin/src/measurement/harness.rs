/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use super::data_point::DataPoint;
use super::hash_join_bench::HashJoinPoint;
use average::{concatenate, impl_from_iterator, Estimate, Max, Min, Quantile, Variance};
use error_chain::ensure;
use numa_gpu::error::Result;
use std::path::PathBuf;

pub fn measure(
    name: &str,
    repeat: u32,
    out_file_name: Option<PathBuf>,
    template: DataPoint,
    mut func: Box<FnMut() -> Result<HashJoinPoint>>,
) -> Result<()> {
    let measurements = (0..=repeat)
        .zip(std::iter::once(true).chain(std::iter::repeat(false)))
        .map(|(_, warm_up)| {
            func().map(|p| DataPoint {
                warm_up: Some(warm_up),
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

    concatenate!(
        Estimator,
        [Variance, variance, mean, error],
        [Quantile, quantile, quantile],
        [Min, min, min],
        [Max, max, max]
    );

    let time_stats: Estimator = measurements
        .iter()
        .filter(|row| row.warm_up == Some(false))
        .filter_map(|row| row.build_ns.and_then(|b| row.probe_ns.map(|p| b + p)))
        .map(|hj_ns| hj_ns / 10_f64.powf(6.0))
        .collect();

    let bw_stats: Estimator = measurements
        .iter()
        .filter(|row| row.warm_up == Some(false))
        .filter_map(|row| {
            row.probe_bytes.and_then(|bytes| {
                row.build_ns
                    .and_then(|build_ns| row.probe_ns.map(|probe_ns| (bytes, build_ns + probe_ns)))
            })
        })
        .map(|(output_bytes, hj_ns)| (output_bytes as f64, hj_ns))
        .map(|(output_bytes, hj_ns)| output_bytes / hj_ns / 2.0_f64.powf(30.0) * 10.0_f64.powf(9.0))
        .collect();

    let tput_stats: Estimator = measurements
        .iter()
        .filter(|row| row.warm_up == Some(false))
        .filter_map(|row| {
            row.probe_tuples.and_then(|tuples| {
                row.build_ns
                    .and_then(|build_ns| row.probe_ns.map(|probe_ns| (tuples, build_ns + probe_ns)))
            })
        })
        .map(|(output_tuples, hj_ns)| (output_tuples as f64, hj_ns))
        .map(|(output_tuples, hj_ns)| output_tuples / hj_ns)
        .collect();

    println!(
        r#"Bench: {}
Sample size: {}
               Time            Bandwidth            Throughput
                ms              GiB/s                GTuples/s
Mean:          {:6.2}          {:6.2}               {:6.2}
Stddev:        {:6.2}          {:6.2}               {:6.2}
Median:        {:6.2}          {:6.2}               {:6.2}
Min:           {:6.2}          {:6.2}               {:6.2}
Max:           {:6.2}          {:6.2}               {:6.2}"#,
        name.replace("_", " "),
        measurements.len(),
        time_stats.mean(),
        bw_stats.mean(),
        tput_stats.mean(),
        time_stats.error(),
        bw_stats.error(),
        tput_stats.error(),
        time_stats.quantile(),
        bw_stats.quantile(),
        tput_stats.quantile(),
        time_stats.min(),
        bw_stats.min(),
        tput_stats.min(),
        time_stats.max(),
        bw_stats.max(),
        tput_stats.max(),
    );

    Ok(())
}
