/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

extern crate accel;
#[macro_use]
extern crate average;
extern crate core; // Required by average::concatenate!{} macro
extern crate csv;
extern crate cuda_sys;
extern crate hostname;
extern crate numa_gpu;
#[macro_use]
extern crate serde_derive;
extern crate serde;

use accel::device::sync;
use accel::event::Event;
use accel::uvec::UVec;

use average::{Estimate, Max, Min, Quantile, Variance};

use numa_gpu::error::Result;
use numa_gpu::operators::hash_join;

use std::path::PathBuf;

#[derive(Debug, Serialize)]
pub struct DataPoint<'h> {
    pub hostname: &'h String,
    pub warm_up: bool,
    pub hash_table_bytes: usize,
    pub build_bytes: usize,
    pub probe_bytes: usize,
    pub join_selectivity: f64,
    pub gpu_ms: f32,
}

fn main() {
    let repeat = 100;

    measure("full_hash_join", repeat).expect("Failure: full hash join benchmark");
}

fn measure(name: &str, repeat: u32) -> Result<()> {
    let hostname = &hostname::get_hostname().ok_or_else(|| "Couldn't get hostname")?;

    let warp_size = 32;
    let cuda_cores = 384;
    let overcommit_factor = 8;

    let block_size = warp_size;
    let grid_size = cuda_cores * overcommit_factor / warp_size;

    let hjb = HashJoinBench {
        hash_table_size: 1024,
        build_size: 50,
        probe_size: 10_usize.pow(7),
        join_selectivity: 0.0,
        build_dim: (grid_size, block_size),
        probe_dim: (grid_size, block_size),
    };

    // FIXME: hard-coded unit sizes
    let measurements = (0..repeat)
        .map(|_| {
            hjb.full_hash_join().map(|gpu_ms| DataPoint {
                hostname,
                warm_up: false,
                hash_table_bytes: hjb.hash_table_size * 16,
                build_bytes: hjb.build_size * 8,
                probe_bytes: hjb.probe_size * 8,
                join_selectivity: 0.0,
                gpu_ms,
            })
        }).collect::<Result<Vec<_>>>()?;

    let csv_path = PathBuf::from(name).with_extension("csv");

    let csv_file = std::fs::File::create(csv_path)?;

    let mut csv = csv::Writer::from_writer(csv_file);
    measurements
        .iter()
        .try_for_each(|row| csv.serialize(row))
        .expect("Couldn't write serialized measurements");

    concatenate!(
        Estimator,
        [Variance, variance, mean, error],
        [Quantile, quantile, quantile],
        [Min, min, min],
        [Max, max, max]
    );

    let time_stats: Estimator = measurements
        .iter()
        .map(|row| row.gpu_ms as f64)
        .collect();

    let tput_stats: Estimator = measurements
        .iter()
        .map(|row| (row.probe_bytes as f64, row.gpu_ms as f64))
        .map(|(bytes, ms)| bytes / ms / 2.0_f64.powf(30.0) * 10.0_f64.powf(3.0))
        .collect();

    println!(
        r#"Bench: {}
Sample size: {}
               Time            Throughput
                ms              GiB/s
Mean:          {:6.2}          {:6.2}
Stddev:        {:6.2}          {:6.2}
Median:        {:6.2}          {:6.2}
Min:           {:6.2}          {:6.2}
Max:           {:6.2}          {:6.2}"#,
        name.replace("_", " "),
        measurements.len(),
        time_stats.mean(),
        tput_stats.mean(),
        time_stats.error(),
        tput_stats.error(),
        time_stats.quantile(),
        tput_stats.quantile(),
        time_stats.min(),
        tput_stats.min(),
        time_stats.max(),
        tput_stats.max(),
    );

    Ok(())
}

struct HashJoinBench {
    hash_table_size: usize,
    build_size: usize,
    probe_size: usize,
    join_selectivity: f64,
    build_dim: (u32, u32),
    probe_dim: (u32, u32),
}

impl HashJoinBench {
    fn full_hash_join(&self) -> Result<f32> {
        let hash_table = hash_join::HashTable::new(self.hash_table_size);
        let mut build_join_attr = UVec::<i64>::new(self.build_size).unwrap();
        let mut build_selection_attr: UVec<i64> = UVec::new(build_join_attr.len()).unwrap();
        let mut counts_result: UVec<u64> = UVec::new((self.probe_dim.0 * self.probe_dim.1) as usize).unwrap();
        let mut probe_join_attr: UVec<i64> = UVec::new(self.probe_size).unwrap();
        let mut probe_selection_attr: UVec<i64> = UVec::new(probe_join_attr.len()).unwrap();

        // Generate some random build data
        for (i, x) in build_join_attr.as_slice_mut().iter_mut().enumerate() {
            *x = i as i64;
        }

        // Generate some random probe data
        for (i, x) in probe_join_attr.as_slice_mut().iter_mut().enumerate() {
            *x = (i % build_join_attr.len()) as i64;
        }

        // Initialize counts
        counts_result
            .iter_mut()
            .map(|count| *count = 0)
            .collect::<()>();

        // Set build selection attributes to 100% selectivity
        build_selection_attr
            .iter_mut()
            .map(|x| *x = 2)
            .collect::<()>();

        // Set probe selection attributes to 100% selectivity
        probe_selection_attr
            .iter_mut()
            .map(|x| *x = 2)
            .collect::<()>();

        let mut hj_op = hash_join::CudaHashJoinBuilder::default()
            .build_dim(self.build_dim.0, self.build_dim.1)
            .probe_dim(self.probe_dim.0, self.probe_dim.1)
            .hash_table(hash_table)
            .result_set(counts_result)
            .build();

        // println!("{:#?}", hj_op);

        let start_event = Event::new().unwrap();
        let stop_event = Event::new().unwrap();

        start_event.record().unwrap();

        let _join_result = hj_op
            .build(build_join_attr, build_selection_attr)
            .probe(probe_join_attr, probe_selection_attr);

        stop_event.record().and_then(|e| e.synchronize()).unwrap();
        let millis = stop_event.elapsed_time(&start_event).unwrap();

        sync().unwrap();
        Ok(millis)
    }
}
