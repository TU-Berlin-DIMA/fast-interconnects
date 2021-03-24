/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use super::data_point::DataPoint;
use super::hash_join_bench::HashJoinPoint;
use crate::error::Result;
use error_chain::ensure;
use std::path::PathBuf;

pub fn measure(
    _name: &str,
    repeat: u32,
    out_file_name: Option<PathBuf>,
    template: DataPoint,
    mut func: Box<dyn FnMut() -> Result<HashJoinPoint>>,
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
