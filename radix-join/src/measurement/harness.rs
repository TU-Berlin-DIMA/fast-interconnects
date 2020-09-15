/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019-2020, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use super::data_point::DataPoint;
use crate::error::Result;
use std::path::PathBuf;

#[derive(Debug, Default)]
pub struct RadixJoinPoint {
    pub partitions_malloc_ns: Option<f64>,
    pub prefix_sum_ns: Option<f64>,
    pub partition_ns: Option<f64>,
    pub join_ns: Option<f64>,
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
                prefix_sum_ns: p.prefix_sum_ns,
                partition_ns: p.partition_ns,
                join_ns: p.join_ns,
                partitions_malloc_ns: p.partitions_malloc_ns,
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
