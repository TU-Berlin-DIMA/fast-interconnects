/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2020, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use crate::data_point::DataPoint;
use crate::error::Result;
use std::io::Write;
use std::time::Duration;

pub fn measure<W, F>(
    repeat: u32,
    writer: Option<Box<W>>,
    template: DataPoint,
    mut func: Box<F>,
) -> Result<()>
where
    W: Write,
    F: FnMut() -> Result<(i64, Duration)>,
{
    let measurements = (0..=repeat)
        .zip(std::iter::once(true).chain(std::iter::repeat(false)))
        .map(|(_, warm_up)| {
            func().map(|(_result, duration)| DataPoint {
                warm_up: Some(warm_up),
                ns: Some(duration.as_nanos() as f64),
                ..template.clone()
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if let Some(w) = writer {
        let mut csv = csv::Writer::from_writer(w);
        measurements.iter().try_for_each(|row| csv.serialize(row))?;
    }

    Ok(())
}
