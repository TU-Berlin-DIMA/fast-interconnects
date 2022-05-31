// Copyright 2020-2022 Clemens Lutz
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
