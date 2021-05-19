/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2020-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

mod data_point;
mod gpu_tlb_latency;

pub use self::data_point::DataPoint;
use self::gpu_tlb_latency::GpuTlbLatency;
use crate::error::Result;
use numa_gpu::runtime::allocator::MemType;
use std::ops::RangeInclusive;

pub struct TlbLatency;

impl TlbLatency {
    pub fn measure<W>(
        device_id: u16,
        mem_type: MemType,
        ranges: RangeInclusive<usize>,
        strides: &[usize],
        iotlb_flush: bool,
        template: DataPoint,
        writer: Option<&mut W>,
    ) -> Result<()>
    where
        W: std::io::Write,
    {
        let gpu_tlb_latency = GpuTlbLatency::new(device_id.into(), template)?;
        let data_points = gpu_tlb_latency.measure(mem_type, ranges, strides, iotlb_flush)?;

        if let Some(w) = writer {
            let mut csv = csv::Writer::from_writer(w);
            data_points.iter().try_for_each(|row| csv.serialize(row))?;
        }

        Ok(())
    }
}
