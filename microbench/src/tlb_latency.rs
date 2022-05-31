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
