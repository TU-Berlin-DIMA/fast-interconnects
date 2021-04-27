/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use super::data_point::DataPoint;
use super::{Benchmark, ItemBytes, MemoryOperation};
use crate::types::{Block, Cycles, Grid, Ilp, OversubRatio, Warp, WarpMul, SM};
use itertools::{iproduct, izip};
use numa_gpu::runtime::memory::Mem;
use numa_gpu::runtime::nvml::ThrottleReasons;
use std::iter;
use std::ops::RangeInclusive;

#[allow(dead_code)]
pub(super) struct GpuMeasurementParameters {
    pub(super) grid_size: Grid,
    pub(super) block_size: Block,
    pub(super) ilp: Ilp,
}

#[allow(dead_code)]
pub(super) struct GpuMeasurement {
    oversub_ratio: RangeInclusive<OversubRatio>,
    warp_mul: RangeInclusive<WarpMul>,
    warp_size: Warp,
    sm_count: SM,
    ilp: RangeInclusive<Ilp>,
    template: DataPoint,
}

impl GpuMeasurement {
    pub(super) fn new(
        oversub_ratio: RangeInclusive<OversubRatio>,
        warp_mul: RangeInclusive<WarpMul>,
        warp_size: Warp,
        sm_count: SM,
        ilp: RangeInclusive<Ilp>,
        template: DataPoint,
    ) -> Self {
        Self {
            oversub_ratio,
            warp_mul,
            warp_size,
            sm_count,
            ilp,
            template,
        }
    }

    pub(super) fn measure<R, S>(
        &self,
        mem: &Mem<u32>,
        mut state: S,
        run: R,
        benches: Vec<Benchmark>,
        ops: Vec<MemoryOperation>,
        item_bytes: Vec<ItemBytes>,
        repeat: u32,
    ) -> Vec<DataPoint>
    where
        R: Fn(
            Benchmark,
            MemoryOperation,
            ItemBytes,
            &mut S,
            &Mem<u32>,
            &GpuMeasurementParameters,
        ) -> Option<(u32, Option<ThrottleReasons>, u64, Cycles, u64)>,
    {
        // Convert newtypes to basic types while std::ops::Step is unstable
        // Step trait is required for std::ops::RangeInclusive Iterator trait
        let (OversubRatio(osr_l), OversubRatio(osr_u)) = self.oversub_ratio.clone().into_inner();
        let (WarpMul(wm_l), WarpMul(wm_u)) = self.warp_mul.clone().into_inner();
        let warp_size = self.warp_size.clone();
        let sm_count = self.sm_count.clone();

        let data_points: Vec<_> = iproduct!(
            iproduct!(
                benches.iter(),
                ops.iter(),
                item_bytes.iter(),
                (osr_l..=osr_u)
                    .filter(|osr| osr.is_power_of_two())
                    .map(|osr| OversubRatio(osr)),
                (wm_l..=wm_u)
                    .filter(|wm| wm.is_power_of_two())
                    .map(|wm| WarpMul(wm))
            ),
            izip!(iter::once(true).chain(iter::repeat(false)), 0..repeat)
        )
        .filter_map(
            |((&bench, &op, &item_bytes, oversub_ratio, warp_mul), (warm_up, _run))| {
                let block_size = warp_mul * warp_size;
                let grid_size = oversub_ratio * sm_count;
                let ilp = Ilp::default(); // FIXME: insert and use a real parameter
                let mp = GpuMeasurementParameters {
                    grid_size,
                    block_size,
                    ilp,
                };

                if let Some((clock_rate_mhz, throttle_reasons, memory_accesses, cycles, ns)) =
                    run(bench, op, item_bytes, &mut state, mem, &mp)
                {
                    Some(DataPoint {
                        benchmark: Some(bench),
                        memory_operation: Some(op),
                        item_bytes: Some(item_bytes),
                        warm_up,
                        grid_size: Some(grid_size),
                        block_size: Some(block_size),
                        ilp: None,
                        throttle_reasons: throttle_reasons.map(|r| r.to_string()),
                        clock_rate_mhz: Some(clock_rate_mhz),
                        memory_accesses,
                        cycles,
                        ns,
                        ..self.template.clone()
                    })
                } else {
                    None
                }
            },
        )
        .collect();
        data_points
    }
}
