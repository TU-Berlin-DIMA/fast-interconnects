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
use super::{Benchmark, MemoryOperation};
use crate::types::{Block, Cycles, Grid, Ilp, OversubRatio, Warp, WarpMul, SM};
use numa_gpu::runtime::memory::Mem;
use numa_gpu::runtime::nvml::ThrottleReasons;
use std::iter;
use std::ops::RangeInclusive;

#[allow(dead_code)]
pub(super) struct GpuMeasurementParameters {
    pub(super) grid_size: Grid,
    pub(super) block_size: Block,
    pub(super) ilp: Ilp,
    pub(super) loop_length: u32,
    pub(super) target_cycles: Cycles,
}

#[allow(dead_code)]
pub(super) struct GpuMeasurement<'h, 'd, 'n> {
    oversub_ratio: RangeInclusive<OversubRatio>,
    warp_mul: RangeInclusive<WarpMul>,
    warp_size: Warp,
    sm_count: SM,
    ilp: RangeInclusive<Ilp>,
    loop_length: u32,
    target_cycles: Cycles,
    template: DataPoint<'h, 'd, 'n>,
}

impl<'h, 'd, 'n> GpuMeasurement<'h, 'd, 'n> {
    pub(super) fn new(
        oversub_ratio: RangeInclusive<OversubRatio>,
        warp_mul: RangeInclusive<WarpMul>,
        warp_size: Warp,
        sm_count: SM,
        ilp: RangeInclusive<Ilp>,
        loop_length: u32,
        target_cycles: Cycles,
        template: DataPoint<'h, 'd, 'n>,
    ) -> Self {
        Self {
            oversub_ratio,
            warp_mul,
            warp_size,
            sm_count,
            ilp,
            loop_length,
            target_cycles,
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
        repeat: u32,
    ) -> Vec<DataPoint<'h, 'd, 'n>>
    where
        R: Fn(
            Benchmark,
            MemoryOperation,
            &mut S,
            &Mem<u32>,
            &GpuMeasurementParameters,
        ) -> (u32, Option<ThrottleReasons>, u64, u64, u64),
    {
        // Convert newtypes to basic types while std::ops::Step is unstable
        // Step trait is required for std::ops::RangeInclusive Iterator trait
        let (OversubRatio(osr_l), OversubRatio(osr_u)) = self.oversub_ratio.clone().into_inner();
        let (WarpMul(wm_l), WarpMul(wm_u)) = self.warp_mul.clone().into_inner();
        let warp_mul_iter = wm_l..=wm_u;
        let warp_size = self.warp_size.clone();
        let sm_count = self.sm_count.clone();

        let data_points: Vec<_> = benches
            .iter()
            .flat_map(|fut| {
                iter::repeat(fut).zip(ops.iter().flat_map(|op| {
                    let oversub_ratio_iter = osr_l..=osr_u;
                    iter::repeat(op).zip(
                        oversub_ratio_iter
                            .filter(|osr| osr.is_power_of_two())
                            .map(|osr| OversubRatio(osr))
                            .flat_map(|osr| {
                                warp_mul_iter
                                    .clone()
                                    .filter(|wm| wm.is_power_of_two())
                                    .map(|wm| WarpMul(wm))
                                    .zip(std::iter::repeat(osr))
                                    .flat_map(|params| {
                                        iter::repeat(params)
                                            .zip(iter::once(true).chain(iter::repeat(false)))
                                            .zip(0..repeat)
                                    })
                            }),
                    )
                }))
            })
            .map(
                |(bench, (op, (((warp_mul, oversub_ratio), warm_up), _run)))| {
                    let block_size = warp_mul * warp_size;
                    let grid_size = oversub_ratio * sm_count;
                    let ilp = Ilp::default(); // FIXME: insert and use a real parameter
                    let mp = GpuMeasurementParameters {
                        grid_size,
                        block_size,
                        ilp,
                        loop_length: self.loop_length,
                        target_cycles: self.target_cycles,
                    };

                    let (clock_rate_mhz, throttle_reasons, memory_accesses, cycles, ns) =
                        run(*bench, *op, &mut state, mem, &mp);

                    DataPoint {
                        benchmark: Some(*bench),
                        memory_operation: Some(*op),
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
                    }
                },
            )
            .collect();
        data_points
    }
}
