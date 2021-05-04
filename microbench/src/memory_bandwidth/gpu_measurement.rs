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
use super::{Benchmark, ItemBytes, MemoryOperation, TileSize};
use crate::types::{Block, Cycles, Grid};
use itertools::{iproduct, izip};
use numa_gpu::runtime::memory::Mem;
use numa_gpu::runtime::nvml::{DeviceClocks, ThrottleReasons};
use nvml_wrapper::device::Device as NvmlDevice;
use nvml_wrapper::NVML;
use std::iter;

#[allow(dead_code)]
pub(super) struct GpuMeasurementParameters {
    pub(super) grid_size: Grid,
    pub(super) block_size: Block,
}

#[allow(dead_code)]
pub(super) struct GpuMeasurement {
    device_id: u32,
    grid_sizes: Vec<Grid>,
    block_sizes: Vec<Block>,
    template: DataPoint,
    nvml: nvml_wrapper::NVML,
}

impl GpuMeasurement {
    pub(super) fn new(
        device_id: u32,
        grid_sizes: Vec<Grid>,
        block_sizes: Vec<Block>,
        template: DataPoint,
    ) -> Self {
        let nvml = NVML::init().expect("Couldn't initialize NVML");

        Self {
            device_id,
            grid_sizes,
            block_sizes,
            template,
            nvml,
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
        tile_sizes: Vec<TileSize>,
        warp_aligned: bool,
        repeat: u32,
    ) -> Vec<DataPoint>
    where
        R: Fn(
            Benchmark,
            MemoryOperation,
            ItemBytes,
            TileSize,
            &mut S,
            &Mem<u32>,
            &GpuMeasurementParameters,
            &NvmlDevice,
        ) -> Option<(u32, Option<ThrottleReasons>, u64, Cycles, u64)>,
    {
        let mut nvml_device = self
            .nvml
            .device_by_index(self.device_id as u32)
            .expect("Couldn't get NVML device");

        // Set a stable GPU clock rate to make the measurements more accurate
        if let Err(e) = nvml_device.set_max_gpu_clocks() {
            eprintln!("Warning: Failed to set the maximum GPU clockrate [{}]", e);
        }

        let data_points: Vec<_> = iproduct!(
            iproduct!(
                benches.iter(),
                ops.iter(),
                item_bytes.iter(),
                tile_sizes.iter(),
                self.grid_sizes.iter(),
                self.block_sizes.iter()
            ),
            izip!(iter::once(true).chain(iter::repeat(false)), 0..repeat)
        )
        .filter_map(
            |((&bench, &op, &item_bytes, &tile_size, &grid_size, &block_size), (warm_up, _run))| {
                let mp = GpuMeasurementParameters {
                    grid_size,
                    block_size,
                };

                if let Some((clock_rate_mhz, throttle_reasons, memory_accesses, cycles, ns)) = run(
                    bench,
                    op,
                    item_bytes,
                    tile_size,
                    &mut state,
                    mem,
                    &mp,
                    &nvml_device,
                ) {
                    Some(DataPoint {
                        benchmark: Some(bench),
                        memory_operation: Some(op),
                        item_bytes: Some(item_bytes),
                        tile_size: Some(tile_size),
                        warp_aligned: Some(warp_aligned || tile_size == TileSize::Threads1),
                        warm_up,
                        grid_size: Some(grid_size),
                        block_size: Some(block_size),
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

impl Drop for GpuMeasurement {
    fn drop(&mut self) {
        self.nvml
            .device_by_index(self.device_id as u32)
            .unwrap()
            .set_default_gpu_clocks()
            .expect("Failed to reset default GPU clock rates");
    }
}
