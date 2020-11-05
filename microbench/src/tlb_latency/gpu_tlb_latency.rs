/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2020 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use super::DataPoint;
use crate::error::{ErrorKind, Result};
use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemType};
use numa_gpu::runtime::memory::{DerefMem, Mem};
use numa_gpu::runtime::utils::EnsurePhysicallyBacked;
use rustacuda::context::{Context, ContextFlags, CurrentContext};
use rustacuda::device::{Device, DeviceAttribute};
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::launch;
use rustacuda::memory::{CopyDestination, DeviceBox};
use rustacuda::module::Module;
use rustacuda::stream::{Stream, StreamFlags};
use std::cmp;
use std::convert::TryInto;
use std::ffi::CString;
use std::iter;
use std::mem;
use std::ops::RangeInclusive;

#[cfg(not(target_arch = "aarch64"))]
use numa_gpu::runtime::nvml::{DeviceClocks, ThrottleReasons};
#[cfg(not(target_arch = "aarch64"))]
use nvml_wrapper::{enum_wrappers::device::Clock, NVML};

#[cfg(target_arch = "aarch64")]
use numa_gpu::runtime::hw_info::CudaDeviceInfo;

const TLB_DATA_POINTS_STR: &str = env!("TLB_DATA_POINTS");

#[derive(Debug)]
pub(super) struct GpuTlbLatency {
    // `module` must be dropped before `context`. Rust specifies the drop order as the field order
    // in the struct. See RFC 1857: https://github.com/rust-lang/rfcs/pull/1857
    module: Module,
    context: Context,
    device: Device,
    device_id: u32,
    template: DataPoint,

    #[cfg(not(target_arch = "aarch64"))]
    nvml: nvml_wrapper::NVML,
}

impl GpuTlbLatency {
    #[cfg(not(target_arch = "aarch64"))]
    pub(super) fn new(device_id: u32, template: DataPoint) -> Result<Self> {
        let device = Device::get_device(device_id)?;
        let context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let module = Self::load_module()?;
        let nvml = NVML::init()?;

        Ok(Self {
            context,
            device,
            device_id,
            module,
            template,
            nvml,
        })
    }

    #[cfg(target_arch = "aarch64")]
    pub(super) fn new(device_id: u32, template: DataPoint) -> Result<Self> {
        let device = Device::get_device(device_id)?;
        let context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let module = Self::load_module()?;

        Ok(Self {
            context,
            device,
            device_id,
            module,
            template,
        })
    }

    fn load_module() -> Result<Module> {
        let module_path = CString::new(env!("CUDAUTILS_PATH")).map_err(|_| {
            ErrorKind::NulCharError(
                "Failed to load CUDA module, check your CUDAUTILS_PATH".to_string(),
            )
        })?;
        let module = Module::load_from_file(&module_path)?;

        Ok(module)
    }

    pub(super) fn measure(
        &self,
        mem_type: MemType,
        ranges: RangeInclusive<usize>,
        strides: &[usize],
    ) -> Result<Vec<DataPoint>> {
        type Position = u64;

        let tlb_data_points: usize = TLB_DATA_POINTS_STR
            .parse()
            .expect("Failed to parse \"TLB_DATA_POINTS\" string to an integer");
        let device = CurrentContext::get_device()?;
        let grid_size =
            GridSize::from(device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32);
        let block_size =
            BlockSize::from(device.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32);

        let position_bytes = mem::size_of::<Position>();
        let data_bytes = *ranges.end();
        let data_len = data_bytes / position_bytes;
        let mut data: Mem<Position> = Allocator::alloc_mem(mem_type, data_len);

        if let Ok(d) = (&mut data).try_into() {
            Position::ensure_physically_backed(d);
        }

        let mut cycles: DerefMem<u32> =
            Allocator::alloc_deref_mem(DerefMemType::CudaPinnedMem, tlb_data_points);
        let mut indices: DerefMem<Position> =
            Allocator::alloc_deref_mem(DerefMemType::CudaPinnedMem, tlb_data_points);

        // Set a stable GPU clock rate to make the measurements more accurate
        #[cfg(not(target_arch = "aarch64"))]
        self.nvml
            .device_by_index(self.device_id as u32)?
            .set_max_gpu_clocks()?;

        let cycle_counter_overhead = self.cycle_counter_overhead()?;

        // FIXME: factor into a closure, so that we can measure different GPU kernels

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let data_points: Vec<DataPoint> = strides.iter().cloned().flat_map(|s| ranges.clone().skip_while(move |r| r % s != 0).step_by(s).zip(iter::repeat(s))).map(|(range, stride)| {
            let module = &self.module;
            let size: usize = range / position_bytes;
            let start_offset: usize = 0;
            let p_stride: usize = stride / position_bytes;
            let iterations: u32 = cmp::max(
                tlb_data_points as u32,
                range
                    .checked_div(stride)
                    .ok_or_else(|| ErrorKind::InvalidArgument("Stride is zero".to_string()))?
                    as u32,
            );

            unsafe {
                launch!(module.initialize_strides<<<grid_size.clone(), block_size.clone(), 0, stream>>>(
                data.as_launchable_mut_ptr(),
                size,
                start_offset,
                p_stride
                ))?;
            }
            stream.synchronize()?;

            // Get GPU clock rate that applications run at
            #[cfg(not(target_arch = "aarch64"))]
            let clock_rate_mhz = self
                .nvml
                .device_by_index(self.device_id as u32)?
                .clock_info(Clock::Graphics)?;

            #[cfg(target_arch = "aarch64")]
            let clock_rate_mhz = CurrentContext::get_device()?.clock_rate()?;

            unsafe {
                launch!(module.tlb_stride_single_thread<<<1, 1, 0, stream>>>(
                data.as_launchable_ptr(),
                size,
                iterations,
                cycles.as_launchable_mut_ptr(),
                indices.as_launchable_mut_ptr()
                ))?;
            }
            stream.synchronize()?;

            // Check if GPU is running in a throttled state
            #[cfg(not(target_arch = "aarch64"))]
            let throttle_reasons: Option<ThrottleReasons> = Some(
                self.nvml
                    .device_by_index(self.device_id as u32)?
                    .current_throttle_reasons()?
                    .into(),
            );

            #[cfg(target_arch = "aarch64")]
            let throttle_reasons: Option<&str> = None;

            let dp: Vec<DataPoint> = cycles
                .iter()
                .zip(indices.iter())
                .map(|(&cycles, &index)| DataPoint {
                    grid_size: Some(1),
                    block_size: Some(1),
                    range_bytes: Some(range),
                    stride_bytes: Some(stride),
                    throttle_reasons: throttle_reasons.as_ref().map(|r| r.to_string()),
                    clock_rate_mhz: Some(clock_rate_mhz),
                    cycle_counter_overhead_cycles: Some(cycle_counter_overhead),
                    index_bytes: Some(index * position_bytes as u64),
                    cycles: Some(cycles),
                    ns: Some((1000 * cycles) / clock_rate_mhz),
                    ..self.template.clone()
                })
                .collect();

            Ok(dp)
        }).collect::<Result<Vec<Vec<DataPoint>>>>()?.concat();

        Ok(data_points)
    }

    /// Measures the overhead of `get_clock()` in cycles
    ///
    /// `get_clock()` is implemented as a native special register on each
    /// architecture. See the documentation in the CUDA module for details.
    fn cycle_counter_overhead(&self) -> Result<u32> {
        let mut overhead = DeviceBox::new(&0_u32)?;
        let mut fake_dependency = DeviceBox::new(&0_u32)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let module = &self.module;

        unsafe {
            launch!(module.cycle_counter_overhead<<<1, 1, 0, stream>>>(
            overhead.as_device_ptr(),
            fake_dependency.as_device_ptr()
            ))?;
        }
        stream.synchronize()?;

        let mut overhead_host = 0_u32;
        overhead.copy_to(&mut overhead_host)?;

        Ok(overhead_host)
    }
}

impl Drop for GpuTlbLatency {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "aarch64"))]
        self.nvml
            .device_by_index(self.device_id as u32)
            .unwrap()
            .set_default_gpu_clocks()
            .expect("Failed to reset default GPU clock rates");
    }
}
