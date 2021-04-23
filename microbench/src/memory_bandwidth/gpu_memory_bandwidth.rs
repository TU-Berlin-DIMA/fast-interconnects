/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use super::gpu_measurement::GpuMeasurementParameters;
use super::MemoryOperation;
use cuda_sys::cuda::CUstream;
use numa_gpu::runtime::memory::Mem;
use numa_gpu::runtime::nvml::ThrottleReasons;
use rustacuda::context::CurrentContext;
use rustacuda::event::{Event, EventFlags};
use rustacuda::memory::{CopyDestination, DeviceBox};
use rustacuda::stream::{Stream, StreamFlags};
use std::mem::transmute_copy;

#[cfg(not(target_arch = "aarch64"))]
use nvml_wrapper::{enum_wrappers::device::Clock, NVML};

#[cfg(target_arch = "aarch64")]
use numa_gpu::runtime::hw_info::CudaDeviceInfo;

pub(super) type GpuBandwidthFn = unsafe extern "C" fn(
    op: MemoryOperation,
    data: *mut u32,
    size: usize,
    cycles: *mut u64,
    grid: u32,
    block: u32,
    stream: CUstream,
);

#[derive(Debug)]
pub(super) struct GpuMemoryBandwidth {
    device_id: u32,
    stream: Stream,

    #[cfg(not(target_arch = "aarch64"))]
    nvml: nvml_wrapper::NVML,
}

impl GpuMemoryBandwidth {
    #[cfg(not(target_arch = "aarch64"))]
    pub(super) fn new(device_id: u32) -> Self {
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).expect("Couldn't create CUDA stream");
        let nvml = NVML::init().expect("Couldn't initialize NVML");

        Self {
            device_id,
            stream,
            nvml,
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub(super) fn new(device_id: u32) -> Self {
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).expect("Couldn't create CUDA stream");

        Self { device_id, stream }
    }

    pub(super) fn run(
        f: GpuBandwidthFn,
        op: MemoryOperation,
        state: &mut Self,
        mem: &Mem<u32>,
        mp: &GpuMeasurementParameters,
    ) -> (u32, Option<ThrottleReasons>, u64, u64) {
        assert!(
            mem.len().is_power_of_two(),
            "Data size must be a power of two!"
        );

        // Get GPU clock rate that applications run at
        #[cfg(not(target_arch = "aarch64"))]
        let clock_rate_mhz = state
            .nvml
            .device_by_index(state.device_id as u32)
            .expect("Couldn't get NVML device")
            .clock_info(Clock::SM)
            .expect("Couldn't get clock rate with NVML");

        #[cfg(target_arch = "aarch64")]
        let clock_rate_mhz = CurrentContext::get_device()
            .expect("Couldn't get CUDA device")
            .clock_rate()
            .expect("Couldn't get clock rate");

        let mut device_cycles = DeviceBox::new(&0_u64).expect("Couldn't allocate device memory");

        let timer_begin = Event::new(EventFlags::DEFAULT).expect("Couldn't create CUDA event");
        let timer_end = Event::new(EventFlags::DEFAULT).expect("Couldn't create CUDA event");
        timer_begin
            .record(&state.stream)
            .expect("Couldn't record CUDA event");

        unsafe {
            // FIXME: Find a safer solution to replace transmute_copy!!!
            let cu_stream = transmute_copy::<Stream, CUstream>(&state.stream);
            f(
                op,
                mem.as_ptr() as *mut u32,
                mem.len(),
                device_cycles.as_device_ptr().as_raw_mut(),
                mp.grid_size.0,
                mp.block_size.0,
                cu_stream,
            )
        };

        timer_end
            .record(&state.stream)
            .expect("Couldn't record CUDA event");

        CurrentContext::synchronize().expect("Couldn't synchronize CUDA context");

        // Check if GPU is running in a throttled state
        #[cfg(not(target_arch = "aarch64"))]
        let throttle_reasons: Option<ThrottleReasons> = Some(
            state
                .nvml
                .device_by_index(state.device_id as u32)
                .expect("Couldn't get NVML device")
                .current_throttle_reasons()
                .expect("Couldn't get current throttle reasons with NVML")
                .into(),
        );

        #[cfg(target_arch = "aarch64")]
        let throttle_reasons = None;

        let ms = timer_end
            .elapsed_time_f32(&timer_begin)
            .expect("Couldn't get elapsed time");
        let ns = ms as f64 * 10.0_f64.powf(6.0);

        let mut cycles = 0;
        device_cycles
            .copy_to(&mut cycles)
            .expect("Couldn't transfer result from device");

        (clock_rate_mhz, throttle_reasons, cycles, ns as u64)
    }
}
