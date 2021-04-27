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
use super::{Benchmark, MemoryOperation};
use crate::types::Cycles;
use numa_gpu::runtime::memory::Mem;
use numa_gpu::runtime::nvml::{DeviceClocks, ThrottleReasons};
use rustacuda::context::CurrentContext;
use rustacuda::event::{Event, EventFlags};
use rustacuda::launch;
use rustacuda::memory::{CopyDestination, DeviceBox};
use rustacuda::module::Module;
use rustacuda::stream::{Stream, StreamFlags};
use std::ffi::CString;

#[cfg(not(target_arch = "aarch64"))]
use nvml_wrapper::{enum_wrappers::device::Clock, NVML};

#[cfg(target_arch = "aarch64")]
use numa_gpu::runtime::hw_info::CudaDeviceInfo;

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
        bench: Benchmark,
        op: MemoryOperation,
        state: &mut Self,
        mem: &Mem<u32>,
        mp: &GpuMeasurementParameters,
    ) -> (u32, Option<ThrottleReasons>, u64, Cycles, u64) {
        assert!(
            mem.len().is_power_of_two(),
            "Data size must be a power of two!"
        );

        // Set a stable GPU clock rate to make the measurements more accurate
        #[cfg(not(target_arch = "aarch64"))]
        state
            .nvml
            .device_by_index(state.device_id as u32)
            .expect("Couldn't get NVML device")
            .set_max_gpu_clocks()
            .expect("Failed to set the maximum GPU clockrate");

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

        // FIXME: load the module lazy globally
        let module_path = CString::new(env!("CUDAUTILS_PATH"))
            .expect("Failed to load CUDA module, check your CUDAUTILS_PATH");
        let module = Module::load_from_file(&module_path).expect("Failed to load CUDA module");
        let stream = &state.stream;

        let mut memory_accesses_device =
            DeviceBox::new(&0_u64).expect("Couldn't allocate device memory");
        let mut measured_cycles = DeviceBox::new(&0_u64).expect("Couldn't allocate device memory");

        let timer_begin = Event::new(EventFlags::DEFAULT).expect("Couldn't create CUDA event");
        let timer_end = Event::new(EventFlags::DEFAULT).expect("Couldn't create CUDA event");
        timer_begin
            .record(&state.stream)
            .expect("Couldn't record CUDA event");

        let function_name = match (bench, op) {
            (Benchmark::Sequential, MemoryOperation::Read) => "gpu_read_bandwidth_seq_kernel",
            (Benchmark::Sequential, MemoryOperation::Write) => "gpu_write_bandwidth_seq_kernel",
            (Benchmark::Sequential, MemoryOperation::CompareAndSwap) => {
                "gpu_cas_bandwidth_seq_kernel"
            }
            (Benchmark::LinearCongruentialGenerator, MemoryOperation::Read) => {
                "gpu_read_bandwidth_lcg_kernel"
            }
            (Benchmark::LinearCongruentialGenerator, MemoryOperation::Write) => {
                "gpu_write_bandwidth_lcg_kernel"
            }
            (Benchmark::LinearCongruentialGenerator, MemoryOperation::CompareAndSwap) => {
                "gpu_cas_bandwidth_lcg_kernel"
            }
        };

        let c_name =
            CString::new(function_name).expect("Failed to convert Rust string into C string");
        let function = module
            .get_function(&c_name)
            .expect(format!("Failed to load the GPU function: {}", function_name).as_str());
        unsafe {
            launch!(
                function<<<mp.grid_size.0, mp.block_size.0, 0, stream>>>(
            mem.as_launchable_ptr(),
            mem.len(),
            mp.loop_length,
            mp.target_cycles.0,
            memory_accesses_device.as_device_ptr(),
            measured_cycles.as_device_ptr()
            )
                   )
            .expect("Failed to run GPU kernel");
        }

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

        let mut memory_accesses = 0;
        memory_accesses_device
            .copy_to(&mut memory_accesses)
            .expect("Couldn't transfer result from device");

        let mut cycles = 0;
        measured_cycles
            .copy_to(&mut cycles)
            .expect("Couldn't transfer result from device");

        (
            clock_rate_mhz,
            throttle_reasons,
            memory_accesses,
            Cycles(cycles),
            ns as u64,
        )
    }
}

impl Drop for GpuMemoryBandwidth {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "aarch64"))]
        self.nvml
            .device_by_index(self.device_id as u32)
            .unwrap()
            .set_default_gpu_clocks()
            .expect("Failed to reset default GPU clock rates");
    }
}
