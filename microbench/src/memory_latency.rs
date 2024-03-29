// Copyright 2018-2022 Clemens Lutz
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

use numa_gpu::runtime::allocator::{Allocator, MemType};
use numa_gpu::runtime::hw_info::NvidiaDriverInfo;
use numa_gpu::runtime::memory::{Mem, MemLock};
use numa_gpu::runtime::nvml::ThrottleReasons;
use numa_gpu::runtime::{cuda_wrapper, hw_info, linux_wrapper, numa};

#[cfg(not(target_arch = "aarch64"))]
use nvml_wrapper::{enum_wrappers::device::Clock, NVML};

#[cfg(target_arch = "aarch64")]
use numa_gpu::runtime::hw_info::CudaDeviceInfo;

use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use rustacuda::stream::{Stream, StreamFlags};
use rustacuda::{launch, CudaFlags};

use serde_derive::Serialize;

use std::convert::TryInto;
use std::ffi::CString;
use std::mem::size_of;
use std::ops::RangeInclusive;

use crate::types::*;
use crate::ArgPageType;

extern "C" {
    pub fn cpu_stride(data: *const u32, iterations: u32) -> u64;
}

pub struct MemoryLatency;

impl MemoryLatency {
    pub fn measure<W>(
        device_id: DeviceId,
        mem_type: MemType,
        range: RangeInclusive<usize>,
        stride: RangeInclusive<usize>,
        repeat: u32,
        writer: Option<&mut W>,
    ) where
        W: std::io::Write,
    {
        if let (MemType::CudaDevMem, DeviceId::Cpu(_)) = (mem_type.clone(), &device_id) {
            panic!("Cannot run benchmark on CPU with the given type of memory. Did you specify GPU device memory?");
        }

        let gpu_id = match device_id {
            DeviceId::Gpu(id) => id,
            _ => 0,
        };

        let (_context, device) = match rustacuda::init(CudaFlags::empty()) {
            Ok(_) => {
                let device = Device::get_device(gpu_id).expect("Couldn't set CUDA device");
                let context = Context::create_and_push(
                    ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
                    device,
                )
                .expect("Couldn't create CUDA context");
                (Some(context), Some(device))
            }
            Err(error) => {
                eprintln!("Warning: {}", error);
                (None, None)
            }
        };

        numa::set_strict(true);

        let buffer_bytes = *range.end() + 1;
        let element_bytes = size_of::<u32>();
        let buffer_len = buffer_bytes / element_bytes;

        let hostname = hostname::get()
            .expect("Couldn't get hostname")
            .into_string()
            .expect("Couldn't convert hostname into UTF-8 string");
        let device_type = match device_id {
            DeviceId::Cpu(_) => "CPU",
            DeviceId::Gpu(_) => "GPU",
        };
        let cpu_node = match device_id {
            DeviceId::Cpu(node) => Some(node),
            _ => None,
        };
        let device_codename = match device_id {
            DeviceId::Cpu(_) => Some(hw_info::cpu_codename().expect("Couldn't get CPU codename")),
            DeviceId::Gpu(_) => device.map(|d| d.name().expect("Couldn't get device codename")),
        };

        let mem_type_description: MemTypeDescription = (&mem_type).into();

        let template = DataPoint {
            hostname: Some(hostname),
            device_type: Some(device_type.to_string()),
            device_codename,
            cpu_node,
            memory_node: mem_type_description.location,
            memory_type: Some(mem_type_description.bare_mem_type),
            page_type: Some(mem_type_description.page_type),
            ..Default::default()
        };

        let mnt = Measurement::new(range, stride, template);

        let mut mem = Allocator::alloc_mem(mem_type, buffer_len);
        mem.mlock().expect("Failed to mlock the memory");

        // Initialize the memory with some non-zero data
        if let Ok(slice) = (&mut mem).try_into() {
            let _: &mut [_] = slice;
            slice.iter_mut().by_ref().zip(0..).for_each(|(x, i)| *x = i);
        }

        let latencies = match device_id {
            DeviceId::Cpu(did) => {
                let ml = CpuMemoryLatency::new(did);
                mnt.measure(
                    mem,
                    ml,
                    CpuMemoryLatency::prepare,
                    CpuMemoryLatency::run,
                    repeat,
                )
            }
            DeviceId::Gpu(did) => {
                let device = device.expect("No device found");
                if let Ok(local_cpu_node) = device.numa_memory_affinity() {
                    linux_wrapper::numa_run_on_node(local_cpu_node).expect(&format!(
                        "Failed to bind main thread to CPU node {}",
                        local_cpu_node
                    ));
                } else {
                    eprintln!(
                        "Warning: Couldn't bind main thread to the CPU closest to GPU {}. This may
                        cause additional latency in measurements.",
                        did
                    );
                }

                let ml = GpuMemoryLatency::new(did);
                let prepare = match mem {
                    Mem::CudaUniMem(_) => GpuMemoryLatency::prepare_prefetch,
                    _ => GpuMemoryLatency::prepare,
                };
                mnt.measure(mem, ml, prepare, GpuMemoryLatency::run, repeat)
            }
        };

        if let Some(w) = writer {
            let mut csv = csv::Writer::from_writer(w);
            latencies
                .iter()
                .try_for_each(|row| csv.serialize(row))
                .expect("Couldn't write serialized measurements")
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct DataPoint {
    pub hostname: Option<String>,
    pub device_type: Option<String>,
    pub device_codename: Option<String>,
    pub cpu_node: Option<u16>,
    pub memory_type: Option<BareMemType>,
    pub memory_node: Option<u16>,
    pub page_type: Option<ArgPageType>,
    pub warm_up: bool,
    pub range_bytes: usize,
    pub stride_bytes: usize,
    pub iterations: u32,
    pub throttle_reasons: Option<String>,
    pub clock_rate_mhz: Option<u32>,
    pub cycles: u64,
    pub ns: u64,
}

#[derive(Debug)]
struct Measurement {
    stride: RangeInclusive<usize>,
    range: RangeInclusive<usize>,
    template: DataPoint,
}

#[derive(Debug)]
struct GpuMemoryLatency {
    // `module` must be dropped before `context`. Rust specifies the drop order as the field order
    // in the struct. See RFC 1857: https://github.com/rust-lang/rfcs/pull/1857
    module: Module,
    device_id: u32,

    #[cfg(not(target_arch = "aarch64"))]
    nvml: nvml_wrapper::NVML,
}

#[derive(Debug)]
struct CpuMemoryLatency;

#[derive(Debug)]
struct MeasurementParameters {
    stride: usize,
    iterations: u32,
}

impl Measurement {
    fn new(
        range: RangeInclusive<usize>,
        stride: RangeInclusive<usize>,
        template: DataPoint,
    ) -> Self {
        Self {
            stride,
            range,
            template,
        }
    }

    fn measure<P, R, S>(
        &self,
        mut mem: Mem<u32>,
        mut state: S,
        prepare: P,
        run: R,
        repeat: u32,
    ) -> Vec<DataPoint>
    where
        P: Fn(&mut S, &mut Mem<u32>, &MeasurementParameters),
        R: Fn(
            &mut S,
            &Mem<u32>,
            &MeasurementParameters,
        ) -> (u32, Option<ThrottleReasons>, u64, u64),
    {
        let stride_iter = self.stride.clone();
        let range_iter = self.range.clone();

        let latencies = stride_iter
            .filter(|stride| stride.is_power_of_two())
            .flat_map(|stride| {
                range_iter
                    .clone()
                    .filter(|range| range.is_power_of_two())
                    .zip(std::iter::repeat(stride))
                    .enumerate()
            })
            .flat_map(|(i, (range, stride))| {
                let iterations = (range / stride) as u32;
                let mut data_points: Vec<DataPoint> = Vec::with_capacity(repeat as usize + 1);
                let mut warm_up = true;

                let mp = MeasurementParameters { stride, iterations };

                if i == 0 {
                    prepare(&mut state, &mut mem, &mp);
                }

                for _ in 0..repeat + 1 {
                    let (clock_rate_mhz, throttle_reasons, cycles, ns) = run(&mut state, &mem, &mp);

                    data_points.push(DataPoint {
                        warm_up,
                        range_bytes: range,
                        stride_bytes: stride,
                        iterations,
                        throttle_reasons: throttle_reasons.map(|r| r.to_string()),
                        clock_rate_mhz: Some(clock_rate_mhz),
                        cycles,
                        ns,
                        ..self.template.clone()
                    });
                    warm_up = false;
                }

                data_points
            })
            .collect::<Vec<_>>();

        latencies
    }
}

impl GpuMemoryLatency {
    #[cfg(not(target_arch = "aarch64"))]
    fn new(device_id: u32) -> Self {
        let module = Self::load_module();
        let nvml = NVML::init().expect("Couldn't initialize NVML");

        Self {
            module,
            device_id,
            nvml,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn new(device_id: u32) -> Self {
        let module = Self::load_module();

        Self { module, device_id }
    }

    fn load_module() -> Module {
        let module_path = CString::new(env!("CUDAUTILS_PATH"))
            .expect("Failed to load CUDA module, check your CUDAUTILS_PATH");
        let module = Module::load_from_file(&module_path).expect("Failed to load CUDA module");

        module
    }

    fn prepare(_state: &mut Self, mem: &mut Mem<u32>, mp: &MeasurementParameters) {
        let len = mem.len();
        match mem.try_into() {
            Ok(slice) => {
                write_strides(slice, mp.stride);
            }
            Err((_, dev_slice)) => {
                let mut host_mem = vec![0; len];
                write_strides(&mut host_mem, mp.stride);
                dev_slice
                    .copy_from(&host_mem)
                    .expect("Couldn't write strides data to device");
            }
        }
    }

    fn prepare_prefetch(_state: &mut Self, mem: &mut Mem<u32>, mp: &MeasurementParameters) {
        let len = mem.len();
        match mem.try_into() {
            Ok(slice) => {
                write_strides(slice, mp.stride);
            }
            Err((_, dev_slice)) => {
                let mut host_mem = vec![0; len];
                write_strides(&mut host_mem, mp.stride);
                dev_slice
                    .copy_from(&host_mem)
                    .expect("Couldn't write strides data to device");
            }
        }

        if let Mem::CudaUniMem(ref mut um) = mem {
            let device_id = cuda_wrapper::current_device_id().expect("Couldn't get CUDA device id");
            let stream =
                Stream::new(StreamFlags::NON_BLOCKING, None).expect("Couldn't create CUDA stream");

            cuda_wrapper::prefetch_async(um.as_unified_ptr(), um.len(), device_id, &stream)
                .expect("Couldn't prefetch unified memory to device");
            stream.synchronize().unwrap();
        }
    }

    fn run(
        _state: &mut Self,
        mem: &Mem<u32>,
        mp: &MeasurementParameters,
    ) -> (u32, Option<ThrottleReasons>, u64, u64) {
        // Get current GPU clock rate
        #[cfg(not(target_arch = "aarch64"))]
        let clock_rate_mhz = _state
            .nvml
            .device_by_index(_state.device_id as u32)
            .expect("Couldn't get NVML device")
            .clock_info(Clock::SM)
            .expect("Couldn't get clock rate with NVML");

        #[cfg(target_arch = "aarch64")]
        let clock_rate_mhz = CurrentContext::get_device()
            .expect("Couldn't get CUDA device")
            .clock_rate()
            .expect("Couldn't get clock rate");

        // Launch GPU code
        let mut dev_cycles = DeviceBox::new(&0_u64).expect("Couldn't allocate device memory");
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).expect("Failed to create CUDA stream");

        unsafe {
            let module = &_state.module;
            launch!(module.gpu_stride<<<1, 1, 0, stream>>>(
                mem.as_launchable_ptr(),
                mp.iterations,
                dev_cycles.as_device_ptr()
            ))
            .expect("Failed to launch gpu_stride kernel");
        };
        stream
            .synchronize()
            .expect("Failed to synchronize CUDA stream");

        // Check if GPU is running in a throttled state
        #[cfg(not(target_arch = "aarch64"))]
        let throttle_reasons: Option<ThrottleReasons> = Some(
            _state
                .nvml
                .device_by_index(_state.device_id as u32)
                .expect("Couldn't get NVML device")
                .current_throttle_reasons()
                .expect("Couldn't get current throttle reasons with NVML")
                .into(),
        );

        #[cfg(target_arch = "aarch64")]
        let throttle_reasons = None;

        let mut cycles = 0;
        dev_cycles
            .copy_to(&mut cycles)
            .expect("Couldn't copy result data from device");
        let ns: u64 = cycles * 1000 / (clock_rate_mhz as u64);

        (clock_rate_mhz, throttle_reasons, cycles, ns)
    }
}

impl CpuMemoryLatency {
    fn new(device_id: u16) -> Self {
        numa::run_on_node(device_id).expect("Couldn't set NUMA node");

        Self
    }

    fn run(
        _state: &mut Self,
        mem: &Mem<u32>,
        mp: &MeasurementParameters,
    ) -> (u32, Option<ThrottleReasons>, u64, u64) {
        let ns = if let Mem::CudaDevMem(_) = mem {
            unreachable!();
        } else {
            // Launch CPU code
            unsafe { cpu_stride(mem.as_ptr(), mp.iterations) }
        };

        let cycles = 0;
        let clock_rate_mhz = 0;

        (clock_rate_mhz, None, cycles, ns)
    }

    fn prepare(_state: &mut Self, mem: &mut Mem<u32>, mp: &MeasurementParameters) {
        if let Ok(slice) = mem.try_into() {
            write_strides(slice, mp.stride);
        } else {
            unreachable!();
        }
    }
}

fn write_strides(data: &mut [u32], stride: usize) -> usize {
    let element_bytes = size_of::<u32>();
    let len = data.len();

    let number_of_strides = data
        .iter_mut()
        .zip((stride / element_bytes)..)
        .map(|(it, next)| *it = (next % len) as u32)
        .count();

    number_of_strides
}
