use numa_gpu::runtime::allocator::{Allocator, MemType};
use numa_gpu::runtime::hw_info;
use numa_gpu::runtime::memory::{DerefMem, Mem};
use numa_gpu::runtime::utils::EnsurePhysicallyBacked;

#[cfg(not(feature = "nvml"))]
use numa_gpu::runtime::hw_info::CudaDeviceInfo;
#[cfg(feature = "nvml")]
use nvml_wrapper::{enum_wrappers::device::Clock, NVML};

use numa_gpu::runtime::nvml::ThrottleReasons;
use numa_gpu::runtime::{cuda_wrapper, numa};

use rustacuda::context::CurrentContext;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;

use serde_derive::Serialize;

use std::convert::{TryFrom, TryInto};
use std::mem::size_of;
use std::ops::RangeInclusive;

use crate::types::*;

extern "C" {
    pub fn gpu_stride(data: *const u32, iterations: u32, cycles: *mut u64);
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

        let device = Device::get_device(gpu_id).expect("Couldn't set CUDA device {}");
        let _context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .expect("Couldn't create CUDA context");

        numa::set_strict(true);

        let buffer_bytes = *range.end() + 1;
        let element_bytes = size_of::<u32>();
        let buffer_len = buffer_bytes / element_bytes;

        let hostname = hostname::get_hostname().expect("Couldn't get hostname");
        let device_type = match device_id {
            DeviceId::Cpu(_) => "CPU",
            DeviceId::Gpu(_) => "GPU",
        };
        let device_codename = match device_id {
            DeviceId::Cpu(_) => hw_info::cpu_codename(),
            DeviceId::Gpu(_) => device.name().expect("Couldn't get device code name"),
        };
        let cpu_node = match device_id {
            DeviceId::Cpu(node) => Some(node),
            _ => None,
        };

        let mem_type_description: MemTypeDescription = (&mem_type).into();

        let template = DataPoint {
            hostname: hostname.as_str(),
            device_type,
            device_codename: device_codename.as_str(),
            cpu_node,
            memory_node: mem_type_description.location,
            memory_type: Some(mem_type_description.bare_mem_type),
            ..Default::default()
        };

        let mnt = Measurement::new(range, stride, template);

        let mem = match DerefMem::<u32>::try_from(Allocator::alloc_mem(mem_type, buffer_len)) {
            Ok(mut demem) => {
                u32::ensure_physically_backed(demem.as_mut_slice());
                demem.into()
            }
            Err((_, mem)) => mem,
        };

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

#[derive(Debug, Default, Serialize)]
struct DataPoint<'h, 'd, 'c> {
    pub hostname: &'h str,
    pub device_type: &'d str,
    pub device_codename: &'c str,
    pub cpu_node: Option<u16>,
    pub memory_type: Option<BareMemType>,
    pub memory_node: Option<u16>,
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
struct Measurement<'h, 'd, 'c> {
    stride: RangeInclusive<usize>,
    range: RangeInclusive<usize>,
    template: DataPoint<'h, 'd, 'c>,
}

#[derive(Debug)]
struct GpuMemoryLatency {
    device_id: u32,

    #[cfg(feature = "nvml")]
    nvml: nvml_wrapper::NVML,
}

#[derive(Debug)]
struct CpuMemoryLatency {
    device_id: u16,
}

#[derive(Debug)]
struct MeasurementParameters {
    range: usize,
    stride: usize,
    iterations: u32,
}

impl<'h, 'd, 'c> Measurement<'h, 'd, 'c> {
    fn new(
        range: RangeInclusive<usize>,
        stride: RangeInclusive<usize>,
        template: DataPoint<'h, 'd, 'c>,
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

                let mp = MeasurementParameters {
                    range,
                    stride,
                    iterations,
                };

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
                        ..self.template
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
    #[cfg(feature = "nvml")]
    fn new(device_id: u32) -> Self {
        let nvml = NVML::init().expect("Couldn't initialize NVML");

        Self { device_id, nvml }
    }

    #[cfg(not(feature = "nvml"))]
    fn new(device_id: u32) -> Self {
        Self { device_id }
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
        state: &mut Self,
        mem: &Mem<u32>,
        mp: &MeasurementParameters,
    ) -> (u32, Option<ThrottleReasons>, u64, u64) {
        // Get current GPU clock rate
        #[cfg(feature = "nvml")]
        let clock_rate_mhz = state
            .nvml
            .device_by_index(state.device_id as u32)
            .expect("Couldn't get NVML device")
            .clock_info(Clock::SM)
            .expect("Couldn't get clock rate with NVML");

        #[cfg(not(feature = "nvml"))]
        let clock_rate_mhz = CurrentContext::get_device()
            .expect("Couldn't get CUDA device")
            .clock_rate()
            .expect("Couldn't get clock rate");

        // Launch GPU code
        let mut dev_cycles = DeviceBox::new(&0_u64).expect("Couldn't allocate device memory");
        unsafe {
            gpu_stride(
                mem.as_ptr(),
                mp.iterations,
                dev_cycles.as_device_ptr().as_raw_mut(),
            )
        };
        CurrentContext::synchronize().unwrap();

        // Check if GPU is running in a throttled state
        #[cfg(feature = "nvml")]
        let throttle_reasons: ThrottleReasons = state
            .nvml
            .device_by_index(state.device_id as u32)
            .expect("Couldn't get NVML device")
            .current_throttle_reasons()
            .expect("Couldn't get current throttle reasons with NVML")
            .into();

        #[cfg(not(feature = "nvml"))]
        let throttle_reasons = None;

        let mut cycles = 0;
        dev_cycles
            .copy_to(&mut cycles)
            .expect("Couldn't copy result data from device");
        let ns: u64 = cycles * 1000 / (clock_rate_mhz as u64);

        (clock_rate_mhz, Some(throttle_reasons), cycles, ns)
    }
}

impl CpuMemoryLatency {
    fn new(device_id: u16) -> Self {
        numa::run_on_node(device_id).expect("Couldn't set NUMA node");

        Self { device_id }
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
