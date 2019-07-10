#[cfg(feature = "nvml")]
use nvml_wrapper::{enum_wrappers::device::Clock, NVML};
#[cfg(not(feature = "nvml"))]
use numa_gpu::runtime::hw_info::CudaDeviceInfo;

use rustacuda::context::CurrentContext;
use rustacuda::memory::{DeviceCopy, UnifiedBuffer};
use rustacuda::prelude::*;

use std::sync::atomic::{AtomicPtr, Ordering};
use std::thread;

extern "C" {
    pub fn gpu_loop(data: *mut CacheLine, iterations: u32, signal: *mut Signal, result: *mut u64);
    pub fn cpu_loop(data: *mut CacheLine, iterations: u32, signal: *mut Signal, result: *mut u64);
}

#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum PingPong {
    None = 0,
    // CPU = 1,
    GPU = 2,
}

unsafe impl DeviceCopy for PingPong {}

#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum Signal {
    Wait = 0,
    Start = 1,
}

unsafe impl DeviceCopy for Signal {}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CacheLine {
    value: PingPong,
    other: [u32; 31],
}

unsafe impl DeviceCopy for CacheLine {}

#[derive(Debug)]
pub struct DataPoint {
    pub iterations: u32,
    pub cpu_nanos: u64,
    pub gpu_nanos: u64,
}

fn cpu_worker(data: AtomicPtr<CacheLine>, iterations: u32, signal: AtomicPtr<Signal>) -> u64 {
    let mut cpu_result = 0u64;

    unsafe {
        cpu_loop(
            data.load(Ordering::SeqCst),
            iterations,
            signal.load(Ordering::SeqCst),
            &mut cpu_result,
        )
    };

    cpu_result
}

pub fn uvm_sync_latency(device_id: u32, iterations: u32) -> DataPoint {
    let device = Device::get_device(device_id).expect("Couldn't set CUDA device");
    let _context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
            .expect("Couldn't create CUDA context");

    #[cfg(feature = "nvml")]
    let nvml = NVML::init().expect("Couldn't initialize NVML");

    #[cfg(feature = "nvml")]
    let nvml_device = nvml
        .device_by_index(device_id)
        .expect("Couldn't get NVML device");

    // Allocate UVM pinned memory
    let mut data = UnifiedBuffer::new(
        &CacheLine {
            value: PingPong::None,
            other: [0; 31],
        },
        1,
    )
    .unwrap();
    let mut signal = UnifiedBuffer::new(&Signal::Wait, 1).unwrap();
    let mut gpu_result = UnifiedBuffer::new(&0_u64, 1).unwrap();

    // Make Rust think we're thread-safe
    // Note that we must handle thread-safety outside of Rust
    let data_ptr = AtomicPtr::new(&mut data[0]);
    let signal_ptr = AtomicPtr::new(&mut signal[0]);
    let gpu_result_ptr = AtomicPtr::new(&mut gpu_result[0]);

    // Launch GPU code
    unsafe {
        gpu_loop(
            data_ptr.load(Ordering::SeqCst),
            iterations,
            signal_ptr.load(Ordering::SeqCst),
            gpu_result_ptr.load(Ordering::SeqCst),
        )
    };

    // Launch CPU code
    let handle = thread::spawn(move || cpu_worker(data_ptr, iterations, signal_ptr));

    // Signal loop start
    data[0].value = PingPong::GPU;
    signal[0] = Signal::Start;

    // Wait on GPU
    CurrentContext::synchronize().unwrap();

    // Get GPU clock rate that applications run at
    #[cfg(feature = "nvml")]
    let clock_rate_mhz = nvml_device
        .applications_clock(Clock::SM)
        .expect("Couldn't get clock rate with NVML");

    #[cfg(not(feature = "nvml"))]
    let clock_rate_mhz = CurrentContext::get_device()
        .expect("Couldn't get CUDA device")
        .clock_rate()
        .expect("Couldn't get clock rate");

    // Wait on CPU
    let cpu_result = handle.join().unwrap();

    DataPoint {
        iterations,
        cpu_nanos: cpu_result,
        gpu_nanos: gpu_result[0] * 1000 / (clock_rate_mhz as u64),
    }
}
