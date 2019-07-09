extern crate accel;
extern crate nvml_wrapper;

use self::accel::device::sync;
use self::accel::UVec;

use self::nvml_wrapper::{enum_wrappers::device::Clock, NVML};

use std::sync::atomic::{AtomicPtr, Ordering};
use std::thread;

extern "C" {
    pub fn gpu_loop(data: *mut CacheLine, iterations: u32, signal: *mut Signal, result: *mut u64);
    pub fn cpu_loop(data: *mut CacheLine, iterations: u32, signal: *mut Signal, result: *mut u64);
}

#[repr(u32)]
pub enum PingPong {
    None = 0,
    // CPU = 1,
    GPU = 2,
}

#[repr(u32)]
pub enum Signal {
    Wait = 0,
    Start = 1,
}

#[repr(C)]
pub struct CacheLine {
    value: PingPong,
    other: [u32; 31],
}

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

pub fn uvm_sync_latency(iterations: u32) -> DataPoint {
    let device_id: u32 = 0;

    let nvml = NVML::init().expect("Couldn't initialize NVML");

    let nvml_device = nvml
        .device_by_index(device_id)
        .expect("Couldn't get NVML device");

    // Allocate UVM pinned memory
    let mut data = UVec::<CacheLine>::new(1).unwrap();
    let mut signal = UVec::<Signal>::new(1).unwrap();
    let mut gpu_result = UVec::<u64>::new(1).unwrap();

    // Initialize
    data[0].value = PingPong::None;
    signal[0] = Signal::Wait;
    gpu_result[0] = 0;

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
    sync().unwrap();

    // Get GPU clock rate that applications run at
    let clock_rate_mhz = nvml_device
        .applications_clock(Clock::SM)
        .expect("Couldn't get clock rate with NVML");

    // Wait on CPU
    let cpu_result = handle.join().unwrap();

    DataPoint {
        iterations,
        cpu_nanos: cpu_result,
        gpu_nanos: gpu_result[0] * 1000 / (clock_rate_mhz as u64),
    }
}
