extern crate cuda_sys;

use std;

use self::cuda_sys::cudart::{
    cudaFree, cudaGetDeviceCount, cudaMallocManaged, cudaMemAttachGlobal, cudaSetDevice,
};

extern "C" {
    pub fn gpu_loop(data: *mut CacheLine, iterations: u32, signal: *mut Signal, result: *mut u64);
    pub fn gpu_loop_sync();
    #[allow(dead_code)]
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

// Implement uvm_sync_latency using CUDA via FFI
#[allow(dead_code)]
pub fn cuda_ffi() {
    let iterations: u32 = 2;
    let device: i32 = 0;

    unsafe { cuda_sys::cuda::cuInit(0) };

    let mut device_count: i32 = 0;
    unsafe { cudaGetDeviceCount(&mut device_count) };

    if device_count < device {
        eprintln!("Error: invalid device {}", device);
        std::process::exit(1);
    }

    unsafe { cudaSetDevice(device) };

    unsafe {
        let void_ptr_ptr: *mut *mut std::os::raw::c_void = std::ptr::null_mut();
        cudaMallocManaged(
            void_ptr_ptr,
            std::mem::size_of::<CacheLine>(),
            cudaMemAttachGlobal,
        );
        let data_ptr = *(void_ptr_ptr as *mut *mut CacheLine);
        // let &mut cacheline = cacheline_ptr.as_mut().unwrap();

        cudaMallocManaged(void_ptr_ptr, std::mem::size_of::<Signal>(), 0);
        let signal_ptr = *(void_ptr_ptr as *mut *mut Signal);
        // let &mut signal = signal_ptr.as_mut().unwrap();

        cudaMallocManaged(void_ptr_ptr, std::mem::size_of::<u64>(), 0);
        let gpu_result_ptr = *void_ptr_ptr as *mut u64;

        // Initialize
        (*data_ptr).value = PingPong::None;
        *signal_ptr = Signal::Wait;
        *gpu_result_ptr = 0;

        let cpu_result = 0u64;

        gpu_loop(data_ptr, iterations, signal_ptr, gpu_result_ptr);

        // start cpu_loop in thread

        // Start signal
        (*data_ptr).value = PingPong::GPU;
        *signal_ptr = Signal::Start;

        gpu_loop_sync();
        // join cpu thread

        cudaFree(data_ptr as *mut std::os::raw::c_void);
        cudaFree(signal_ptr as *mut std::os::raw::c_void);
        cudaFree(gpu_result_ptr as *mut std::os::raw::c_void);

        println!("CPU: {}\nGPU: {}", cpu_result, *gpu_result_ptr);
    }
}
