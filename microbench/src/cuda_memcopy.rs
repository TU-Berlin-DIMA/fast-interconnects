use cuda_sys::cuda::{
    cuEventCreate, cuEventDestroy_v2, cuEventElapsedTime, cuEventRecord, cuMemAllocHost_v2,
    cuMemAlloc_v2, cuMemFreeHost, cuMemFree_v2, cuMemGetInfo_v2, cuMemHostRegister_v2,
    cuMemHostUnregister, cuMemcpyAsync, cuStreamCreate, cuStreamDestroy_v2, cuStreamSynchronize,
    CUevent, CUstream,
};

use numa_gpu::error::ToResult;
use numa_gpu::runtime::numa::NumaMemory;
use numa_gpu::runtime::utils::EnsurePhysicallyBacked;

use rustacuda::prelude::*;

use serde_derive::Serialize;

use std::ffi::c_void;
use std::iter;
use std::mem::{size_of, zeroed};
use std::ptr::null_mut;
use std::slice;
use std::time::Instant;

use crate::types::*;

#[derive(Clone, Debug, Default, Serialize)]
struct DataPoint<'h, 'c> {
    pub hostname: &'h str,
    pub device_codename: &'c str,
    pub allocation_type: Option<MemoryAllocationType>,
    pub copy_method: Option<CopyMethod>,
    pub memory_node: u16,
    pub warm_up: bool,
    pub transfers_overlap: Option<bool>,
    pub bytes: usize,
    pub malloc_ns: Option<u64>,
    pub dynamic_pin_ns: Option<u64>,
    pub copy_ns: Option<u64>,
}

pub struct CudaMemcopy;

struct Measurement<'h, 'c> {
    memory_node: u16,
    template: DataPoint<'h, 'c>,
}

#[derive(Debug, Clone, Copy, Serialize, Eq, PartialEq)]
pub enum CopyMethod {
    HostToDevice,
    DeviceToHost,
    Bidirectional,
}

enum HostMem {
    NumaMem(NumaMemory<u32>),
    PinnedMem(*mut u32),
}

impl CudaMemcopy {
    pub fn measure<W>(device_id: u32, memory_node: u16, repeat: u32, writer: Option<&mut W>)
    where
        W: std::io::Write,
    {
        let device = Device::get_device(device_id).expect("Couldn't set CUDA device");
        let _context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .expect("Couldn't create CUDA context");

        let alloc_types = vec![
            MemoryAllocationType::Pageable,
            MemoryAllocationType::Pinned,
            MemoryAllocationType::DynamicallyPinned,
        ];
        let copy_method = vec![
            CopyMethod::HostToDevice,
            CopyMethod::DeviceToHost,
            CopyMethod::Bidirectional,
        ];

        let free_dmem_bytes = unsafe {
            let mut free = 0;
            let mut total = 0;
            cuMemGetInfo_v2(&mut free, &mut total)
                .to_result()
                .expect("Couldn't get free device memory size");
            free
        };

        let byte_sizes: Vec<_> = (2_usize.pow(20)..free_dmem_bytes)
            .filter(|x: &usize| x.is_power_of_two())
            .collect();

        let hostname = hostname::get_hostname().expect("Couldn't get hostname");
        let device_codename = device.name().expect("Couldn't get device code name");

        let template = DataPoint {
            hostname: hostname.as_str(),
            device_codename: device_codename.as_str(),
            memory_node,
            ..Default::default()
        };

        let mnt = Measurement::new(memory_node, template);
        let latencies = mnt.measure(alloc_types, copy_method, byte_sizes, repeat);

        if let Some(w) = writer {
            let mut csv = csv::Writer::from_writer(w);
            latencies
                .iter()
                .try_for_each(|row| csv.serialize(row))
                .expect("Couldn't write serialized measurements")
        }
    }
}

impl<'h, 'c> Measurement<'h, 'c> {
    fn new(memory_node: u16, template: DataPoint<'h, 'c>) -> Self {
        Self {
            memory_node,
            template,
        }
    }

    fn measure(
        &self,
        alloc_types: Vec<MemoryAllocationType>,
        copy_methods: Vec<CopyMethod>,
        byte_sizes: Vec<usize>,
        repeat: u32,
    ) -> Vec<DataPoint<'h, 'c>> {
        // FIXME: Open questions: how to tell CUDA to allocate on specific NUMA node?

        let element_bytes = size_of::<u32>();

        let data_points: Vec<_> = alloc_types
            .iter()
            .flat_map(|at| {
                iter::repeat(at).zip(copy_methods.iter().flat_map(|cm| {
                    iter::repeat(cm).zip(byte_sizes.iter().map(|bs| bs / element_bytes).flat_map(
                        |buffer_len| {
                            iter::repeat(buffer_len)
                                .zip(iter::once(true).chain(iter::repeat(false)))
                                .zip(0..(repeat + 1))
                        },
                    ))
                }))
            })
            .map(|(alloc_type, (copy_method, ((buf_len, warm_up), _run)))| {
                let buf_bytes = buf_len * element_bytes;
                let (malloc_ns, dynamic_pin_ns, copy_ms, transfers_overlap) =
                    self.run(alloc_type.clone(), *copy_method, buf_len);

                DataPoint {
                    warm_up,
                    transfers_overlap,
                    allocation_type: Some(alloc_type.clone()),
                    copy_method: Some(copy_method.clone()),
                    bytes: buf_bytes,
                    malloc_ns,
                    dynamic_pin_ns,
                    copy_ns: copy_ms.map(|t| (t as f64 * 10.0_f64.powf(6.0)) as u64),
                    ..self.template.clone()
                }
            })
            .collect();

        data_points
    }

    fn run(
        &self,
        alloc_type: MemoryAllocationType,
        copy_method: CopyMethod,
        buf_len: usize,
    ) -> (Option<u64>, Option<u64>, Option<f32>, Option<bool>) {
        let element_bytes = size_of::<u32>();
        let buf_bytes = buf_len * element_bytes;

        let (mut hmem, malloc_ns, dynamic_pin_ns) = match alloc_type {
            MemoryAllocationType::Pageable => {
                let timer = Instant::now();
                let m = HostMem::NumaMem(NumaMemory::new(buf_len, self.memory_node));
                let duration = timer.elapsed();
                let ns: u64 = duration.as_secs() * 10_u64.pow(9) + duration.subsec_nanos() as u64;

                (m, Some(ns), None)
            }
            MemoryAllocationType::Pinned => unsafe {
                let timer = Instant::now();
                let mut ptr = zeroed();
                cuMemAllocHost_v2(&mut ptr, buf_bytes)
                    .to_result()
                    .expect("Couldn't allocate pinned host memory");
                let duration = timer.elapsed();
                let ns: u64 = duration.as_secs() * 10_u64.pow(9) + duration.subsec_nanos() as u64;

                (HostMem::PinnedMem(ptr as *mut u32), Some(ns), None)
            },
            MemoryAllocationType::DynamicallyPinned => {
                let alloc_timer = Instant::now();
                let mut m = NumaMemory::new(buf_len, self.memory_node);
                let alloc_duration = alloc_timer.elapsed();
                let alloc_ns: u64 =
                    alloc_duration.as_secs() * 10_u64.pow(9) + alloc_duration.subsec_nanos() as u64;

                let pin_timer = Instant::now();
                unsafe { cuMemHostRegister_v2(m.as_mut_ptr() as *mut c_void, buf_bytes, 0) }
                    .to_result()
                    .expect("Couldn't dynamically pin memory");
                let pin_duration = pin_timer.elapsed();
                let pin_ns: u64 =
                    pin_duration.as_secs() * 10_u64.pow(9) + pin_duration.subsec_nanos() as u64;

                (HostMem::NumaMem(m), Some(alloc_ns), Some(pin_ns))
            }
        };

        u32::ensure_physically_backed(unsafe {
            slice::from_raw_parts_mut(hmem.as_mut_ptr(), buf_len)
        });

        let mut dmem: *mut c_void = null_mut();
        unsafe { cuMemAlloc_v2(&mut dmem as *mut *mut c_void as *mut u64, buf_bytes) }
            .to_result()
            .expect("Couldn't allocate device memory");

        let mut stream_0: CUstream = unsafe { zeroed() };
        let mut stream_1: CUstream = unsafe { zeroed() };
        unsafe {
            cuStreamCreate(&mut stream_0, 1)
                .to_result()
                .expect("Couldn't create stream");
            cuStreamCreate(&mut stream_1, 1)
                .to_result()
                .expect("Couldn't create stream");
        }

        let (copy_ms, transfers_overlap) = match copy_method {
            CopyMethod::HostToDevice => unsafe {
                let (start_event, stop_event) =
                    Self::time_cuda_memcpy(dmem as *mut u32, hmem.as_ptr(), buf_len, stream_0);

                cuStreamSynchronize(stream_0)
                    .to_result()
                    .expect("Couldn't sync CUDA stream");

                let mut ms = 0.0;
                cuEventElapsedTime(&mut ms, start_event, stop_event)
                    .to_result()
                    .expect("Couldn't calculate elapsed time");

                cuEventDestroy_v2(start_event)
                    .to_result()
                    .expect("Couldn't destroy CUDA event");
                cuEventDestroy_v2(stop_event)
                    .to_result()
                    .expect("Couldn't destroy CUDA event");

                (Some(ms), None)
            },
            CopyMethod::DeviceToHost => unsafe {
                let (start_event, stop_event) = Self::time_cuda_memcpy(
                    hmem.as_mut_ptr(),
                    dmem as *const u32,
                    buf_len,
                    stream_0,
                );

                cuStreamSynchronize(stream_0)
                    .to_result()
                    .expect("Couldn't sync CUDA stream");

                let mut ms = 0.0;
                cuEventElapsedTime(&mut ms, start_event, stop_event)
                    .to_result()
                    .expect("Couldn't calculate elapsed time");

                cuEventDestroy_v2(start_event)
                    .to_result()
                    .expect("Couldn't destroy CUDA event");
                cuEventDestroy_v2(stop_event)
                    .to_result()
                    .expect("Couldn't destroy CUDA event");

                (Some(ms), None)
            },
            CopyMethod::Bidirectional => unsafe {
                let (h2d_start_event, h2d_stop_event) =
                    Self::time_cuda_memcpy(dmem as *mut u32, hmem.as_ptr(), buf_len / 2, stream_0);
                let (d2h_start_event, d2h_stop_event) = Self::time_cuda_memcpy(
                    hmem.as_mut_ptr().offset((buf_len / 2) as isize),
                    (dmem as *const u32).offset((buf_len / 2) as isize),
                    buf_len / 2,
                    stream_1,
                );

                cuStreamSynchronize(stream_0)
                    .to_result()
                    .expect("Couldn't sync CUDA stream");
                cuStreamSynchronize(stream_1)
                    .to_result()
                    .expect("Couldn't sync CUDA stream");

                let mut h2d_ms = 0.0;
                cuEventElapsedTime(&mut h2d_ms, h2d_start_event, h2d_stop_event)
                    .to_result()
                    .expect("Couldn't calculate elapsed time");
                let mut ms = 0.0;
                cuEventElapsedTime(&mut ms, h2d_start_event, d2h_stop_event)
                    .to_result()
                    .expect("Couldn't calculate elapsed time");

                cuEventDestroy_v2(h2d_start_event)
                    .to_result()
                    .expect("Couldn't destroy CUDA event");
                cuEventDestroy_v2(h2d_stop_event)
                    .to_result()
                    .expect("Couldn't destroy CUDA event");
                cuEventDestroy_v2(d2h_start_event)
                    .to_result()
                    .expect("Couldn't destroy CUDA event");
                cuEventDestroy_v2(d2h_stop_event)
                    .to_result()
                    .expect("Couldn't destroy CUDA event");

                if h2d_ms * 1.5 < ms {
                    eprintln!("Warning: Non-overlapping transfer detected using {:?} allocation with {} MiB.", alloc_type, buf_bytes / 2_usize.pow(20));
                    (Some(ms), Some(false))
                } else {
                    (Some(ms), Some(true))
                }
            },
        };

        if alloc_type == MemoryAllocationType::DynamicallyPinned {
            unsafe { cuMemHostUnregister(hmem.as_mut_ptr() as *mut c_void) }
                .to_result()
                .expect("Couldn't unregister dynamically pinned memory");
        }

        if let HostMem::PinnedMem(ptr) = hmem {
            unsafe { cuMemFreeHost(ptr as *mut c_void) }
                .to_result()
                .expect("Couldn't free pinned host memory");
        }

        unsafe {
            cuStreamDestroy_v2(stream_0)
                .to_result()
                .expect("Couldn't destroy stream");
            cuStreamDestroy_v2(stream_1)
                .to_result()
                .expect("Couldn't destroy stream");
            cuMemFree_v2(dmem as u64)
                .to_result()
                .expect("Couldn't free device memory");
        }

        (malloc_ns, dynamic_pin_ns, copy_ms, transfers_overlap)
    }

    unsafe fn time_cuda_memcpy<T: Copy>(
        dst: *mut T,
        src: *const T,
        len: usize,
        stream: CUstream,
    ) -> (CUevent, CUevent) {
        let element_bytes = size_of::<T>();

        let mut start_event: CUevent = zeroed();
        let mut stop_event: CUevent = zeroed();
        cuEventCreate(&mut start_event, 0)
            .to_result()
            .expect("Couldn't create event");
        cuEventCreate(&mut stop_event, 0)
            .to_result()
            .expect("Couldn't create event");

        cuEventRecord(start_event, stream);
        cuMemcpyAsync(
            dst as *mut c_void as u64,
            src as *const c_void as u64,
            len * element_bytes,
            stream,
        )
        .to_result()
        .expect("Couldn't perform async CUDA memcpy");
        cuEventRecord(stop_event, stream)
            .to_result()
            .expect("Couldn't record event");

        (start_event, stop_event)
    }
}

impl HostMem {
    pub fn as_ptr(&self) -> *const u32 {
        match self {
            HostMem::NumaMem(m) => m.as_ptr(),
            HostMem::PinnedMem(ptr) => *ptr,
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut u32 {
        match self {
            HostMem::NumaMem(m) => m.as_mut_ptr(),
            HostMem::PinnedMem(ptr) => *ptr,
        }
    }
}
