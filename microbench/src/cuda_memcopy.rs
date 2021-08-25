/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use super::{ArgPageType, CopyMethod};
use crate::types::*;
use itertools::iproduct;
use numa_gpu::runtime::allocator::{Allocator, MemType};
use numa_gpu::runtime::cuda_wrapper;
use numa_gpu::runtime::memory::{
    LaunchableMem, LaunchableMutSlice, LaunchableSlice, Mem, MemLock, PageLock,
};
use numa_gpu::runtime::numa::NumaMemory;
use rustacuda::context::{Context, ContextFlags};
use rustacuda::device::Device;
use rustacuda::event::{Event, EventFlags};
use rustacuda::memory::{DeviceBuffer, DeviceCopy};
use rustacuda::stream::{Stream, StreamFlags};
use rustacuda::CudaFlags;
use serde_derive::Serialize;
use std::convert::TryInto;
use std::iter;
use std::mem::size_of;
use std::time::Instant;

#[derive(Clone, Debug, Default, Serialize)]
struct DataPoint<'h, 'c> {
    pub hostname: &'h str,
    pub device_codename: &'c str,
    pub mem_type: Option<BareMemType>,
    pub page_type: Option<ArgPageType>,
    pub memory_node: Option<u16>,
    pub copy_method: Option<CopyMethod>,
    pub warm_up: bool,
    pub transfers_overlap: Option<bool>,
    pub bytes: usize,
    pub malloc_ns: Option<u64>,
    pub pin_ns: Option<u64>,
    pub mlock_ns: Option<u64>,
    pub copy_ns: Option<u64>,
}

pub struct CudaMemcopy;
struct Measurement;

impl CudaMemcopy {
    pub fn measure<W>(
        device_id: u32,
        mem_type: MemType,
        copy_directions: &[CopyMethod],
        byte_sizes: &[usize],
        repeat: u32,
        writer: Option<&mut W>,
    ) where
        W: std::io::Write,
    {
        rustacuda::init(CudaFlags::empty()).expect("Couldn't initialize CUDA");
        let device = Device::get_device(device_id).expect("Couldn't set CUDA device");
        let _context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .expect("Couldn't create CUDA context");

        let hostname = hostname::get()
            .expect("Couldn't get hostname")
            .into_string()
            .expect("Couldn't convert hostname into UTF-8 string");
        let device_codename = device.name().expect("Couldn't get device code name");

        let mem_type_description: MemTypeDescription = (&mem_type).into();

        let template = DataPoint {
            hostname: hostname.as_str(),
            device_codename: device_codename.as_str(),
            mem_type: Some(mem_type_description.bare_mem_type),
            page_type: Some(mem_type_description.page_type),
            memory_node: mem_type_description.location,
            ..Default::default()
        };

        let element_bytes = size_of::<u32>();

        let data_points: Vec<_> = iproduct!(
            copy_directions.iter(),
            byte_sizes.iter().map(|bs| bs / element_bytes),
            0..(repeat + 1)
        )
        .zip(iter::once(true).chain(iter::repeat(false)))
        .map(|((copy_method, buf_len, _run), warm_up)| {
            let buf_bytes = buf_len * element_bytes;
            let (malloc_ns, pin_ns, mlock_ns, copy_ms, transfers_overlap) =
                Measurement::run(mem_type.clone(), *copy_method, buf_len);

            DataPoint {
                warm_up,
                transfers_overlap,
                copy_method: Some(copy_method.clone()),
                bytes: buf_bytes,
                malloc_ns,
                pin_ns,
                mlock_ns,
                copy_ns: copy_ms.map(|t| (t as f64 * 10.0_f64.powf(6.0)) as u64),
                ..template.clone()
            }
        })
        .collect();

        if let Some(w) = writer {
            let mut csv = csv::Writer::from_writer(w);
            data_points
                .iter()
                .try_for_each(|row| csv.serialize(row))
                .expect("Couldn't write serialized measurements")
        }
    }
}

impl Measurement {
    fn run(
        mem_type: MemType,
        copy_method: CopyMethod,
        buffer_len: usize,
    ) -> (
        Option<u64>,
        Option<u64>,
        Option<u64>,
        Option<f32>,
        Option<bool>,
    ) {
        let element_bytes = size_of::<u32>();
        let buf_bytes = buffer_len * element_bytes;

        let (mut hmem, malloc_ns, dynamic_pin_ns) =
            if let MemType::NumaPinnedMem { node, page_type } = mem_type {
                let alloc_timer = Instant::now();
                let mut mem = NumaMemory::new(buffer_len, node, page_type);
                let alloc_duration = alloc_timer.elapsed();
                let alloc_ns = alloc_duration.as_nanos() as u64;

                let pin_timer = Instant::now();
                mem.page_lock().expect("Failed to pin memory");
                let pin_duration = pin_timer.elapsed();
                let pin_ns = pin_duration.as_nanos() as u64;

                (Mem::NumaMem(mem), Some(alloc_ns), Some(pin_ns))
            } else {
                let timer = Instant::now();
                let mem = Allocator::alloc_mem(mem_type.clone(), buffer_len);
                let duration = timer.elapsed();
                let ns: u64 = duration.as_nanos() as u64;

                (mem, Some(ns), None)
            };

        let mlock_ns = {
            let timer = Instant::now();
            hmem.mlock().expect("Failed to mlock the memory");
            let duration = timer.elapsed();

            Some(duration.as_nanos() as u64)
        };

        // Initialize the memory with some non-zero data
        if let std::result::Result::Ok(slice) = (&mut hmem).try_into() {
            let _: &mut [_] = slice;

            slice.iter_mut().by_ref().zip(0..).for_each(|(x, i)| *x = i);
        };

        let mut dmem = unsafe {
            DeviceBuffer::uninitialized(buffer_len).expect("Failed to allocate device memory")
        };

        let stream_0 =
            Stream::new(StreamFlags::NON_BLOCKING, None).expect("Failed to create CUDA stream");
        let stream_1 =
            Stream::new(StreamFlags::NON_BLOCKING, None).expect("Failed to create CUDA stream");

        let (copy_ms, transfers_overlap) = match copy_method {
            CopyMethod::HostToDevice => {
                let (start_event, stop_event) = Self::time_cuda_memcpy(
                    dmem.as_launchable_mut_slice(),
                    hmem.as_launchable_slice(),
                    &stream_0,
                );
                stream_0
                    .synchronize()
                    .expect("Failed to synchronize CUDA stream");
                let ms = stop_event
                    .elapsed_time_f32(&start_event)
                    .expect("Failed to calculate elapsed time");

                (Some(ms), None)
            }
            CopyMethod::DeviceToHost => {
                let (start_event, stop_event) = Self::time_cuda_memcpy(
                    hmem.as_launchable_mut_slice(),
                    dmem.as_launchable_slice(),
                    &stream_0,
                );
                stream_0
                    .synchronize()
                    .expect("Failed to synchronize CUDA stream");
                let ms = stop_event
                    .elapsed_time_f32(&start_event)
                    .expect("Failed to calculate elapsed time");

                (Some(ms), None)
            }
            CopyMethod::Bidirectional => {
                let (hmem_fst, hmem_snd) = unsafe { hmem.as_launchable_mut_slice().as_mut_slice() }
                    .split_at_mut(buffer_len / 2);
                let (dmem_fst, dmem_snd) = unsafe { dmem.as_launchable_mut_slice().as_mut_slice() }
                    .split_at_mut(buffer_len / 2);

                let (h2d_start_event, h2d_stop_event) = Self::time_cuda_memcpy(
                    hmem_fst.as_launchable_mut_slice(),
                    dmem_fst.as_launchable_slice(),
                    &stream_0,
                );

                let (d2h_start_event, d2h_stop_event) = Self::time_cuda_memcpy(
                    dmem_snd.as_launchable_mut_slice(),
                    hmem_snd.as_launchable_slice(),
                    &stream_1,
                );

                stream_0
                    .synchronize()
                    .expect("Failed to synchronize CUDA stream");
                stream_1
                    .synchronize()
                    .expect("Failed to synchronize CUDA stream");

                let h2d_ms = h2d_stop_event
                    .elapsed_time_f32(&h2d_start_event)
                    .expect("Failed to calculate elapsed time");
                let ms = d2h_stop_event
                    .elapsed_time_f32(&d2h_start_event)
                    .expect("Failed to calculate elapsed time");

                if h2d_ms * 1.5 < ms {
                    eprintln!("Warning: Non-overlapping transfer detected using {:?} allocation with {} MiB.", &mem_type, buf_bytes / 2_usize.pow(20));
                    (Some(ms), Some(false))
                } else {
                    (Some(ms), Some(true))
                }
            }
        };

        (
            malloc_ns,
            dynamic_pin_ns,
            mlock_ns,
            copy_ms,
            transfers_overlap,
        )
    }

    fn time_cuda_memcpy<T: DeviceCopy>(
        dst: LaunchableMutSlice<T>,
        src: LaunchableSlice<T>,
        stream: &Stream,
    ) -> (Event, Event) {
        let start_event = Event::new(EventFlags::DEFAULT).expect("Failed to create CUDA event");
        let stop_event = Event::new(EventFlags::DEFAULT).expect("Failed to create CUDA event");

        start_event
            .record(stream)
            .expect("Failed to record CUDA event");

        unsafe {
            cuda_wrapper::async_copy(dst.as_mut_slice(), src.as_slice(), stream)
                .expect("Couldn't perform async CUDA memcpy");
        }

        stop_event
            .record(stream)
            .expect("Failed to record CUDA event");

        (start_event, stop_event)
    }
}
