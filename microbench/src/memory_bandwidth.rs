/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

mod cpu_measurement;
mod cpu_memory_bandwidth;
mod data_point;
mod gpu_measurement;
mod gpu_memory_bandwidth;

use self::cpu_measurement::{CpuMeasurement, CpuNamedBandwidthFn};
use self::cpu_memory_bandwidth::CpuMemoryBandwidth;
use self::data_point::DataPoint;
use self::gpu_measurement::GpuMeasurement;
use self::gpu_memory_bandwidth::GpuMemoryBandwidth;
use crate::types::{
    Cycles, DeviceId, Ilp, MemTypeDescription, OversubRatio, ThreadCount, Warp, WarpMul, SM,
};
use numa_gpu::runtime::allocator::{Allocator, MemType};
use numa_gpu::runtime::memory::DerefMem;
use numa_gpu::runtime::utils::EnsurePhysicallyBacked;
use numa_gpu::runtime::{hw_info, numa};
use rustacuda::context::{Context, ContextFlags};
use rustacuda::device::Device;
use rustacuda::device::DeviceAttribute;
use rustacuda::CudaFlags;
use serde_derive::Serialize;
use std::convert::TryInto;
use std::mem::size_of;
use std::ops::RangeInclusive;

extern "C" {
    fn cpu_bandwidth_seq(
        op: MemoryOperation,
        data: *mut u32,
        size: usize,
        tid: usize,
        num_threads: usize,
    );
    fn cpu_bandwidth_lcg(
        op: MemoryOperation,
        data: *mut u32,
        size: usize,
        tid: usize,
        num_threads: usize,
    );
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Serialize)]
enum MemoryOperation {
    Read,
    Write,
    CompareAndSwap,
}

// FIXME: use Benchmark for CPU measurment, too
#[derive(Clone, Copy, Debug, Serialize)]
enum Benchmark {
    Sequential,
    LinearCongruentialGenerator,
}

pub struct MemoryBandwidth;

impl MemoryBandwidth {
    pub fn measure<W>(
        device_id: DeviceId,
        mem_type: MemType,
        range_bytes: usize,
        threads: RangeInclusive<ThreadCount>,
        oversub_ratio: RangeInclusive<OversubRatio>,
        warp_mul: RangeInclusive<WarpMul>,
        ilp: RangeInclusive<Ilp>,
        loop_length: u32,
        target_cycles: Cycles,
        repeat: u32,
        writer: Option<&mut W>,
    ) where
        W: std::io::Write,
    {
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

        let item_bytes = size_of::<u32>();
        let buffer_len = range_bytes / item_bytes;

        let hostname = hostname::get()
            .expect("Couldn't get hostname")
            .into_string()
            .expect("Couldn't convert hostname into UTF-8 string");
        let (device_type, cpu_node) = match device_id {
            DeviceId::Cpu(id) => ("CPU", Some(id)),
            DeviceId::Gpu(_) => ("GPU", None),
        };
        let device_codename = match device_id {
            DeviceId::Cpu(_) => Some(hw_info::cpu_codename().expect("Couldn't get CPU codename")),
            DeviceId::Gpu(_) => device.map(|d| d.name().expect("Couldn't get device codename")),
        };
        let mem_type_description: MemTypeDescription = (&mem_type).into();

        let template = DataPoint {
            hostname: hostname.as_str(),
            device_type,
            device_codename,
            cpu_node,
            memory_node: mem_type_description.location,
            memory_type: Some(mem_type_description.bare_mem_type),
            page_type: Some(mem_type_description.page_type),
            range_bytes,
            item_bytes,
            ..Default::default()
        };

        let mut mem = Allocator::alloc_mem(mem_type, buffer_len);
        mem.ensure_physically_backed();

        let bandwidths = match device_id {
            DeviceId::Cpu(cpu_node) => {
                let mnt = CpuMeasurement::new(threads, template);
                let demem: DerefMem<_> = mem.try_into().expect("Cannot run benchmark on CPU with the given type of memory. Did you specify GPU device memory?");
                mnt.measure(
                    &demem,
                    CpuMemoryBandwidth::new(cpu_node),
                    CpuMemoryBandwidth::run,
                    vec![
                        CpuNamedBandwidthFn {
                            f: cpu_bandwidth_seq,
                            name: "sequential",
                        },
                        CpuNamedBandwidthFn {
                            f: cpu_bandwidth_lcg,
                            name: "linear_congruential_generator",
                        },
                    ],
                    vec![
                        MemoryOperation::Read,
                        MemoryOperation::Write,
                        MemoryOperation::CompareAndSwap,
                    ],
                    repeat,
                )
            }
            DeviceId::Gpu(did) => {
                let device = device.expect("No device found");
                let warp_size = Warp(
                    device
                        .get_attribute(DeviceAttribute::WarpSize)
                        .expect("Couldn't get device warp size"),
                );
                let sm_count = SM(device
                    .get_attribute(DeviceAttribute::MultiprocessorCount)
                    .expect("Couldn't get device multiprocessor count"));
                let mnt = GpuMeasurement::new(
                    oversub_ratio,
                    warp_mul,
                    warp_size,
                    sm_count,
                    ilp,
                    loop_length,
                    target_cycles,
                    template,
                );

                let ml = GpuMemoryBandwidth::new(did);
                let l = mnt.measure(
                    &mem,
                    ml,
                    GpuMemoryBandwidth::run,
                    vec![
                        Benchmark::Sequential,
                        Benchmark::LinearCongruentialGenerator,
                    ],
                    vec![
                        MemoryOperation::Write,
                        MemoryOperation::Read,
                        MemoryOperation::CompareAndSwap,
                    ],
                    repeat,
                );
                l
            }
        };

        if let Some(w) = writer {
            let mut csv = csv::Writer::from_writer(w);
            bandwidths
                .iter()
                .try_for_each(|row| csv.serialize(row))
                .expect("Couldn't write serialized measurements")
        }
    }
}
