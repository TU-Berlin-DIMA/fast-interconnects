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

use self::cpu_measurement::CpuMeasurement;
use self::cpu_memory_bandwidth::CpuMemoryBandwidth;
use self::data_point::DataPoint;
use self::gpu_measurement::GpuMeasurement;
use self::gpu_memory_bandwidth::GpuMemoryBandwidth;
use crate::types::{
    Cycles, DeviceId, Ilp, MemTypeDescription, OversubRatio, ThreadCount, Warp, WarpMul, SM,
};
use numa_gpu::runtime::allocator::{Allocator, MemType};
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::memory::DerefMem;
use numa_gpu::runtime::utils::EnsurePhysicallyBacked;
use numa_gpu::runtime::{hw_info, numa};
use rustacuda::context::{Context, ContextFlags};
use rustacuda::device::Device;
use rustacuda::device::DeviceAttribute;
use rustacuda::CudaFlags;
use serde_derive::Serialize;
use serde_repr::Serialize_repr;
use std::convert::TryInto;
use std::mem::size_of;
use std::ops::RangeInclusive;

#[repr(C)]
#[derive(Clone, Copy, Debug, Serialize)]
enum MemoryOperation {
    Read,
    Write,
    CompareAndSwap,
}

#[derive(Clone, Copy, Debug, Serialize)]
enum Benchmark {
    Sequential,
    LinearCongruentialGenerator,
}

#[derive(Clone, Copy, Debug, Serialize_repr)]
#[repr(usize)]
enum ItemBytes {
    Bytes4 = 4,
    Bytes8 = 8,
    Bytes16 = 16,
}

pub struct MemoryBandwidth;

impl MemoryBandwidth {
    pub fn measure<W>(
        device_id: DeviceId,
        mem_type: MemType,
        range_bytes: usize,
        threads: Vec<ThreadCount>,
        cpu_affinity: CpuAffinity,
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
        let benchmarks = vec![
            Benchmark::Sequential,
            Benchmark::LinearCongruentialGenerator,
        ];
        let operators = vec![
            MemoryOperation::Read,
            MemoryOperation::Write,
            MemoryOperation::CompareAndSwap,
        ];
        let item_bytes = vec![ItemBytes::Bytes4, ItemBytes::Bytes8, ItemBytes::Bytes16];

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

        let physical_item_bytes = size_of::<u32>();
        let buffer_len = range_bytes / physical_item_bytes;

        let hostname = hostname::get()
            .expect("Couldn't get hostname")
            .into_string()
            .expect("Couldn't convert hostname into UTF-8 string");
        let (device_type, cpu_node) = match device_id {
            DeviceId::Cpu(id) => ("CPU".to_string(), Some(id)),
            DeviceId::Gpu(_) => ("GPU".to_string(), None),
        };
        let device_codename = match device_id {
            DeviceId::Cpu(_) => Some(hw_info::cpu_codename().expect("Couldn't get CPU codename")),
            DeviceId::Gpu(_) => device.map(|d| d.name().expect("Couldn't get device codename")),
        };
        let mem_type_description: MemTypeDescription = (&mem_type).into();

        let template = DataPoint {
            hostname: hostname,
            device_type,
            device_codename,
            cpu_node,
            memory_node: mem_type_description.location,
            memory_type: Some(mem_type_description.bare_mem_type),
            page_type: Some(mem_type_description.page_type),
            range_bytes,
            ..Default::default()
        };

        let mut mem = Allocator::alloc_mem(mem_type, buffer_len);
        mem.ensure_physically_backed();

        let bandwidths = match device_id {
            DeviceId::Cpu(cpu_node) => {
                let mnt = CpuMeasurement::new(threads, cpu_affinity, template);
                let demem: DerefMem<_> = mem.try_into().expect("Cannot run benchmark on CPU with the given type of memory. Did you specify GPU device memory?");
                mnt.measure(
                    &demem,
                    CpuMemoryBandwidth::new(cpu_node, loop_length, target_cycles),
                    CpuMemoryBandwidth::run,
                    benchmarks,
                    operators,
                    item_bytes,
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
                    template,
                );

                let ml = GpuMemoryBandwidth::new(did, loop_length, target_cycles);
                let l = mnt.measure(
                    &mem,
                    ml,
                    GpuMemoryBandwidth::run,
                    benchmarks,
                    operators,
                    item_bytes,
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
