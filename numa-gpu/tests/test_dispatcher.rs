/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2021, Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use assert_approx_eq::assert_approx_eq;
use numa_gpu::runtime::dispatcher::{
    HetMorselExecutorBuilder, IntoHetMorselIterator, MorselSpec, WorkerCpuAffinity,
};
use rustacuda::memory::{CopyDestination, DeviceBox, LockedBuffer};
use rustacuda::module::Module;
use rustacuda::{launch, quick_init};
use std::error::Error;
use std::ffi::CString;
use std::mem::size_of;
use std::sync::Mutex;

#[test]
fn test_iter() -> Result<(), Box<dyn Error>> {
    let _ctx = quick_init()?;

    let module_data = CString::new(include_str!("../resources/dot.ptx"))?;
    let module = Module::load_from_string(&module_data)?;
    let cuda_dot = module.get_function(&CString::new("dot")?)?;

    let data_len = 2_usize.pow(20);
    let morsel_len = 1024;
    let morsel_spec = MorselSpec {
        cpu_morsel_bytes: morsel_len * 2 * size_of::<f32>(),
        gpu_morsel_bytes: morsel_len * 2 * size_of::<f32>(),
    };
    assert_eq!(data_len % morsel_len, 0);

    let mut data_0 = LockedBuffer::new(&1.0_f32, data_len)?;
    let mut data_1 = LockedBuffer::new(&1.0_f32, data_len)?;
    let mut result = DeviceBox::new(&0.0f32)?;
    let result_ptr = result.as_device_ptr();
    let cpu_result = Mutex::new(0.0_f32);

    let mut executor = HetMorselExecutorBuilder::new()
        .morsel_spec(morsel_spec)
        .worker_cpu_affinity(WorkerCpuAffinity::default())
        .gpu_ids(vec![0])
        .build()?;

    (data_0.as_mut_slice(), data_1.as_mut_slice())
        .into_het_morsel_iter(&mut executor)
        .fold(
            |(x, y)| {
                let sum: f32 = x.iter().zip(y.iter()).map(|(x, y)| x * y).sum();
                let mut total = cpu_result.lock().unwrap();
                *total += sum;
                Ok(())
            },
            |(x, y), stream| {
                assert_ne!(x.len(), 0);
                assert_eq!(x.len(), morsel_len);
                assert_eq!(x.len(), y.len());

                unsafe {
                    launch!(cuda_dot<<<10, 1024, 0, stream>>>(
                    x.len(),
                    x.as_launchable_ptr(),
                    y.as_launchable_ptr(),
                    result_ptr
                    ))?;
                }

                Ok(())
            },
        )?;

    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;
    result_host += cpu_result.into_inner()?;

    eprintln!("Expecting {}, got {}", data_len, result_host);
    assert_approx_eq!(data_len as f32, result_host);

    Ok(())
}

#[test]
fn test_iter_non_divisor_morsel_len() -> Result<(), Box<dyn Error>> {
    let _ctx = quick_init()?;

    let module_data = CString::new(include_str!("../resources/dot.ptx"))?;
    let module = Module::load_from_string(&module_data)?;
    let cuda_dot = module.get_function(&CString::new("dot")?)?;

    let data_len = 2_usize.pow(20);
    let morsel_len = 1023;
    let morsel_spec = MorselSpec {
        cpu_morsel_bytes: morsel_len * 2 * size_of::<f32>(),
        gpu_morsel_bytes: morsel_len * 2 * size_of::<f32>(),
    };

    let mut data_0 = LockedBuffer::new(&1.0_f32, data_len)?;
    let mut data_1 = LockedBuffer::new(&1.0_f32, data_len)?;
    let mut result = DeviceBox::new(&0.0f32)?;
    let result_ptr = result.as_device_ptr();
    let cpu_result = Mutex::new(0.0_f32);

    let mut executor = HetMorselExecutorBuilder::new()
        .morsel_spec(morsel_spec)
        .worker_cpu_affinity(WorkerCpuAffinity::default())
        .gpu_ids(vec![0])
        .build()?;

    (data_0.as_mut_slice(), data_1.as_mut_slice())
        .into_het_morsel_iter(&mut executor)
        .fold(
            |(x, y)| {
                let sum: f32 = x.iter().zip(y.iter()).map(|(x, y)| x * y).sum();
                let mut total = cpu_result.lock().unwrap();
                *total += sum;
                Ok(())
            },
            |(x, y), stream| {
                assert_ne!(x.len(), 0);
                assert_eq!(x.len(), y.len());

                unsafe {
                    launch!(cuda_dot<<<10, 1024, 0, stream>>>(
                    x.len(),
                    x.as_launchable_ptr(),
                    y.as_launchable_ptr(),
                    result_ptr
                    ))?;
                }

                Ok(())
            },
        )?;

    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;
    result_host += cpu_result.into_inner()?;

    eprintln!("Expecting {}, got {}", data_len, result_host);
    assert_approx_eq!(data_len as f32, result_host);

    Ok(())
}
