/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

extern crate assert_approx_eq;
extern crate error_chain;
extern crate numa_gpu;
extern crate rustacuda;

use assert_approx_eq::assert_approx_eq;

use error_chain::bail;

use numa_gpu::runtime::backend::{
    CudaTransferStrategy, IntoCudaIterator, IntoCudaIteratorWithStrategy,
};

use rustacuda::context::CurrentContext;
use rustacuda::device::DeviceAttribute;
use rustacuda::launch;
use rustacuda::memory::{CopyDestination, DeviceBox, UnifiedBuffer};
use rustacuda::module::Module;
use rustacuda::quick_init;

use std::error::Error;
use std::ffi::CString;

mod errors {
    use error_chain::*;

    error_chain! {}
}

#[test]
fn test_cuda_iterator_2_pageable_copy() -> Result<(), Box<dyn Error>> {
    let _ctx = quick_init()?;

    let module_data = CString::new(include_str!("../resources/dot.ptx"))?;
    let module = Module::load_from_string(&module_data)?;
    let cuda_dot = module.get_function(&CString::new("dot")?)?;

    let data_len = 2_usize.pow(20);
    let chunk_len = 1024_usize;
    assert_eq!(data_len % chunk_len, 0);

    let mut data_0 = vec![1.0_f32; data_len];
    let mut data_1 = vec![1.0_f32; data_len];
    let mut result = DeviceBox::new(&0.0f32)?;
    let result_ptr = result.as_device_ptr();

    (data_0.as_mut_slice(), data_1.as_mut_slice())
        .into_cuda_iter_with_strategy(CudaTransferStrategy::PageableCopy, chunk_len)?
        .fold(|(x, y), range, stream| {
            assert_ne!(range.len(), 0);
            assert_eq!(range.len(), chunk_len);

            unsafe {
                launch!(cuda_dot<<<10, 1024, 0, stream>>>(
                range.len(),
                x.as_launchable_ptr(),
                y.as_launchable_ptr(),
                result_ptr
                ))?;
            }

            Ok(())
        })?;

    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;

    assert_approx_eq!(data_len as f32, result_host);

    Ok(())
}

#[test]
fn test_cuda_iterator_2_pinned_copy() -> Result<(), Box<dyn Error>> {
    let _ctx = quick_init()?;

    let module_data = CString::new(include_str!("../resources/dot.ptx"))?;
    let module = Module::load_from_string(&module_data)?;
    let cuda_dot = module.get_function(&CString::new("dot")?)?;

    let data_len = 2_usize.pow(20);
    let chunk_len = 1024_usize;
    assert_eq!(data_len % chunk_len, 0);

    let mut data_0 = vec![1.0_f32; data_len];
    let mut data_1 = vec![1.0_f32; data_len];
    let mut result = DeviceBox::new(&0.0f32)?;
    let result_ptr = result.as_device_ptr();

    (data_0.as_mut_slice(), data_1.as_mut_slice())
        .into_cuda_iter_with_strategy(CudaTransferStrategy::PinnedCopy, chunk_len)?
        .fold(|(x, y), range, stream| {
            assert_ne!(range.len(), 0);
            assert_eq!(range.len(), chunk_len);

            unsafe {
                launch!(cuda_dot<<<10, 1024, 0, stream>>>(
                range.len(),
                x.as_launchable_ptr(),
                y.as_launchable_ptr(),
                result_ptr
                ))?;
            }

            Ok(())
        })?;

    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;

    assert_approx_eq!(data_len as f32, result_host);

    Ok(())
}

#[test]
fn test_cuda_iterator_2_lazy_pinned_copy() -> Result<(), Box<dyn Error>> {
    let _ctx = quick_init()?;

    let module_data = CString::new(include_str!("../resources/dot.ptx"))?;
    let module = Module::load_from_string(&module_data)?;
    let cuda_dot = module.get_function(&CString::new("dot")?)?;

    let data_len = 2_usize.pow(20);
    let chunk_len = 1024_usize;
    assert_eq!(data_len % chunk_len, 0);

    let mut data_0 = vec![1.0_f32; data_len];
    let mut data_1 = vec![1.0_f32; data_len];
    let mut result = DeviceBox::new(&0.0f32)?;
    let result_ptr = result.as_device_ptr();

    (data_0.as_mut_slice(), data_1.as_mut_slice())
        .into_cuda_iter_with_strategy(CudaTransferStrategy::LazyPinnedCopy, chunk_len)?
        .fold(|(x, y), range, stream| {
            assert_ne!(range.len(), 0);
            assert_eq!(range.len(), chunk_len);

            unsafe {
                launch!(cuda_dot<<<10, 1024, 0, stream>>>(
                range.len(),
                x.as_launchable_ptr(),
                y.as_launchable_ptr(),
                result_ptr
                ))?;
            }

            Ok(())
        })?;

    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;

    assert_approx_eq!(data_len as f32, result_host);

    Ok(())
}

#[test]
fn test_cuda_iterator_2_prefetch() -> Result<(), Box<dyn Error>> {
    let _ctx = quick_init()?;

    let is_concurrent_managed_access_supported =
        CurrentContext::get_device()?.get_attribute(DeviceAttribute::ConcurrentManagedAccess)?;
    if is_concurrent_managed_access_supported == 0 {
        return Ok(());
    }

    let module_data = CString::new(include_str!("../resources/dot.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    let data_len = 2_usize.pow(20);
    let chunk_len = 1024_usize;
    assert_eq!(data_len % chunk_len, 0);

    let mut data_0 = UnifiedBuffer::new(&1.0_f32, data_len)?;
    let mut data_1 = UnifiedBuffer::new(&1.0_f32, data_len)?;
    let mut result = DeviceBox::new(&0.0f32)?;

    (&mut data_0, &mut data_1)
        .into_cuda_iter(chunk_len)?
        .fold(|(x, y), range, stream| {
            assert_ne!(range.len(), 0);
            assert_eq!(range.len(), chunk_len);

            unsafe {
                launch!(module.dot<<<10, 1024, 0, stream>>>(
                range.len(),
                x.as_launchable_ptr(),
                y.as_launchable_ptr(),
                result.as_device_ptr()
                ))?;
            }

            Ok(())
        })?;

    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;

    assert_approx_eq!(data_len as f32, result_host);

    Ok(())
}

#[test]
// Cache-coherence is only supported on Power9, skip the test for other arches.
#[cfg(target_arch = "powerpc64")]
fn test_cuda_iterator_2_coherence() -> Result<(), Box<dyn Error>> {
    let _ctx = quick_init()?;

    let module_data = CString::new(include_str!("../resources/dot.ptx"))?;
    let module = Module::load_from_string(&module_data)?;
    let cuda_dot = module.get_function(&CString::new("dot")?)?;

    let data_len = 2_usize.pow(20);
    let chunk_len = 1024_usize;
    assert_eq!(data_len % chunk_len, 0);

    let mut data_0 = vec![1.0_f32; data_len];
    let mut data_1 = vec![1.0_f32; data_len];
    let mut result = DeviceBox::new(&0.0f32)?;
    let result_ptr = result.as_device_ptr();

    (data_0.as_mut_slice(), data_1.as_mut_slice())
        .into_cuda_iter_with_strategy(CudaTransferStrategy::Coherence, chunk_len)?
        .fold(|(x, y), range, stream| {
            assert_ne!(range.len(), 0);
            assert_eq!(range.len(), chunk_len);

            unsafe {
                launch!(cuda_dot<<<10, 1024, 0, stream>>>(
                range.len(),
                x.as_launchable_ptr(),
                y.as_launchable_ptr(),
                result_ptr
                ))?;
            }

            Ok(())
        })?;

    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;

    assert_approx_eq!(data_len as f32, result_host);

    Ok(())
}

#[test]
fn test_cuda_iterator_2_pageable_copy_non_divisor_chunk_len() -> Result<(), Box<dyn Error>> {
    let _ctx = quick_init()?;

    let module_data = CString::new(include_str!("../resources/dot.ptx"))?;
    let module = Module::load_from_string(&module_data)?;
    let cuda_dot = module.get_function(&CString::new("dot")?)?;

    let data_len = 2_usize.pow(20);
    let chunk_len = 1023_usize;

    let mut data_0 = vec![1.0_f32; data_len];
    let mut data_1 = vec![1.0_f32; data_len];
    let mut result = DeviceBox::new(&0.0f32)?;
    let result_ptr = result.as_device_ptr();

    (data_0.as_mut_slice(), data_1.as_mut_slice())
        .into_cuda_iter_with_strategy(CudaTransferStrategy::PageableCopy, chunk_len)?
        .fold(|(x, y), range, stream| {
            assert_ne!(range.len(), 0);

            unsafe {
                launch!(cuda_dot<<<10, 1024, 0, stream>>>(
                range.len(),
                x.as_launchable_ptr(),
                y.as_launchable_ptr(),
                result_ptr
                ))?;
            }

            Ok(())
        })?;

    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;

    assert_approx_eq!(data_len as f32, result_host);

    Ok(())
}

#[test]
fn test_cuda_iterator_2_pinned_copy_non_divisor_chunk_len() -> Result<(), Box<dyn Error>> {
    let _ctx = quick_init()?;

    let module_data = CString::new(include_str!("../resources/dot.ptx"))?;
    let module = Module::load_from_string(&module_data)?;
    let cuda_dot = module.get_function(&CString::new("dot")?)?;

    let data_len = 2_usize.pow(20);
    let chunk_len = 1023_usize;

    let mut data_0 = vec![1.0_f32; data_len];
    let mut data_1 = vec![1.0_f32; data_len];
    let mut result = DeviceBox::new(&0.0f32)?;
    let result_ptr = result.as_device_ptr();

    (data_0.as_mut_slice(), data_1.as_mut_slice())
        .into_cuda_iter_with_strategy(CudaTransferStrategy::PinnedCopy, chunk_len)?
        .fold(|(x, y), range, stream| {
            assert_ne!(range.len(), 0);

            unsafe {
                launch!(cuda_dot<<<10, 1024, 0, stream>>>(
                range.len(),
                x.as_launchable_ptr(),
                y.as_launchable_ptr(),
                result_ptr
                ))?;
            }

            Ok(())
        })?;

    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;

    assert_approx_eq!(data_len as f32, result_host);

    Ok(())
}

#[test]
fn test_cuda_iterator_2_lazy_pinned_copy_non_divisor_chunk_len() -> Result<(), Box<dyn Error>> {
    let _ctx = quick_init()?;

    let module_data = CString::new(include_str!("../resources/dot.ptx"))?;
    let module = Module::load_from_string(&module_data)?;
    let cuda_dot = module.get_function(&CString::new("dot")?)?;

    let data_len = 2_usize.pow(20);
    let chunk_len = 1023_usize;

    let mut data_0 = vec![1.0_f32; data_len];
    let mut data_1 = vec![1.0_f32; data_len];
    let mut result = DeviceBox::new(&0.0f32)?;
    let result_ptr = result.as_device_ptr();

    (data_0.as_mut_slice(), data_1.as_mut_slice())
        .into_cuda_iter_with_strategy(CudaTransferStrategy::LazyPinnedCopy, chunk_len)?
        .fold(|(x, y), range, stream| {
            assert_ne!(range.len(), 0);

            unsafe {
                launch!(cuda_dot<<<10, 1024, 0, stream>>>(
                range.len(),
                x.as_launchable_ptr(),
                y.as_launchable_ptr(),
                result_ptr
                ))?;
            }

            Ok(())
        })?;

    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;

    assert_approx_eq!(data_len as f32, result_host);

    Ok(())
}

#[test]
fn test_cuda_iterator_2_prefetch_non_divisor_chunk_len() -> Result<(), Box<dyn Error>> {
    let _ctx = quick_init()?;

    let is_concurrent_managed_access_supported =
        CurrentContext::get_device()?.get_attribute(DeviceAttribute::ConcurrentManagedAccess)?;
    if is_concurrent_managed_access_supported == 0 {
        return Ok(());
    }

    let module_data = CString::new(include_str!("../resources/dot.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    let data_len = 2_usize.pow(20);
    let chunk_len = 1023_usize;

    let mut data_0 = UnifiedBuffer::new(&1.0_f32, data_len)?;
    let mut data_1 = UnifiedBuffer::new(&1.0_f32, data_len)?;
    let mut result = DeviceBox::new(&0.0f32)?;

    (&mut data_0, &mut data_1)
        .into_cuda_iter(chunk_len)?
        .fold(|(x, y), range, stream| {
            assert_ne!(range.len(), 0);

            unsafe {
                launch!(module.dot<<<10, 1024, 0, stream>>>(
                range.len(),
                x.as_launchable_ptr(),
                y.as_launchable_ptr(),
                result.as_device_ptr()
                ))?;
            }

            Ok(())
        })?;

    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;

    assert_approx_eq!(data_len as f32, result_host);

    Ok(())
}

#[test]
// Cache-coherence is only supported on Power9, skip the test for other arches.
#[cfg(target_arch = "powerpc64")]
fn test_cuda_iterator_2_coherence_non_divisor_chunk_len() -> Result<(), Box<dyn Error>> {
    let _ctx = quick_init()?;

    let module_data = CString::new(include_str!("../resources/dot.ptx"))?;
    let module = Module::load_from_string(&module_data)?;
    let cuda_dot = module.get_function(&CString::new("dot")?)?;

    let data_len = 2_usize.pow(20);
    let chunk_len = 1023_usize;

    let mut data_0 = vec![1.0_f32; data_len];
    let mut data_1 = vec![1.0_f32; data_len];
    let mut result = DeviceBox::new(&0.0f32)?;
    let result_ptr = result.as_device_ptr();

    (data_0.as_mut_slice(), data_1.as_mut_slice())
        .into_cuda_iter_with_strategy(CudaTransferStrategy::Coherence, chunk_len)?
        .fold(|(x, y), range, stream| {
            assert_ne!(range.len(), 0);

            unsafe {
                launch!(cuda_dot<<<10, 1024, 0, stream>>>(
                range.len(),
                x.as_launchable_ptr(),
                y.as_launchable_ptr(),
                result_ptr
                ))?;
            }

            Ok(())
        })?;

    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;

    assert_approx_eq!(data_len as f32, result_host);

    Ok(())
}
