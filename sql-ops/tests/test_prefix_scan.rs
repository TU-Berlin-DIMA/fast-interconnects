/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019-2020 Clemens Lutz, German Research Center for Artificial
 * Intelligence
 * Author: Clemens Lutz, DFKI GmbH <clemens.lutz@dfki.de>
 */

use rand::{thread_rng, Rng};
use rustacuda::context::CurrentContext;
use rustacuda::device::DeviceAttribute;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::launch;
use rustacuda::memory::CopyDestination;
use rustacuda::prelude::*;
use sql_ops::prefix_scan::{GpuPrefixScanState, GpuPrefixSum};
use std::error::Error;
use std::ffi::CString;
use std::mem::size_of;

fn block_prefix_sum<G, B>(
    data_len: usize,
    grid_size: G,
    block_size: B,
) -> Result<(), Box<dyn Error>>
where
    G: Into<GridSize>,
    B: Into<BlockSize>,
{
    let _context = rustacuda::quick_init()?;
    let module_path = CString::new(env!("CUDAUTILS_PATH"))?;
    let module = Module::load_from_file(&module_path)?;
    let _warp_size = CurrentContext::get_device()?.get_attribute(DeviceAttribute::WarpSize)? as u32;
    let log2_num_banks = env!("LOG2_NUM_BANKS").parse::<u32>()?;

    let data: Vec<u64> = (0..data_len)
        .into_iter()
        .scan(thread_rng(), |rng, _| Some(rng.gen()))
        .collect();
    let mut dev_data = DeviceBuffer::from_slice(&data)?;

    let bs: BlockSize = block_size.into();
    // let shared_mem_size = bs.x / warp_size;
    let mut shared_mem_size = bs.x * size_of::<u64>() as u32;
    shared_mem_size += shared_mem_size >> log2_num_banks;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let data_len_u32 = data_len as u32;

    unsafe {
        launch!(module.host_block_exclusive_prefix_sum_uint64<<<grid_size, bs, shared_mem_size, stream>>>(
            dev_data.as_device_ptr(),
            data_len_u32,
            0_u32
        ))?;
    }

    stream.synchronize()?;

    let mut result = vec![0; data_len];
    dev_data.copy_to(&mut result)?;

    let prefix_sum: Vec<_> = data
        .iter()
        .scan(0, |sum, &item| {
            let old_sum = *sum;
            *sum += item;
            Some(old_sum)
        })
        .collect();

    prefix_sum
        .iter()
        .cloned()
        .zip(result.iter().cloned())
        .for_each(|(check, res)| {
            assert_eq!(check, res);
        });

    Ok(())
}

fn device_prefix_sum<G, B>(
    data_len: usize,
    grid_size: G,
    block_size: B,
) -> Result<(), Box<dyn Error>>
where
    G: Into<GridSize>,
    B: Into<BlockSize>,
{
    let _context = rustacuda::quick_init()?;
    let module_path = CString::new(env!("CUDAUTILS_PATH"))?;
    let module = Module::load_from_file(&module_path)?;
    let gs: GridSize = grid_size.into();
    let bs: BlockSize = block_size.into();

    let data: Vec<u64> = (0..data_len)
        .into_iter()
        .scan(thread_rng(), |rng, _| Some(rng.gen()))
        .collect();
    let mut dev_data = DeviceBuffer::from_slice(&data)?;

    let state_len = GpuPrefixSum::state_len(gs.clone(), bs.clone())?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let data_len_u32 = data_len as u32;
    let mut dev_state: DeviceBuffer<GpuPrefixScanState<u64>> =
        unsafe { DeviceBuffer::uninitialized(state_len)? };

    unsafe {
        launch!(module.host_device_exclusive_prefix_sum_initialize_uint64<<<gs.clone(), bs.clone(), 0, stream>>>(
            dev_state.as_device_ptr()
        ))?;

        // Race-condition occurs without sync and also with stream.synchronize().
        // Synchronizing on the context seems to solve the problem.
        CurrentContext::synchronize()?;

        launch!(module.host_device_exclusive_prefix_sum_uint64<<<gs.clone(), bs.clone(), 0, stream>>>(
            dev_data.as_device_ptr(),
            data_len_u32,
            0_u32,
            dev_state.as_device_ptr()
        ))?;
    }

    stream.synchronize()?;

    let mut result = vec![0; data_len];
    dev_data.copy_to(&mut result)?;

    let prefix_sum: Vec<_> = data
        .iter()
        .scan(0, |sum, &item| {
            let old_sum = *sum;
            *sum += item;
            Some(old_sum)
        })
        .collect();

    prefix_sum
        .iter()
        .cloned()
        .zip(result.iter().cloned())
        .for_each(|(check, res)| {
            assert_eq!(check, res);
        });

    Ok(())
}

#[test]
fn block_prefix_sum_block_size() -> Result<(), Box<dyn Error>> {
    block_prefix_sum(1024_usize, 1_u32, 1024_u32)
}

#[test]
fn block_prefix_sum_block_size_divisor() -> Result<(), Box<dyn Error>> {
    block_prefix_sum(512_usize, 1_u32, 1024_u32)
}

#[test]
fn block_prefix_sum_block_size_multiple() -> Result<(), Box<dyn Error>> {
    block_prefix_sum(2048_usize, 1_u32, 1024_u32)
}

#[test]
fn device_prefix_sum_single_warp() -> Result<(), Box<dyn Error>> {
    device_prefix_sum(1024_usize, 1_u32, 32_u32)
}

#[test]
fn device_prefix_sum_two_warps() -> Result<(), Box<dyn Error>> {
    device_prefix_sum(1024_usize, 1_u32, 64_u32)
}

#[test]
fn device_prefix_sum_two_blocks() -> Result<(), Box<dyn Error>> {
    device_prefix_sum(2_usize * 64, 2_u32, 64_u32)
}

#[test]
fn device_prefix_sum_three_blocks() -> Result<(), Box<dyn Error>> {
    device_prefix_sum(3_usize * 1024, 3_u32, 1024_u32)
}

#[test]
fn device_prefix_sum_hundred_blocks() -> Result<(), Box<dyn Error>> {
    device_prefix_sum(100_usize * 1024, 100_u32, 1024_u32)
}

#[test]
fn device_prefix_sum_multiple_items_per_thread() -> Result<(), Box<dyn Error>> {
    device_prefix_sum(100_usize * 1024, 2_u32, 1024_u32)
}
