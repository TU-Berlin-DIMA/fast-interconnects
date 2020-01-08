/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019 Clemens Lutz, German Research Center for Artificial
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
