/*
 * Copyright 2020 Clemens Lutz, German Research Center for Artificial Intelligence
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use crate::error::Result;
use rustacuda::context::CurrentContext;
use rustacuda::device::DeviceAttribute;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::DeviceCopy;

#[repr(C)]
#[allow(unused)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuPrefixScanState<T>
where
    T: Clone + DeviceCopy + Default + Sized,
{
    status: T,
    aggregate: T,
    prefix: T,
    __padding: T,
}

unsafe impl<T: Clone + DeviceCopy + Default> DeviceCopy for GpuPrefixScanState<T> {}

pub struct GpuPrefixSum;

impl GpuPrefixSum {
    // Computes state length as number of 'GpuPrefixScanState<T>' elements
    pub fn state_len<G, B>(grid_size: G, block_size: B) -> Result<usize>
    where
        G: Into<GridSize>,
        B: Into<BlockSize>,
    {
        let warp_size =
            CurrentContext::get_device()?.get_attribute(DeviceAttribute::WarpSize)? as usize;
        let gs: GridSize = grid_size.into();
        let bs: BlockSize = block_size.into();

        Ok((gs.x as usize * bs.x as usize) / warp_size + warp_size)
    }
}
