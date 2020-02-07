/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2020, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use super::tables::LineItem;
use crate::error::{ErrorKind, Result};
use crate::types::ArgSelectionVariant;
use rustacuda::event::{Event, EventFlags};
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::launch;
use rustacuda::memory::{CopyDestination, DeviceBox};
use rustacuda::module::Module;
use rustacuda::stream::{Stream, StreamFlags};
use std::ffi::CString;
use std::time::Duration;

pub struct Query6Gpu {
    grid_size: GridSize,
    block_size: BlockSize,
    selection_variant: ArgSelectionVariant,
    module: Module,
}

impl Query6Gpu {
    pub fn new(
        grid_size: GridSize,
        block_size: BlockSize,
        selection_variant: ArgSelectionVariant,
    ) -> Result<Self> {
        let module_path = CString::new(env!("CUDAUTILS_PATH")).map_err(|_| {
            ErrorKind::NulCharError(
                "Failed to load CUDA module, check your CUDAUTILS_PATH".to_string(),
            )
        })?;
        let module = Module::load_from_file(&module_path)?;

        Ok(Self {
            grid_size,
            block_size,
            selection_variant,
            module,
        })
    }

    pub fn run(&self, lineitem: &LineItem) -> Result<(i64, Duration)> {
        let module = &self.module;
        let grid_size = &self.grid_size;
        let block_size = &self.block_size;

        let l_shipdate = lineitem.shipdate.as_launchable_ptr();
        let l_discount = lineitem.discount.as_launchable_ptr();
        let l_quantity = lineitem.quantity.as_launchable_ptr();
        let l_extendedprice = lineitem.extendedprice.as_launchable_ptr();

        let mut revenue = DeviceBox::<u64>::new(&0)?;
        let mut negative_revenue = DeviceBox::<u64>::new(&0)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let start_event = Event::new(EventFlags::DEFAULT)?;
        let stop_event = Event::new(EventFlags::DEFAULT)?;

        start_event.record(&stream)?;

        match self.selection_variant {
            ArgSelectionVariant::Branching => unsafe {
                launch!(
                module.tpch_q6_branching<<<grid_size, block_size, 0, stream>>>(
                    lineitem.len() as u64,
                    l_shipdate,
                    l_discount,
                    l_quantity,
                    l_extendedprice,
                    revenue.as_device_ptr(),
                    negative_revenue.as_device_ptr()
                    )
                )?
            },
            ArgSelectionVariant::Predication => unsafe {
                launch!(
                module.tpch_q6_predication<<<grid_size, block_size, 0, stream>>>(
                    lineitem.len() as u64,
                    l_shipdate,
                    l_discount,
                    l_quantity,
                    l_extendedprice,
                    revenue.as_device_ptr(),
                    negative_revenue.as_device_ptr()
                    )
                )?
            },
        }

        stop_event.record(&stream)?;
        stop_event.synchronize()?;
        let time = stop_event.elapsed_time_f32(&start_event)?;
        let duration = Duration::from_secs_f32(time / 1000.0);

        let mut host_revenue = 0;
        let mut host_negative_revenue = 0;
        revenue.copy_to(&mut host_revenue)?;
        negative_revenue.copy_to(&mut host_negative_revenue)?;

        let total_revenue = host_revenue as i64 - host_negative_revenue as i64;

        Ok((total_revenue, duration))
    }
}
