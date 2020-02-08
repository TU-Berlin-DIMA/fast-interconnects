/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2020, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

mod data_point;
pub mod error;
mod harness;
mod query_6;
mod types;

use crate::data_point::DataPoint;
use crate::error::Result;
use crate::query_6::cpu::Query6Cpu;
use crate::query_6::gpu::Query6Gpu;
use crate::query_6::tables::{LineItem, LineItemTuple};
use crate::types::*;
use num_rational::Ratio;
use numa_gpu::runtime::allocator::DerefMemType;
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::hw_info::cpu_codename;
use numa_gpu::runtime::numa::NodeRatio;
use rustacuda::device::DeviceAttribute;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::prelude::*;
use std::mem;
use std::path::PathBuf;
use std::time::Duration;
use structopt::StructOpt;

fn main() -> Result<()> {
    // Parse commandline arguments
    let cmd = CmdOpt::from_args();

    // Initialize CUDA
    let _context = if cmd.execution_method != ArgExecutionMethod::Cpu {
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(cmd.device_id.into())?;
        Some(Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device,
        )?)
    } else {
        None
    };

    let node_ratios = Box::new([NodeRatio {
        node: cmd.rel_location,
        ratio: Ratio::from_integer(1),
    }]);
    let mem_type: DerefMemType = ArgMemTypeHelper {
        mem_type: cmd.rel_mem_type,
        node_ratios: node_ratios.clone(),
    }
    .into();

    let cpu_affinity = if let Some(ref cpu_affinity_file) = cmd.cpu_affinity {
        CpuAffinity::from_file(cpu_affinity_file.as_path())?
    } else {
        CpuAffinity::default()
    };

    let csv_file = cmd
        .csv
        .as_ref()
        .map(|file_name| std::fs::File::create(file_name))
        .transpose()?
        .map(|writer| Box::new(writer));
    let mut template = cmd.fill_data_point(&DataPoint::new()?)?;

    match cmd.query {
        6 => {
            let lineitem = LineItem::new(cmd.scale_factor, mem_type)?;
            template.tuples = Some(lineitem.len());
            template.bytes = Some(mem::size_of::<LineItemTuple>() * lineitem.len());
            let query: Box<dyn FnMut() -> Result<(i64, Duration)>> = match cmd.execution_method {
                ArgExecutionMethod::Cpu => {
                    let q = Query6Cpu::new(cmd.threads, &cpu_affinity, cmd.selection_variant);
                    Box::new(move || q.run(&lineitem))
                }
                ArgExecutionMethod::Gpu => {
                    // Device tuning
                    let device = Device::get_device(cmd.device_id.into())?;
                    let multiprocessors =
                        device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
                    let warp_size = device.get_attribute(DeviceAttribute::WarpSize)? as u32;
                    let warp_overcommit_factor = 4;
                    let grid_overcommit_factor = 2;

                    let block_size = BlockSize::x(warp_size * warp_overcommit_factor);
                    let grid_size = GridSize::x(multiprocessors * grid_overcommit_factor);

                    let q = Query6Gpu::new(grid_size, block_size, cmd.selection_variant)?;
                    Box::new(move || q.run(&lineitem))
                }
                em @ _ => unimplemented!("Execution method {:?} is not yet implemented!", em),
            };

            harness::measure(cmd.repeat, csv_file, template, Box::new(query))?;
        }
        q @ _ => panic!(format!("TPC-H query {} is not supported!", q)),
    };

    Ok(())
}

#[derive(StructOpt)]
#[structopt(rename_all = "kebab-case")]
struct CmdOpt {
    /// TPC-H query to run
    query: u32,

    /// TPC-H scale factor
    #[structopt(long, default_value = "1")]
    scale_factor: u32,

    /// Selection variant
    #[structopt(
        long,
        default_value = "Branching",
        raw(
            possible_values = "&ArgSelectionVariant::variants()",
            case_insensitive = "true"
        )
    )]
    selection_variant: ArgSelectionVariant,

    /// Number of times to repeat the benchmark
    #[structopt(long, default_value = "30")]
    repeat: u32,

    /// Output filename for measurement CSV file
    #[structopt(long, parse(from_os_str))]
    csv: Option<PathBuf>,

    /// Memory type with which to allocate data
    #[structopt(
        long,
        default_value = "Unified",
        raw(possible_values = "&ArgMemType::variants()", case_insensitive = "true")
    )]
    rel_mem_type: ArgMemType,

    #[structopt(long, default_value = "0")]
    /// Allocate memory for inner relation on CPU or GPU (See numactl -H and CUDA device list)
    rel_location: u16,

    /// Execute on device(s) with in-place or streaming-transfer method
    #[structopt(
        long,
        default_value = "CPU",
        raw(
            possible_values = "&ArgExecutionMethod::variants()",
            case_insensitive = "true"
        )
    )]
    execution_method: ArgExecutionMethod,

    #[structopt(long, default_value = "0")]
    /// Execute on GPU (See CUDA device list)
    device_id: u16,

    #[structopt(long, default_value = "1")]
    threads: usize,

    /// Path to CPU affinity map file for CPU workers
    #[structopt(long, parse(from_os_str))]
    cpu_affinity: Option<PathBuf>,
}

impl CmdOpt {
    fn fill_data_point(&self, data_point: &DataPoint) -> Result<DataPoint> {
        // Get device information
        let dev_codename_str = match self.execution_method {
            ArgExecutionMethod::Cpu => vec![cpu_codename()?],
            ArgExecutionMethod::Gpu | ArgExecutionMethod::GpuStream => {
                let device = Device::get_device(self.device_id.into())?;
                vec![device.name()?]
            }
            ArgExecutionMethod::Het | ArgExecutionMethod::GpuBuildHetProbe => {
                let device = Device::get_device(self.device_id.into())?;
                vec![cpu_codename()?, device.name()?]
            }
        };

        let dp = DataPoint {
            tpch_query: Some(self.query),
            scale_factor: Some(self.scale_factor),
            selection_variant: Some(self.selection_variant),
            execution_method: Some(self.execution_method),
            device_codename: Some(dev_codename_str),
            threads: if self.execution_method != ArgExecutionMethod::Gpu
                && self.execution_method != ArgExecutionMethod::GpuStream
            {
                Some(self.threads)
            } else {
                None
            },
            relation_memory_type: Some(self.rel_mem_type),
            relation_memory_location: Some(self.rel_location),
            ..data_point.clone()
        };

        Ok(dp)
    }
}
