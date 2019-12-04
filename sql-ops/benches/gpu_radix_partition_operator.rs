/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019 Clemens Lutz, German Research Center for Artificial Intelligence
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use datagen::relation::UniformRelation;
use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemType};
use numa_gpu::runtime::memory::Mem;
use papi::event_set::{EventSetBuilder, Sample};
use papi::Papi;
use rustacuda::context::CurrentContext;
use rustacuda::device::DeviceAttribute;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::DeviceBuffer;
use rustacuda::stream::{Stream, StreamFlags};
use serde_derive::Serialize;
use sql_ops::partition::gpu_radix_partition::{
    GpuRadixPartitionAlgorithm, GpuRadixPartitioner, PartitionedRelation,
};
use std::error::Error;
use std::fs;
use std::io::Write;
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::time::Instant;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "GPU Radix Partition Benchmark",
    about = "A benchmark of the GPU radix partition operator using PAPI."
)]
struct Options {
    /// No effect (passed by Cargo to run only benchmarks instead of unit tests)
    #[structopt(long)]
    bench: bool,

    /// Output path for the measurements CSV file
    #[structopt(long, default_value = "target/bench/gpu_radix_partition_operator.csv")]
    csv: PathBuf,

    /// PAPI configuration file
    #[structopt(long, requires("papi-preset"), default_value = "resources/papi.toml")]
    papi_config: PathBuf,

    /// Choose a PAPI preset from the PAPI configuration file
    #[structopt(long, default_value = "gpu_default")]
    papi_preset: String,

    /// Number of tuples in the relation
    #[structopt(long, default_value = "10000000")]
    tuples: usize,

    /// Number of samples to gather
    #[structopt(long, default_value = "30")]
    repeat: u32,
}

#[derive(Clone, Debug, Default, Serialize)]
struct DataPoint {
    pub group: String,
    pub function: String,
    pub hostname: String,
    pub device_codename: Option<String>,
    pub threads: Option<usize>,
    pub grid_size: Option<u32>,
    pub block_size: Option<u32>,
    pub tuples: Option<usize>,
    pub ns: Option<u128>,
    pub papi_name_0: Option<String>,
    pub papi_value_0: Option<i64>,
    pub papi_name_1: Option<String>,
    pub papi_value_1: Option<i64>,
    pub papi_name_2: Option<String>,
    pub papi_value_2: Option<i64>,
    pub papi_name_3: Option<String>,
    pub papi_value_3: Option<i64>,
    pub papi_name_4: Option<String>,
    pub papi_value_4: Option<i64>,
}

fn gpu_radix_partition_benchmark<W: Write>(
    bench_group: &str,
    bench_function: &str,
    algorithm: GpuRadixPartitionAlgorithm,
    tuples: usize,
    papi: &Papi,
    papi_preset: &str,
    repeat: u32,
    csv_writer: &mut csv::Writer<W>,
) -> Result<(), Box<dyn Error>> {
    const PAYLOAD_RANGE: RangeInclusive<usize> = 1..=10000;
    const NUMA_NODE: u16 = 0;
    const RADIX_BITS: [u32; 2] = [8, 10];

    let _context = rustacuda::quick_init()?;
    let device = CurrentContext::get_device()?;

    let multiprocessors = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
    let warp_size = device.get_attribute(DeviceAttribute::WarpSize)? as u32;
    let warp_overcommit_factor = 4;
    let grid_overcommit_factor = 2;
    let block_size = BlockSize::x(warp_size * warp_overcommit_factor);
    let grid_size = GridSize::x(multiprocessors * grid_overcommit_factor);

    let mut host_data_key =
        Allocator::alloc_deref_mem::<i32>(DerefMemType::NumaMem(NUMA_NODE), tuples);
    let mut host_data_pay =
        Allocator::alloc_deref_mem::<i32>(DerefMemType::NumaMem(NUMA_NODE), tuples);

    UniformRelation::gen_primary_key(host_data_key.as_mut_slice()).unwrap();
    UniformRelation::gen_attr(host_data_pay.as_mut_slice(), PAYLOAD_RANGE).unwrap();

    let dev_data_key = Mem::CudaDevMem(DeviceBuffer::from_slice(host_data_key.as_mut_slice())?);
    let dev_data_pay = Mem::CudaDevMem(DeviceBuffer::from_slice(host_data_pay.as_mut_slice())?);

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let template = DataPoint {
        group: bench_group.to_string(),
        function: bench_function.to_string(),
        hostname: hostname::get()?
            .into_string()
            .expect("Couldn't convert hostname into UTF-8 string"),
        device_codename: Some(device.name()?),
        tuples: Some(tuples),
        ..DataPoint::default()
    };

    RADIX_BITS
        .iter()
        .map(|&radix_bits| {
            let mut radix_prnr = GpuRadixPartitioner::new(
                algorithm,
                radix_bits,
                Allocator::mem_alloc_fn(MemType::CudaDevMem),
                GridSize::from(grid_size.clone()),
                BlockSize::from(block_size.clone()),
            )?;

            let mut partitioned_relation = PartitionedRelation::new(
                tuples,
                radix_bits,
                Allocator::mem_alloc_fn(MemType::CudaDevMem),
                Allocator::mem_alloc_fn(MemType::CudaDevMem),
            );

            let result: Result<(), Box<dyn Error>> = (0..repeat).into_iter().try_for_each(|_| {
                let ready_event_set = EventSetBuilder::new(&papi)?
                    .use_preset(papi_preset)?
                    .build()?;
                let mut sample = Sample::default();
                ready_event_set.init_sample(&mut sample)?;

                let timer = Instant::now();
                let running_event_set = ready_event_set.start()?;

                radix_prnr.partition(
                    dev_data_key.as_launchable_slice(),
                    dev_data_pay.as_launchable_slice(),
                    &mut partitioned_relation,
                    &stream,
                )?;
                stream.synchronize()?;

                running_event_set.stop(&mut sample)?;
                let time = timer.elapsed();
                let sample_vec = sample.into_iter().collect::<Vec<_>>();

                let dp = DataPoint {
                    grid_size: Some(grid_size.x),
                    block_size: Some(block_size.x),
                    ns: Some(time.as_nanos()),
                    papi_name_0: sample_vec.get(0).map(|x| x.0.clone()),
                    papi_value_0: sample_vec.get(0).map(|x| x.1.clone()),
                    papi_name_1: sample_vec.get(1).map(|x| x.0.clone()),
                    papi_value_1: sample_vec.get(1).map(|x| x.1.clone()),
                    papi_name_2: sample_vec.get(2).map(|x| x.0.clone()),
                    papi_value_2: sample_vec.get(2).map(|x| x.1.clone()),
                    papi_name_3: sample_vec.get(3).map(|x| x.0.clone()),
                    papi_value_3: sample_vec.get(3).map(|x| x.1.clone()),
                    papi_name_4: sample_vec.get(4).map(|x| x.0.clone()),
                    papi_value_4: sample_vec.get(4).map(|x| x.1.clone()),
                    ..template.clone()
                };

                csv_writer.serialize(dp)?;
                Ok(())
            });
            result?;

            Ok(())
        })
        .collect::<Result<(), Box<dyn Error>>>()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::from_args();
    let papi_config = papi::Config::parse_file(&options.papi_config)?;
    let papi = Papi::init_with_config(papi_config)?;

    if let Some(parent) = options.csv.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    let csv_file = std::fs::File::create(&options.csv)?;
    let mut csv_writer = csv::Writer::from_writer(csv_file);

    gpu_radix_partition_benchmark(
        "gpu_radix_partition",
        "block",
        GpuRadixPartitionAlgorithm::Block,
        options.tuples,
        &papi,
        &options.papi_preset,
        options.repeat,
        &mut csv_writer,
    )?;

    Ok(())
}
