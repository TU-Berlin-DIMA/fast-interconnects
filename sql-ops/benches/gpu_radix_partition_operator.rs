/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019 Clemens Lutz, German Research Center for Artificial Intelligence
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use datagen::relation::{KeyAttribute, UniformRelation};
use num_rational::Ratio;
use num_traits::cast::FromPrimitive;
use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemType};
use numa_gpu::runtime::memory::Mem;
use numa_gpu::runtime::numa::NodeRatio;
use papi::event_set::Sample;

use rustacuda::context::{CacheConfig, CurrentContext, SharedMemoryConfig};
use rustacuda::device::DeviceAttribute;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::DeviceBuffer;
use rustacuda::memory::DeviceCopy;
use rustacuda::stream::{Stream, StreamFlags};
use serde_derive::Serialize;
use sql_ops::partition::gpu_radix_partition::{
    GpuRadixPartitionAlgorithm, GpuRadixPartitionable, GpuRadixPartitioner, PartitionedRelation,
};
use std::convert::TryInto;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::mem;
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::time::Instant;
use structopt::clap::arg_enum;
use structopt::StructOpt;

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgMemType {
        System,
        Numa,
        NumaLazyPinned,
        DistributedNuma,
        Pinned,
        Unified,
        Device,
    }
}

#[derive(Debug)]
pub struct ArgMemTypeHelper {
    pub mem_type: ArgMemType,
    pub node_ratios: Box<[NodeRatio]>,
}

impl From<ArgMemTypeHelper> for MemType {
    fn from(
        ArgMemTypeHelper {
            mem_type,
            node_ratios,
        }: ArgMemTypeHelper,
    ) -> Self {
        match mem_type {
            ArgMemType::System => MemType::SysMem,
            ArgMemType::Numa => MemType::NumaMem(node_ratios[0].node),
            ArgMemType::NumaLazyPinned => MemType::NumaMem(node_ratios[0].node),
            ArgMemType::DistributedNuma => MemType::DistributedNumaMem(node_ratios),
            ArgMemType::Pinned => MemType::CudaPinnedMem,
            ArgMemType::Unified => MemType::CudaUniMem,
            ArgMemType::Device => MemType::CudaDevMem,
        }
    }
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "GPU Radix Partition Benchmark",
    about = "A benchmark of the GPU radix partition operator using PAPI."
)]
struct Options {
    /// No effect (passed by Cargo to run only benchmarks instead of unit tests)
    #[structopt(long)]
    bench: bool,

    /// Number of tuples in the relation
    #[structopt(long, default_value = "10000000")]
    tuples: usize,

    #[structopt(long = "radix-bits", default_value = "8,10", require_delimiter = true)]
    /// Radix bits with which to partition
    radix_bits: Vec<u32>,

    /// Execute on CUDA device with ID
    #[structopt(long = "device-id", default_value = "0")]
    device_id: u16,

    /// Memory type with which to allocate input relation
    #[structopt(
        long = "input-mem-type",
        default_value = "Device",
        possible_values = &ArgMemType::variants(),
        case_insensitive = true
    )]
    input_mem_type: ArgMemType,

    /// Memory type with which to allocate output relation
    #[structopt(
        long = "output-mem-type",
        default_value = "Device",
        possible_values = &ArgMemType::variants(),
        case_insensitive = true
    )]
    output_mem_type: ArgMemType,

    /// Allocate memory for input relation on NUMA node (See numactl -H)
    #[structopt(long = "input-location", default_value = "0")]
    input_location: u16,

    /// Allocate memory for output relation on NUMA node (See numactl -H)
    #[structopt(long = "output-location", default_value = "0")]
    output_location: u16,

    /// Output path for the measurements CSV file
    #[structopt(long, default_value = "target/bench/gpu_radix_partition_operator.csv")]
    csv: PathBuf,

    /// PAPI configuration file
    #[structopt(long, requires("papi-preset"), default_value = "resources/papi.toml")]
    papi_config: PathBuf,

    /// Choose a PAPI preset from the PAPI configuration file
    #[structopt(long, default_value = "gpu_default")]
    papi_preset: String,

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
    pub input_mem_type: Option<ArgMemType>,
    pub output_mem_type: Option<ArgMemType>,
    pub input_location: Option<u16>,
    pub output_location: Option<u16>,
    pub tuple_bytes: Option<usize>,
    pub tuples: Option<usize>,
    pub radix_bits: Option<u32>,
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

fn gpu_radix_partition_benchmark<T, W>(
    bench_group: &str,
    bench_function: &str,
    algorithm: GpuRadixPartitionAlgorithm,
    radix_bits_list: &[u32],
    input_data: &(Mem<T>, Mem<T>),
    output_mem_type: &MemType,
    // papi: &Papi,
    _papi_preset: &str,
    repeat: u32,
    template: &DataPoint,
    csv_writer: &mut csv::Writer<W>,
) -> Result<(), Box<dyn Error>>
where
    T: Clone + Default + Send + DeviceCopy + FromPrimitive + GpuRadixPartitionable,
    W: Write,
{
    CurrentContext::set_cache_config(CacheConfig::PreferShared)?;
    CurrentContext::set_shared_memory_config(SharedMemoryConfig::FourByteBankSize)?;
    let device = CurrentContext::get_device()?;

    let multiprocessors = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
    let warp_size = device.get_attribute(DeviceAttribute::WarpSize)? as u32;
    let warp_overcommit_factor = 32;
    let grid_overcommit_factor = 1;
    let block_size = BlockSize::x(warp_size * warp_overcommit_factor);
    let grid_size = GridSize::x(multiprocessors * grid_overcommit_factor);

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let template = DataPoint {
        group: bench_group.to_string(),
        function: bench_function.to_string(),
        tuple_bytes: Some(2 * mem::size_of::<T>()),
        tuples: Some(input_data.0.len()),
        ..template.clone()
    };

    radix_bits_list
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
                input_data.0.len(),
                radix_bits,
                grid_size.x,
                Allocator::mem_alloc_fn(output_mem_type.clone()),
                Allocator::mem_alloc_fn(output_mem_type.clone()),
            );

            let result: Result<(), Box<dyn Error>> = (0..repeat).into_iter().try_for_each(|_| {
                // let ready_event_set = EventSetBuilder::new(&papi)?
                //     .use_preset(papi_preset)?
                //     .build()?;
                let sample = Sample::default();
                // ready_event_set.init_sample(&mut sample)?;

                // let running_event_set = ready_event_set.start()?;
                let timer = Instant::now();

                radix_prnr.partition(
                    input_data.0.as_launchable_slice(),
                    input_data.1.as_launchable_slice(),
                    &mut partitioned_relation,
                    &stream,
                )?;
                stream.synchronize()?;

                let time = timer.elapsed();
                // running_event_set.stop(&mut sample)?;
                let sample_vec = sample.into_iter().collect::<Vec<_>>();

                let dp = DataPoint {
                    radix_bits: Some(radix_bits),
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

fn alloc_and_gen<T>(tuples: usize, mem_type: MemType) -> Result<(Mem<T>, Mem<T>), Box<dyn Error>>
where
    T: Clone + Default + Send + DeviceCopy + FromPrimitive + KeyAttribute,
{
    const PAYLOAD_RANGE: RangeInclusive<usize> = 1..=10000;

    let host_alloc = match mem_type.clone().try_into() {
        Err(_) => Allocator::deref_mem_alloc_fn::<T>(DerefMemType::SysMem),
        Ok(mt) => Allocator::deref_mem_alloc_fn::<T>(mt),
    };
    let mut host_data_key = host_alloc(tuples);
    let mut host_data_pay = host_alloc(tuples);

    UniformRelation::gen_primary_key_par(host_data_key.as_mut_slice(), None).unwrap();
    UniformRelation::gen_attr_par(host_data_pay.as_mut_slice(), PAYLOAD_RANGE).unwrap();

    let dev_data = if let MemType::CudaDevMem = mem_type {
        (
            Mem::CudaDevMem(DeviceBuffer::from_slice(host_data_key.as_mut_slice())?),
            Mem::CudaDevMem(DeviceBuffer::from_slice(host_data_pay.as_mut_slice())?),
        )
    } else {
        (host_data_key.into(), host_data_pay.into())
    };

    Ok(dev_data)
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::from_args();
    let _papi_config = papi::Config::parse_file(&options.papi_config)?;
    // let papi = Papi::init_with_config(papi_config)?;

    let _context = rustacuda::quick_init()?;

    if let Some(parent) = options.csv.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    let csv_file = std::fs::File::create(&options.csv)?;
    let mut csv_writer = csv::Writer::from_writer(csv_file);

    let input_mem_type: MemType = ArgMemTypeHelper {
        mem_type: options.input_mem_type,
        node_ratios: Box::new([NodeRatio {
            node: options.input_location,
            ratio: Ratio::from_integer(0),
        }]),
    }
    .into();

    let output_mem_type: MemType = ArgMemTypeHelper {
        mem_type: options.output_mem_type,
        node_ratios: Box::new([NodeRatio {
            node: options.output_location,
            ratio: Ratio::from_integer(0),
        }]),
    }
    .into();

    let input_data = alloc_and_gen(options.tuples, input_mem_type)?;

    let template = DataPoint {
        hostname: hostname::get()?
            .into_string()
            .expect("Couldn't convert hostname into UTF-8 string"),
        device_codename: Some(CurrentContext::get_device()?.name()?),
        input_mem_type: Some(options.input_mem_type),
        output_mem_type: Some(options.output_mem_type),
        input_location: Some(options.input_location),
        output_location: Some(options.output_location),
        ..DataPoint::default()
    };

    // gpu_radix_partition_benchmark::<i64, _>(
    //     "gpu_radix_partition",
    //     "chunked",
    //     GpuRadixPartitionAlgorithm::Chunked,
    //     &options.radix_bits,
    //     &input_data,
    //     &output_mem_type,
    //     &papi,
    //     &options.papi_preset,
    //     options.repeat,
    //     &template,
    //     &mut csv_writer,
    // )?;

    // gpu_radix_partition_benchmark::<i64, _>(
    //     "gpu_radix_partition",
    //     "chunked_laswwc",
    //     GpuRadixPartitionAlgorithm::ChunkedLASWWC,
    //     &options.radix_bits,
    //     &input_data,
    //     &output_mem_type,
    //     // &papi,
    //     &options.papi_preset,
    //     options.repeat,
    //     &template,
    //     &mut csv_writer,
    // )?;

    gpu_radix_partition_benchmark::<i64, _>(
        "gpu_radix_partition",
        "chunked_sswwc",
        GpuRadixPartitionAlgorithm::ChunkedSSWWC,
        &options.radix_bits,
        &input_data,
        &output_mem_type,
        // &papi,
        &options.papi_preset,
        options.repeat,
        &template,
        &mut csv_writer,
    )?;

    Ok(())
}
