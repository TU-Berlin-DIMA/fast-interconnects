/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2020 Clemens Lutz, German Research Center for Artificial Intelligence
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use datagen::relation::{KeyAttribute, UniformRelation};
use itertools::iproduct;
use num_rational::Ratio;
use num_traits::cast::FromPrimitive;
use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemType};
use numa_gpu::runtime::memory::Mem;
use numa_gpu::runtime::numa::NodeRatio;
use rustacuda::context::{CacheConfig, CurrentContext, SharedMemoryConfig};
use rustacuda::device::DeviceAttribute;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::DeviceBuffer;
use rustacuda::memory::DeviceCopy;
use rustacuda::stream::{Stream, StreamFlags};
use serde_derive::Serialize;
use serde_repr::Serialize_repr;
use sql_ops::partition::gpu_radix_partition::{
    GpuHistogramAlgorithm, GpuRadixPartitionAlgorithm, GpuRadixPartitionable, GpuRadixPartitioner,
    PartitionedRelation,
};
use std::convert::TryInto;
use std::error::Error;
use std::fs;
use std::io::Write;
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

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgHistogramAlgorithm {
        GpuChunked,
        GpuContiguous,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize_repr)]
    #[repr(usize)]
    pub enum ArgTupleBytes {
        Bytes8 = 8,
        Bytes16 = 16,
    }
}

impl Into<GpuHistogramAlgorithm> for ArgHistogramAlgorithm {
    fn into(self) -> GpuHistogramAlgorithm {
        match self {
            Self::GpuChunked => GpuHistogramAlgorithm::GpuChunked,
            Self::GpuContiguous => GpuHistogramAlgorithm::GpuContiguous,
        }
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgRadixPartitionAlgorithm {
        NC,
        LASWWC,
        SSWWC,
        SSWWCNT,
        SSWWCv2,
        HSSWWC,
        HSSWWCv2,
        HSSWWCv3,
        HSSWWCv4,
    }
}

impl Into<GpuRadixPartitionAlgorithm> for ArgRadixPartitionAlgorithm {
    fn into(self) -> GpuRadixPartitionAlgorithm {
        match self {
            Self::NC => GpuRadixPartitionAlgorithm::NC,
            Self::LASWWC => GpuRadixPartitionAlgorithm::LASWWC,
            Self::SSWWC => GpuRadixPartitionAlgorithm::SSWWC,
            Self::SSWWCNT => GpuRadixPartitionAlgorithm::SSWWCNT,
            Self::SSWWCv2 => GpuRadixPartitionAlgorithm::SSWWCv2,
            Self::HSSWWC => GpuRadixPartitionAlgorithm::HSSWWC,
            Self::HSSWWCv2 => GpuRadixPartitionAlgorithm::HSSWWCv2,
            Self::HSSWWCv3 => GpuRadixPartitionAlgorithm::HSSWWCv3,
            Self::HSSWWCv4 => GpuRadixPartitionAlgorithm::HSSWWCv4,
        }
    }
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "GPU Radix Partition Benchmark",
    about = "A benchmark of the GPU radix partition operator."
)]
struct Options {
    /// Select the histogram algorithms to run
    #[structopt(
        long,
        default_value = "GpuChunked",
        possible_values = &ArgHistogramAlgorithm::variants(),
        case_insensitive = true,
        require_delimiter = true
    )]
    histogram_algorithms: Vec<ArgHistogramAlgorithm>,

    /// Select the radix partition algorithms to run
    #[structopt(
        long,
        default_value = "NC",
        possible_values = &ArgRadixPartitionAlgorithm::variants(),
        case_insensitive = true,
        require_delimiter = true
    )]
    partition_algorithms: Vec<ArgRadixPartitionAlgorithm>,

    /// No effect (passed by Cargo to run only benchmarks instead of unit tests)
    #[structopt(long)]
    bench: bool,

    /// Number of tuples in the relation
    #[structopt(long, default_value = "10000000")]
    tuples: usize,

    /// Tuple size (bytes)
    #[structopt(
        long = "tuple-bytes",
        default_value = "Bytes8",
        possible_values = &ArgTupleBytes::variants(),
        case_insensitive = true
    )]
    tuple_bytes: ArgTupleBytes,

    #[structopt(long = "radix-bits", default_value = "8,10", require_delimiter = true)]
    /// Radix bits with which to partition
    radix_bits: Vec<u32>,

    /// Execute on CUDA device with ID
    #[structopt(long = "device-id", default_value = "0")]
    device_id: u16,

    /// Execute with CUDA grid size (Default: #SMs)
    #[structopt(long = "grid-size")]
    grid_size: Option<u32>,

    /// Device memory buffer sizes for HSSWWC variants (in KiB)
    #[structopt(long, default_value = "2048", require_delimiter = true)]
    dmem_buffer_sizes: Vec<usize>,

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

    /// Number of samples to gather
    #[structopt(long, default_value = "1")]
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
    pub dmem_buffer_bytes: Option<usize>,
    pub input_mem_type: Option<ArgMemType>,
    pub output_mem_type: Option<ArgMemType>,
    pub input_location: Option<u16>,
    pub output_location: Option<u16>,
    pub tuple_bytes: Option<ArgTupleBytes>,
    pub tuples: Option<usize>,
    pub radix_bits: Option<u32>,
    pub ns: Option<u128>,
}

fn gpu_radix_partition_benchmark<T, W>(
    bench_group: &str,
    bench_function: &str,
    histogram_algorithm: GpuHistogramAlgorithm,
    partition_algorithm: GpuRadixPartitionAlgorithm,
    radix_bits_list: &[u32],
    input_data: &(Mem<T>, Mem<T>),
    output_mem_type: &MemType,
    grid_size_hint: &Option<GridSize>,
    dmem_buffer_bytes: usize,
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
    let grid_size = grid_size_hint
        .as_ref()
        .cloned()
        .unwrap_or_else(|| GridSize::x(multiprocessors * grid_overcommit_factor));

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let template = DataPoint {
        group: bench_group.to_string(),
        function: bench_function.to_string(),
        dmem_buffer_bytes: Some(dmem_buffer_bytes),
        ..template.clone()
    };

    radix_bits_list
        .iter()
        .map(|&radix_bits| {
            let mut radix_prnr = GpuRadixPartitioner::new(
                histogram_algorithm,
                partition_algorithm,
                radix_bits,
                Allocator::mem_alloc_fn(MemType::CudaDevMem),
                &grid_size,
                &block_size,
                dmem_buffer_bytes,
            )?;

            let mut partitioned_relation = PartitionedRelation::new(
                input_data.0.len(),
                histogram_algorithm,
                radix_bits,
                &grid_size,
                Allocator::mem_alloc_fn(output_mem_type.clone()),
                Allocator::mem_alloc_fn(output_mem_type.clone()),
            );

            let result: Result<(), Box<dyn Error>> = (0..repeat).into_iter().try_for_each(|_| {
                let timer = Instant::now();

                radix_prnr.partition(
                    input_data.0.as_launchable_slice(),
                    input_data.1.as_launchable_slice(),
                    &mut partitioned_relation,
                    &stream,
                )?;
                stream.synchronize()?;

                let time = timer.elapsed();

                let dp = DataPoint {
                    radix_bits: Some(radix_bits),
                    grid_size: Some(grid_size.x),
                    block_size: Some(block_size.x),
                    ns: Some(time.as_nanos()),
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

fn alloc_and_gen<T>(tuples: usize, mem_type: &MemType) -> Result<(Mem<T>, Mem<T>), Box<dyn Error>>
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

    let grid_size_hint = options.grid_size.map(GridSize::from);

    let template = DataPoint {
        hostname: hostname::get()?
            .into_string()
            .expect("Couldn't convert hostname into UTF-8 string"),
        device_codename: Some(CurrentContext::get_device()?.name()?),
        input_mem_type: Some(options.input_mem_type),
        output_mem_type: Some(options.output_mem_type),
        input_location: Some(options.input_location),
        output_location: Some(options.output_location),
        tuple_bytes: Some(options.tuple_bytes),
        tuples: Some(options.tuples),
        ..DataPoint::default()
    };

    match options.tuple_bytes {
        ArgTupleBytes::Bytes8 => {
            let input_data = alloc_and_gen(options.tuples, &input_mem_type)?;
            for (histogram_algorithm, partition_algorithm, dmem_buffer_size) in iproduct!(
                options.histogram_algorithms,
                options.partition_algorithms,
                options.dmem_buffer_sizes
            ) {
                gpu_radix_partition_benchmark::<i32, _>(
                    "gpu_radix_partition",
                    &(histogram_algorithm.to_string() + &partition_algorithm.to_string()),
                    histogram_algorithm.into(),
                    partition_algorithm.into(),
                    &options.radix_bits,
                    &input_data,
                    &output_mem_type,
                    &grid_size_hint,
                    dmem_buffer_size * 1024,
                    options.repeat,
                    &template,
                    &mut csv_writer,
                )?;
            }
        }
        ArgTupleBytes::Bytes16 => {
            let input_data = alloc_and_gen(options.tuples, &input_mem_type)?;
            for (histogram_algorithm, partition_algorithm, dmem_buffer_size) in iproduct!(
                options.histogram_algorithms,
                options.partition_algorithms,
                options.dmem_buffer_sizes
            ) {
                gpu_radix_partition_benchmark::<i64, _>(
                    "gpu_radix_partition",
                    &(histogram_algorithm.to_string() + &partition_algorithm.to_string()),
                    histogram_algorithm.into(),
                    partition_algorithm.into(),
                    &options.radix_bits,
                    &input_data,
                    &output_mem_type,
                    &grid_size_hint,
                    dmem_buffer_size * 1024,
                    options.repeat,
                    &template,
                    &mut csv_writer,
                )?;
            }
        }
    }

    Ok(())
}
