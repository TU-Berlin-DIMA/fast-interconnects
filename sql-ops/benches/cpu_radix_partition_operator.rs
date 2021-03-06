/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2021 Clemens Lutz, German Research Center for Artificial Intelligence
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use datagen::relation::{KeyAttribute, UniformRelation, ZipfRelation};
use itertools::{iproduct, izip};
use num_traits::cast::FromPrimitive;
use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemType};
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::hw_info;
use numa_gpu::runtime::linux_wrapper;
use numa_gpu::runtime::memory::{DerefMem, MemLock};
use numa_gpu::runtime::numa::PageType;
use rustacuda::memory::DeviceCopy;
use serde_derive::Serialize;
use serde_repr::Serialize_repr;
use sql_ops::partition::cpu_radix_partition::{
    CpuHistogramAlgorithm, CpuRadixPartitionAlgorithm, CpuRadixPartitionable, CpuRadixPartitioner,
};
use sql_ops::partition::{PartitionOffsets, PartitionedRelation, RadixPartitionInputChunkable};
use std::error::Error;
use std::fs;
use std::io::Write;
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use structopt::clap::arg_enum;
use structopt::StructOpt;

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgPageType {
        Default,
        Small,
        TransparentHuge,
        Huge2MB,
        Huge1GB,
    }
}

impl From<ArgPageType> for PageType {
    fn from(arg_page_type: ArgPageType) -> PageType {
        match arg_page_type {
            ArgPageType::Default => PageType::Default,
            ArgPageType::Small => PageType::Small,
            ArgPageType::TransparentHuge => PageType::TransparentHuge,
            ArgPageType::Huge2MB => PageType::Huge2MB,
            ArgPageType::Huge1GB => PageType::Huge1GB,
        }
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgHistogramAlgorithm {
        Chunked,
        ChunkedSimd,
    }
}

impl Into<CpuHistogramAlgorithm> for ArgHistogramAlgorithm {
    fn into(self) -> CpuHistogramAlgorithm {
        match self {
            Self::Chunked => CpuHistogramAlgorithm::Chunked,
            Self::ChunkedSimd => CpuHistogramAlgorithm::ChunkedSimd,
        }
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgRadixPartitionAlgorithm {
        NC,
        Swwc,
        SwwcSimd,
    }
}

impl Into<CpuRadixPartitionAlgorithm> for ArgRadixPartitionAlgorithm {
    fn into(self) -> CpuRadixPartitionAlgorithm {
        match self {
            Self::NC => CpuRadixPartitionAlgorithm::NC,
            Self::Swwc => CpuRadixPartitionAlgorithm::Swwc,
            Self::SwwcSimd => CpuRadixPartitionAlgorithm::SwwcSimd,
        }
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

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgDataDistribution {
        Uniform,
        Unique,
        Zipf,
    }
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "CPU Radix Partition Benchmark",
    about = "A benchmark of the CPU radix partition operator."
)]
struct Options {
    /// Select the prefix sum algorithms to run
    #[structopt(
        long,
        default_value = "Chunked",
        possible_values = &ArgHistogramAlgorithm::variants(),
        case_insensitive = true,
        require_delimiter = true
    )]
    prefix_sum_algorithms: Vec<ArgHistogramAlgorithm>,

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

    #[structopt(
        long = "radix-bits",
        default_value = "8,10,12,14,16",
        require_delimiter = true
    )]
    /// Radix bits with which to partition
    radix_bits: Vec<u32>,

    #[structopt(long = "threads")]
    threads: Option<usize>,

    /// Path to CPU affinity map file for CPU workers
    #[structopt(long = "cpu-affinity", parse(from_os_str))]
    cpu_affinity: Option<PathBuf>,

    /// Allocate memory for input relation on NUMA node (See numactl -H)
    #[structopt(long = "input-location", default_value = "0")]
    input_location: u16,

    /// Allocate memory for output relation on NUMA node (See numactl -H)
    #[structopt(long = "output-location", default_value = "0")]
    output_location: u16,

    /// Page type with with to allocate memory
    #[structopt(
        long = "page-type",
        default_value = "Default",
        possible_values = &ArgPageType::variants(),
        case_insensitive = true
    )]
    page_type: ArgPageType,

    /// Relation's data distribution
    #[structopt(
        long = "data-distribution",
        default_value = "Uniform",
        possible_values = &ArgDataDistribution::variants(),
        case_insensitive = true
    )]
    data_distribution: ArgDataDistribution,

    /// Zipf exponent for Zipf-sampled relation
    #[structopt(
        long = "zipf-exponent",
        required_ifs(&[("data-distribution", "Zipf"), ("data-distribution", "zipf")])
    )]
    zipf_exponent: Option<f64>,

    /// Output path for the measurements CSV file
    #[structopt(long, default_value = "target/bench/cpu_radix_partition_operator.csv")]
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
    pub input_location: Option<u16>,
    pub output_location: Option<u16>,
    pub page_type: Option<ArgPageType>,
    pub tuple_bytes: Option<ArgTupleBytes>,
    pub tuples: Option<usize>,
    pub data_distribution: Option<ArgDataDistribution>,
    pub zipf_exponent: Option<f64>,
    pub radix_bits: Option<u32>,
    pub warm_up: Option<bool>,
    pub prefix_sum_ns: Option<u128>,
    pub partition_ns: Option<u128>,
}

fn cpu_radix_partition_benchmark<T, W>(
    bench_group: &str,
    bench_function: &str,
    prefix_sum_algorithm: CpuHistogramAlgorithm,
    partition_algorithm: CpuRadixPartitionAlgorithm,
    radix_bits_list: &[u32],
    input_data: &(DerefMem<T>, DerefMem<T>),
    output_mem_type: &MemType,
    threads: usize,
    cpu_affinity: &CpuAffinity,
    repeat: u32,
    template: &DataPoint,
    csv_writer: &mut csv::Writer<W>,
) -> Result<(), Box<dyn Error>>
where
    T: Clone + Default + Send + Sync + FromPrimitive + CpuRadixPartitionable,
    W: Write,
{
    let tuples = input_data.0.len();

    let template = DataPoint {
        group: bench_group.to_string(),
        function: bench_function.to_string(),
        ..template.clone()
    };

    let boxed_cpu_affinity = Arc::new(cpu_affinity.clone());
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .start_handler(move |tid| {
            boxed_cpu_affinity
                .clone()
                .set_affinity(tid as u16)
                .expect("Couldn't set CPU core affinity")
        })
        .build()?;

    radix_bits_list
        .iter()
        .map(|&radix_bits| {
            let mut radix_prnrs: Vec<_> = (0..threads)
                .map(|tid| {
                    let cpu_id = cpu_affinity
                        .thread_to_cpu(tid as u16)
                        .expect("Failed to map thread ID to CPU ID");
                    let local_node = linux_wrapper::numa_node_of_cpu(cpu_id)
                        .expect("Failed to map CPU to NUMA node");
                    CpuRadixPartitioner::new(
                        prefix_sum_algorithm,
                        partition_algorithm,
                        radix_bits,
                        DerefMemType::NumaMem {
                            node: local_node,
                            page_type: PageType::Default,
                        },
                    )
                })
                .collect();

            let mut partition_offsets = PartitionOffsets::new(
                prefix_sum_algorithm.into(),
                threads as u32,
                radix_bits,
                Allocator::mem_alloc_fn(MemType::SysMem),
            );
            partition_offsets.mlock()?;

            let mut partitioned_relation = PartitionedRelation::new(
                tuples,
                prefix_sum_algorithm.into(),
                radix_bits,
                threads as u32,
                Allocator::mem_alloc_fn(output_mem_type.clone()),
                Allocator::mem_alloc_fn(output_mem_type.clone()),
            );
            partitioned_relation.mlock()?;

            let result: Result<(), Box<dyn Error>> = (0..repeat)
                .zip(std::iter::once(true).chain(std::iter::repeat(false)))
                .try_for_each(|(_, warm_up)| {
                    let prefix_sum_timer = Instant::now();

                    let data_key_chunks = input_data.0.input_chunks::<T>(threads as u32)?;
                    thread_pool.scope(|s| {
                        for (_tid, radix_prnr, key_chunk, offsets_chunk) in izip!(
                            0..threads,
                            radix_prnrs.iter_mut(),
                            data_key_chunks.into_iter(),
                            partition_offsets.chunks_mut()
                        ) {
                            s.spawn(move |_| {
                                radix_prnr
                                    .prefix_sum(key_chunk, offsets_chunk)
                                    .expect("Failed to prefix sum the data");
                            });
                        }
                    });

                    let prefix_sum_time = prefix_sum_timer.elapsed();
                    let partition_timer = Instant::now();

                    let data_key_chunks = input_data.0.input_chunks::<T>(threads as u32)?;
                    let data_pay_chunks = input_data.1.input_chunks::<T>(threads as u32)?;
                    thread_pool.scope(|s| {
                        for (
                            _tid,
                            radix_prnr,
                            key_chunk,
                            pay_chunk,
                            offsets_chunk,
                            partitioned_chunk,
                        ) in izip!(
                            0..threads,
                            radix_prnrs.iter_mut(),
                            data_key_chunks.into_iter(),
                            data_pay_chunks.into_iter(),
                            partition_offsets.chunks_mut(),
                            partitioned_relation.chunks_mut()
                        ) {
                            s.spawn(move |_| {
                                radix_prnr
                                    .partition(
                                        key_chunk,
                                        pay_chunk,
                                        offsets_chunk,
                                        partitioned_chunk,
                                    )
                                    .expect("Failed to partition the data");
                            });
                        }
                    });

                    let partition_time = partition_timer.elapsed();

                    let dp = DataPoint {
                        radix_bits: Some(radix_bits),
                        threads: Some(threads),
                        warm_up: Some(warm_up),
                        prefix_sum_ns: Some(prefix_sum_time.as_nanos()),
                        partition_ns: Some(partition_time.as_nanos()),
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

fn alloc_and_gen<T>(
    tuples: usize,
    mem_type: &DerefMemType,
    data_distribution: ArgDataDistribution,
    zipf_exponent: Option<f64>,
) -> Result<(DerefMem<T>, DerefMem<T>), Box<dyn Error>>
where
    T: Clone + Default + Send + DeviceCopy + FromPrimitive + KeyAttribute,
{
    let key_range = tuples;
    const PAYLOAD_RANGE: RangeInclusive<usize> = 1..=10000;

    let mut data_key = Allocator::alloc_deref_mem(mem_type.clone(), tuples);
    let mut data_pay = Allocator::alloc_deref_mem(mem_type.clone(), tuples);

    data_key.mlock()?;
    data_pay.mlock()?;

    match data_distribution {
        ArgDataDistribution::Unique => {
            UniformRelation::gen_primary_key_par(data_key.as_mut_slice(), None)?;
        }
        ArgDataDistribution::Uniform => {
            UniformRelation::gen_attr_par(data_key.as_mut_slice(), 1..=key_range)?;
        }
        ArgDataDistribution::Zipf if !(zipf_exponent.unwrap() > 0.0) => {
            UniformRelation::gen_attr_par(data_key.as_mut_slice(), 1..=key_range)?;
        }
        ArgDataDistribution::Zipf => {
            ZipfRelation::gen_attr_par(data_key.as_mut_slice(), key_range, zipf_exponent.unwrap())?;
        }
    }

    UniformRelation::gen_attr_par(data_pay.as_mut_slice(), PAYLOAD_RANGE)?;

    Ok((data_key, data_pay))
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::from_args();

    let threads = if let Some(threads) = options.threads {
        threads
    } else {
        num_cpus::get_physical()
    };

    let cpu_affinity = if let Some(ref cpu_affinity_file) = options.cpu_affinity {
        CpuAffinity::from_file(cpu_affinity_file.as_path())?
    } else {
        CpuAffinity::default()
    };

    let input_mem_type = DerefMemType::NumaMem {
        node: options.input_location,
        page_type: options.page_type.into(),
    };
    let output_mem_type = MemType::NumaMem {
        node: options.output_location,
        page_type: options.page_type.into(),
    };

    if let Some(parent) = options.csv.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    let csv_file = std::fs::File::create(&options.csv)?;
    let mut csv_writer = csv::Writer::from_writer(csv_file);

    let template = DataPoint {
        hostname: hostname::get()?
            .into_string()
            .expect("Couldn't convert hostname into UTF-8 string"),
        device_codename: Some(hw_info::cpu_codename()?),
        input_location: Some(options.input_location),
        output_location: Some(options.output_location),
        page_type: Some(options.page_type),
        tuple_bytes: Some(options.tuple_bytes),
        tuples: Some(options.tuples),
        data_distribution: Some(options.data_distribution),
        zipf_exponent: options.zipf_exponent,
        ..DataPoint::default()
    };

    match options.tuple_bytes {
        ArgTupleBytes::Bytes8 => {
            let input_data = alloc_and_gen(
                options.tuples,
                &input_mem_type,
                options.data_distribution,
                options.zipf_exponent,
            )?;
            for (prefix_sum_algorithm, partition_algorithm) in
                iproduct!(options.prefix_sum_algorithms, options.partition_algorithms)
            {
                cpu_radix_partition_benchmark::<i32, _>(
                    "cpu_radix_partition",
                    &(prefix_sum_algorithm.to_string() + &partition_algorithm.to_string()),
                    prefix_sum_algorithm.into(),
                    partition_algorithm.into(),
                    &options.radix_bits,
                    &input_data,
                    &output_mem_type,
                    threads,
                    &cpu_affinity,
                    options.repeat,
                    &template,
                    &mut csv_writer,
                )?;
            }
        }
        ArgTupleBytes::Bytes16 => {
            let input_data = alloc_and_gen(
                options.tuples,
                &input_mem_type,
                options.data_distribution,
                options.zipf_exponent,
            )?;
            for (prefix_sum_algorithm, partition_algorithm) in
                iproduct!(options.prefix_sum_algorithms, options.partition_algorithms)
            {
                cpu_radix_partition_benchmark::<i64, _>(
                    "cpu_radix_partition",
                    &(prefix_sum_algorithm.to_string() + &partition_algorithm.to_string()),
                    prefix_sum_algorithm.into(),
                    partition_algorithm.into(),
                    &options.radix_bits,
                    &input_data,
                    &output_mem_type,
                    threads,
                    &cpu_affinity,
                    options.repeat,
                    &template,
                    &mut csv_writer,
                )?;
            }
        }
    }

    Ok(())
}
