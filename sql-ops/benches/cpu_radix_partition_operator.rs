/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2020 Clemens Lutz, German Research Center for Artificial Intelligence
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use datagen::relation::{KeyAttribute, UniformRelation, ZipfRelation};
use num_traits::cast::FromPrimitive;
use numa_gpu::runtime::allocator::{Allocator, DerefMemType};
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::hw_info;
use numa_gpu::runtime::memory::DerefMem;
use rustacuda::memory::DeviceCopy;
use serde_derive::Serialize;
use serde_repr::Serialize_repr;
use sql_ops::partition::cpu_radix_partition::{
    CpuRadixPartitionAlgorithm, CpuRadixPartitionable, CpuRadixPartitioner, PartitionedRelation,
};
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
    pub enum ArgRadixPartitionAlgorithm {
        Chunked,
        ChunkedSwwc,
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

impl Into<CpuRadixPartitionAlgorithm> for ArgRadixPartitionAlgorithm {
    fn into(self) -> CpuRadixPartitionAlgorithm {
        match self {
            Self::Chunked => CpuRadixPartitionAlgorithm::Chunked,
            Self::ChunkedSwwc => CpuRadixPartitionAlgorithm::ChunkedSwwc,
        }
    }
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "CPU Radix Partition Benchmark",
    about = "A benchmark of the CPU radix partition operator."
)]
struct Options {
    /// Select the algorithms to run
    #[structopt(
        long,
        default_value = "Chunked",
        possible_values = &ArgRadixPartitionAlgorithm::variants(),
        case_insensitive = true,
        require_delimiter = true
    )]
    algorithms: Vec<ArgRadixPartitionAlgorithm>,

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

    /// Allocate memory for base relation on NUMA node (See numactl -H)
    #[structopt(long = "rel-location", default_value = "0")]
    rel_location: u16,

    /// Use small pages (false) or huge pages (true); no selection defaults to the OS configuration
    #[structopt(long = "huge-pages")]
    huge_pages: Option<bool>,

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
    pub huge_pages: Option<bool>,
    pub tuple_bytes: Option<ArgTupleBytes>,
    pub tuples: Option<usize>,
    pub data_distribution: Option<ArgDataDistribution>,
    pub zipf_exponent: Option<f64>,
    pub radix_bits: Option<u32>,
    pub ns: Option<u128>,
}

fn cpu_radix_partition_benchmark<T, W>(
    bench_group: &str,
    bench_function: &str,
    algorithm: CpuRadixPartitionAlgorithm,
    radix_bits_list: &[u32],
    input_data: &(DerefMem<T>, DerefMem<T>),
    output_mem_type: &DerefMemType,
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
    let chunk_len = (tuples + threads - 1) / threads;
    let data_key_chunks: Vec<_> = input_data.0.chunks(chunk_len).collect();
    let data_pay_chunks: Vec<_> = input_data.1.chunks(chunk_len).collect();

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
                .map(|_| {
                    CpuRadixPartitioner::new(
                        algorithm,
                        radix_bits,
                        Allocator::deref_mem_alloc_fn(output_mem_type.clone()),
                    )
                })
                .collect();

            let mut partitioned_relation_chunks: Vec<_> = data_key_chunks
                .iter()
                .map(|chunk| chunk.len())
                .map(|chunk_len| {
                    PartitionedRelation::new(
                        chunk_len,
                        radix_bits,
                        Allocator::deref_mem_alloc_fn(output_mem_type.clone()),
                        Allocator::deref_mem_alloc_fn(output_mem_type.clone()),
                    )
                })
                .collect();

            let result: Result<(), Box<dyn Error>> = (0..repeat).into_iter().try_for_each(|_| {
                let timer = Instant::now();

                thread_pool.scope(|s| {
                    for ((((_tid, radix_prnr), key_chunk), pay_chunk), partitioned_chunk) in (0
                        ..threads)
                        .zip(radix_prnrs.iter_mut())
                        .zip(data_key_chunks.iter())
                        .zip(data_pay_chunks.iter())
                        .zip(partitioned_relation_chunks.iter_mut())
                    {
                        s.spawn(move |_| {
                            radix_prnr
                                .partition(key_chunk, pay_chunk, partitioned_chunk)
                                .expect("Failed to partition data");
                        });
                    }
                });

                let time = timer.elapsed();

                let dp = DataPoint {
                    radix_bits: Some(radix_bits),
                    threads: Some(threads),
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

    let input_mem_type = DerefMemType::NumaMem(options.rel_location, options.huge_pages);
    let output_mem_type = DerefMemType::NumaMem(options.rel_location, options.huge_pages);

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
        input_location: Some(options.rel_location),
        output_location: Some(options.rel_location),
        huge_pages: options.huge_pages,
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
            for algorithm in options.algorithms {
                cpu_radix_partition_benchmark::<i32, _>(
                    "cpu_radix_partition",
                    &algorithm.to_string(),
                    algorithm.into(),
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
            for algorithm in options.algorithms {
                cpu_radix_partition_benchmark::<i64, _>(
                    "cpu_radix_partition",
                    &algorithm.to_string(),
                    algorithm.into(),
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
