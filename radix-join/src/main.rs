/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2020 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

mod error;
mod execution_methods;
mod measurement;
mod types;

use crate::error::Result;
use crate::execution_methods::gpu_radix_join::gpu_radix_join;
use crate::measurement::data_point::DataPoint;
use crate::measurement::harness::{self, RadixJoinPoint};
use crate::types::*;
use data_store::join_data::{JoinDataBuilder, JoinDataGenFn};
use datagen::relation::KeyAttribute;
use num_rational::Ratio;
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::numa::NodeRatio;
use rustacuda::device::DeviceAttribute;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::DeviceCopy;
use rustacuda::prelude::*;
use serde::de::DeserializeOwned;
use sql_ops::join::{cuda_radix_join, no_partitioning_join, HashingScheme};
use sql_ops::partition::gpu_radix_partition::GpuRadixPartitionable;
use std::mem::size_of;
use std::path::PathBuf;
use structopt::StructOpt;

fn main() -> Result<()> {
    // Parse commandline arguments
    let cmd = CmdOpt::from_args();

    // Initialize CUDA
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(cmd.device_id.into())?;
    let _context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    match cmd.tuple_bytes {
        ArgTupleBytes::Bytes8 => {
            let (hjc, dp) = args_to_bench::<i32>(&cmd, device)?;
            harness::measure("radix_join", cmd.repeat, cmd.csv, dp, hjc)?;
        }
        ArgTupleBytes::Bytes16 => {
            unimplemented!();
            // let (hjc, dp) = args_to_bench::<i64>(&cmd, device)?;
            // harness::measure("radix_join", cmd.repeat, cmd.csv, dp, hjc)?;
        }
    };

    Ok(())
}

#[derive(StructOpt)]
#[structopt(
    name = "radix-join",
    about = "A partitioned hash join optimized for large-to-large joins on GPUs with fast interconnects"
)]
struct CmdOpt {
    /// Number of times to repeat benchmark
    #[structopt(short = "r", long = "repeat", default_value = "30")]
    repeat: u32,

    /// Output filename for measurement CSV file
    #[structopt(long = "csv", parse(from_os_str))]
    csv: Option<PathBuf>,

    /// Memory type with which to allocate data.
    //   unified: CUDA Unified memory (default)
    //   numa: NUMA-local memory on node specified with [inner,outer]-rel-location
    #[structopt(
        long = "rel-mem-type",
        default_value = "Unified",
        possible_values = &ArgMemType::variants(),
        case_insensitive = true
    )]
    mem_type: ArgMemType,

    /// Hashing scheme to use in hash table.
    //   linearprobing: Linear probing (default)
    //   perfect: Perfect hashing for unique primary keys
    #[structopt(
        long = "hashing-scheme",
        default_value = "LinearProbing",
        possible_values = &ArgHashingScheme::variants(),
        case_insensitive = true
    )]
    hashing_scheme: ArgHashingScheme,

    /// Memory type with which to allocate the partitioned data.
    #[structopt(
        long = "partitions-mem-type",
        default_value = "Unified",
        possible_values = &ArgMemType::variants(),
        case_insensitive = true
    )]
    partitions_mem_type: ArgMemType,

    /// Allocate memory for the partitioned data on NUMA nodes (e.g.: 0,1,2) or GPU (See numactl -H and CUDA device list)
    #[structopt(
        long = "partitions-location",
        default_value = "0",
        require_delimiter = true
    )]
    partitions_location: Vec<u16>,

    /// Proportions with which the partitioned data are allocated on multiple nodes; in percent (e.g.: 20,60,20)
    #[structopt(
        long = "partitions-proportions",
        default_value = "100",
        require_delimiter = true
    )]
    partitions_proportions: Vec<usize>,

    /// Allocate memory for inner relation on CPU or GPU (See numactl -H and CUDA device list)
    #[structopt(long = "inner-rel-location", default_value = "0")]
    inner_rel_location: u16,

    /// Allocate memory for outer relation on CPU or GPU (See numactl -H and CUDA device list)
    #[structopt(long = "outer-rel-location", default_value = "0")]
    outer_rel_location: u16,

    /// Use small pages (false) or huge pages (true); no selection defaults to the OS configuration
    #[structopt(long = "huge-pages")]
    huge_pages: Option<bool>,

    /// Use a pre-defined or custom data set.
    //   blanas: Blanas et al. "Main memory hash join algorithms for multi-core CPUs"
    //   blanas4mb: Blanas, but with a 4 MiB inner relation
    //   kim: Kim et al. "Sort vs. hash revisited"
    //   test: A small data set for testing on the laptop
    #[structopt(
        short = "s",
        long = "data-set",
        default_value = "Test",
        possible_values = &ArgDataSet::variants(),
        case_insensitive = true
    )]
    data_set: ArgDataSet,

    /// Outer relation's data distribution
    #[structopt(
        long = "data-distribution",
        default_value = "Uniform",
        possible_values = &ArgDataDistribution::variants(),
        case_insensitive = true
    )]
    data_distribution: ArgDataDistribution,

    /// Zipf exponent for Zipf-sampled outer relations
    #[structopt(long = "zipf-exponent", required_if("data-distribution", "Zipf"))]
    zipf_exponent: Option<f64>,

    /// Selectivity of the join, in percent
    #[structopt(
        long = "selectivity",
        default_value = "100",
        validator = is_percent
    )]
    selectivity: u32,

    /// Load data set from a TSV file with "key value" pairs and automatic gzip decompression
    #[structopt(
        long = "inner-rel-file",
        parse(from_os_str),
        conflicts_with = "data_set",
        requires = "outer_rel_file"
    )]
    inner_rel_file: Option<PathBuf>,

    /// Load data set from a TSV file with "key value" pairs and automatic gzip decompression
    #[structopt(
        long = "outer-rel-file",
        parse(from_os_str),
        conflicts_with = "data_set",
        requires = "inner_rel_file"
    )]
    outer_rel_file: Option<PathBuf>,

    /// Set the tuple size (bytes)
    #[structopt(
        long = "tuple-bytes",
        default_value = "Bytes8",
        possible_values = &ArgTupleBytes::variants(),
        case_insensitive = true
    )]
    tuple_bytes: ArgTupleBytes,

    /// Set the inner relation size (tuples); required for `-data-set Custom`
    #[structopt(long = "inner-rel-tuples", required_if("data_set", "Custom"))]
    inner_rel_tuples: Option<usize>,

    /// Set the outer relation size (tuples); required for `--data-set Custom`
    #[structopt(long = "outer-rel-tuples", required_if("data_set", "Custom"))]
    outer_rel_tuples: Option<usize>,

    /// Select the histogram algorithm to run
    #[structopt(
        long,
        default_value = "GpuChunked",
        possible_values = &ArgHistogramAlgorithm::variants(),
        case_insensitive = true
    )]
    histogram_algorithm: ArgHistogramAlgorithm,

    /// Select the radix partition algorithm to run
    #[structopt(
        long,
        default_value = "NC",
        possible_values = &ArgRadixPartitionAlgorithm::variants(),
        case_insensitive = true
    )]
    partition_algorithm: ArgRadixPartitionAlgorithm,

    /// Join execution strategy.
    #[structopt(
        long = "execution-strategy",
        default_value = "GpuRJ",
        possible_values = &ArgExecutionMethod::variants(),
        case_insensitive = true
    )]
    execution_method: ArgExecutionMethod,

    #[structopt(long = "radix-bits", default_value = "8")]
    /// Radix bits with which to partition
    radix_bits: u32,

    #[structopt(short = "i", long = "device-id", default_value = "0")]
    /// Execute on GPU (See CUDA device list)
    device_id: u16,

    /// Device memory buffer sizes per partition per thread block for HSSWWC variants (in KiB)
    #[structopt(long, default_value = "8", require_delimiter = true)]
    dmem_buffer_size: usize,

    #[structopt(short = "t", long = "threads", default_value = "1")]
    threads: usize,

    /// Path to CPU affinity map file for CPU workers
    #[structopt(long = "cpu-affinity", parse(from_os_str))]
    cpu_affinity: Option<PathBuf>,
}

fn is_percent(x: String) -> std::result::Result<(), String> {
    x.parse::<i32>()
        .map_err(|_| {
            String::from(
                "Failed to parse integer. The value must be a percentage between [0, 100].",
            )
        })
        .and_then(|x| {
            if 0 <= x && x <= 100 {
                Ok(())
            } else {
                Err(String::from(
                    "The value must be a percentage between [0, 100].",
                ))
            }
        })
}

fn args_to_bench<T>(
    cmd: &CmdOpt,
    device: Device,
) -> Result<(Box<dyn FnMut() -> Result<RadixJoinPoint>>, DataPoint)>
where
    T: Default
        + DeviceCopy
        + Sync
        + Send
        + GpuRadixPartitionable
        + no_partitioning_join::NullKey
        + no_partitioning_join::CudaHashJoinable
        + no_partitioning_join::CpuHashJoinable
        + cuda_radix_join::CudaRadixJoinable
        + KeyAttribute
        + num_traits::FromPrimitive
        + DeserializeOwned,
{
    // Device tuning
    let multiprocessors = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
    let warp_size = device.get_attribute(DeviceAttribute::WarpSize)? as u32;
    let warp_overcommit_factor = 4;
    let grid_overcommit_factor = 2;

    let block_size = BlockSize::x(warp_size * warp_overcommit_factor);
    let grid_size = GridSize::x(multiprocessors * grid_overcommit_factor);

    let huge_pages = cmd.huge_pages;

    let mut data_builder = JoinDataBuilder::default();
    data_builder
        .inner_mem_type(
            ArgMemTypeHelper {
                mem_type: cmd.mem_type,
                node_ratios: Box::new([NodeRatio {
                    node: cmd.inner_rel_location,
                    ratio: Ratio::from_integer(1),
                }]),
                huge_pages,
            }
            .into(),
        )
        .outer_mem_type(
            ArgMemTypeHelper {
                mem_type: cmd.mem_type,
                node_ratios: Box::new([NodeRatio {
                    node: cmd.outer_rel_location,
                    ratio: Ratio::from_integer(1),
                }]),
                huge_pages,
            }
            .into(),
        );

    let exec_method = cmd.execution_method;
    let histogram_algorithm = cmd.histogram_algorithm;
    let partition_algorithm = cmd.partition_algorithm;
    let radix_bits = cmd.radix_bits;
    let dmem_buffer_bytes = cmd.dmem_buffer_size * 1024; // convert KiB to bytes
    let mem_type = cmd.partitions_mem_type;
    let threads = cmd.threads;

    // Convert ArgHashingScheme to HashingScheme
    let hashing_scheme = HashingScheme::from(cmd.hashing_scheme);

    let node_ratios: Box<[NodeRatio]> = cmd
        .partitions_location
        .iter()
        .zip(cmd.partitions_proportions.iter())
        .map(|(node, pct)| NodeRatio {
            node: *node,
            ratio: Ratio::new(*pct, 100),
        })
        .collect();

    // Load file or generate data set
    let (mut join_data, malloc_time, data_gen_time) =
        if let (Some(inner_rel_path), Some(outer_rel_path)) = (
            cmd.inner_rel_file.as_ref().and_then(|p| p.to_str()),
            cmd.outer_rel_file.as_ref().and_then(|p| p.to_str()),
        ) {
            data_builder.build_with_files::<T>(inner_rel_path, outer_rel_path)?
        } else {
            let data_distribution = match cmd.data_distribution {
                ArgDataDistribution::Uniform => DataDistribution::Uniform,
                ArgDataDistribution::Zipf => DataDistribution::Zipf(cmd.zipf_exponent.unwrap()),
            };

            let (inner_relation_len, outer_relation_len, data_gen) = data_gen_fn::<_>(
                cmd.data_set,
                cmd.inner_rel_tuples,
                cmd.outer_rel_tuples,
                data_distribution,
                Some(cmd.selectivity),
            );
            data_builder
                .inner_len(inner_relation_len)
                .outer_len(outer_relation_len)
                .build_with_data_gen(data_gen)?
        };

    // Construct data point template for CSV
    let dp = DataPoint::new()?
        .fill_from_cmd_options(cmd)?
        .fill_from_join_data(&join_data)
        .set_init_time(malloc_time, data_gen_time);

    let cpu_affinity = if let Some(ref cpu_affinity_file) = cmd.cpu_affinity {
        CpuAffinity::from_file(cpu_affinity_file.as_path())?
    } else {
        CpuAffinity::default()
    };

    // Create closure that wraps a hash join benchmark function
    let hjc: Box<dyn FnMut() -> Result<RadixJoinPoint>> = match exec_method {
        ArgExecutionMethod::GpuRJ => Box::new(move || {
            let partitions_mem_type = ArgMemTypeHelper {
                mem_type,
                node_ratios: node_ratios.clone(),
                huge_pages,
            }
            .into();

            gpu_radix_join(
                &mut join_data,
                hashing_scheme,
                histogram_algorithm.into(),
                partition_algorithm.into(),
                radix_bits,
                dmem_buffer_bytes,
                threads,
                cpu_affinity.clone(),
                partitions_mem_type,
                (grid_size.clone(), block_size.clone()),
                (grid_size.clone(), block_size.clone()),
                (grid_size.clone(), block_size.clone()),
            )
        }),
    };

    Ok((hjc, dp))
}

fn data_gen_fn<T>(
    description: ArgDataSet,
    inner_rel_tuples: Option<usize>,
    outer_rel_tuples: Option<usize>,
    data_distribution: DataDistribution,
    selectivity: Option<u32>,
) -> (usize, usize, JoinDataGenFn<T>)
where
    T: Copy + Send + KeyAttribute + num_traits::FromPrimitive,
{
    match description {
        ArgDataSet::Blanas => (
            datagen::popular::Blanas::primary_key_len(),
            datagen::popular::Blanas::foreign_key_len(),
            Box::new(move |pk_rel, fk_rel| {
                datagen::popular::Blanas::gen(pk_rel, fk_rel, selectivity).map_err(|e| e.into())
            }),
        ),
        ArgDataSet::Kim => (
            datagen::popular::Kim::primary_key_len(),
            datagen::popular::Kim::foreign_key_len(),
            Box::new(move |pk_rel, fk_rel| {
                datagen::popular::Kim::gen(pk_rel, fk_rel, selectivity).map_err(|e| e.into())
            }),
        ),
        ArgDataSet::Blanas4MB => {
            let gen = move |pk_rel: &mut [_], fk_rel: &mut [_]| {
                datagen::relation::UniformRelation::gen_primary_key_par(pk_rel, selectivity)?;
                datagen::relation::UniformRelation::gen_attr_par(fk_rel, 1..=pk_rel.len())?;
                Ok(())
            };

            (512 * 2_usize.pow(10), 256 * 2_usize.pow(20), Box::new(gen))
        }
        ArgDataSet::Test => {
            let gen = move |pk_rel: &mut [_], fk_rel: &mut [_]| {
                datagen::relation::UniformRelation::gen_primary_key(pk_rel, selectivity)?;
                datagen::relation::UniformRelation::gen_foreign_key_from_primary_key(
                    fk_rel, pk_rel,
                );
                Ok(())
            };

            (1000, 1000, Box::new(gen))
        }
        ArgDataSet::Lutz2Gv32G => {
            let gen = move |pk_rel: &mut [_], fk_rel: &mut [_]| {
                datagen::relation::UniformRelation::gen_primary_key_par(pk_rel, selectivity)?;
                datagen::relation::UniformRelation::gen_attr_par(fk_rel, 1..=pk_rel.len())?;
                Ok(())
            };

            (
                2 * 2_usize.pow(30) / (2 * size_of::<T>()),
                32 * 2_usize.pow(30) / (2 * size_of::<T>()),
                Box::new(gen),
            )
        }
        ArgDataSet::Lutz32Gv32G => {
            let gen = move |pk_rel: &mut [_], fk_rel: &mut [_]| {
                datagen::relation::UniformRelation::gen_primary_key_par(pk_rel, selectivity)?;
                datagen::relation::UniformRelation::gen_attr_par(fk_rel, 1..=pk_rel.len())?;
                Ok(())
            };

            (
                32 * 2_usize.pow(30) / (2 * size_of::<T>()),
                32 * 2_usize.pow(30) / (2 * size_of::<T>()),
                Box::new(gen),
            )
        }
        ArgDataSet::Custom => {
            let uniform_gen = Box::new(move |pk_rel: &mut [_], fk_rel: &mut [_]| {
                datagen::relation::UniformRelation::gen_primary_key_par(pk_rel, selectivity)?;
                datagen::relation::UniformRelation::gen_attr_par(fk_rel, 1..=pk_rel.len())?;
                Ok(())
            });

            let gen: JoinDataGenFn<T> = match data_distribution {
                DataDistribution::Uniform => uniform_gen,
                DataDistribution::Zipf(exp) if !(exp > 0.0) => uniform_gen,
                DataDistribution::Zipf(exp) => {
                    Box::new(move |pk_rel: &mut [_], fk_rel: &mut [_]| {
                        datagen::relation::UniformRelation::gen_primary_key_par(
                            pk_rel,
                            selectivity,
                        )?;
                        datagen::relation::ZipfRelation::gen_attr_par(fk_rel, pk_rel.len(), exp)?;
                        Ok(())
                    })
                }
            };

            (
                inner_rel_tuples.expect(
                    "Couldn't find inner relation size. Did you specify --inner-rel-tuples?",
                ),
                outer_rel_tuples.expect(
                    "Couldn't find outer relation size. Did you specify --outer-rel-tuples?",
                ),
                gen,
            )
        }
    }
}
