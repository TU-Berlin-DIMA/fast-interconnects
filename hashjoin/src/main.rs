/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

mod error;
mod measurement;
pub mod operators;
mod types;

use crate::error::Result;
use crate::measurement::data_point::DataPoint;
use crate::measurement::harness;
use crate::measurement::hash_join_bench::{HashJoinBenchBuilder, HashJoinPoint};
use crate::operators::hash_join;
use crate::types::*;

use numa_gpu::runtime::allocator;
use numa_gpu::runtime::hw_info::CudaDeviceInfo;
use numa_gpu::runtime::numa::NodeRatio;
use numa_gpu::runtime::utils::EnsurePhysicallyBacked;

use num_rational::Ratio;

use rustacuda::device::DeviceAttribute;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::DeviceCopy;
use rustacuda::prelude::*;

use serde::de::DeserializeOwned;

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
            harness::measure("hash_join_kim", cmd.repeat, cmd.csv, dp, hjc)?;
        }
        ArgTupleBytes::Bytes16 => {
            let (hjc, dp) = args_to_bench::<i64>(&cmd, device)?;
            harness::measure("hash_join_kim", cmd.repeat, cmd.csv, dp, hjc)?;
        }
    };

    Ok(())
}

#[derive(StructOpt)]
#[structopt(name = "hash_join", about = "A benchmark for the hash join operator")]
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
        raw(possible_values = "&ArgMemType::variants()", case_insensitive = "true")
    )]
    mem_type: ArgMemType,

    /// Hashing scheme to use in hash table.
    //   linearprobing: Linear probing (default)
    //   perfect: Perfect hashing for unique primary keys
    #[structopt(
        long = "hashing-scheme",
        default_value = "LinearProbing",
        raw(
            possible_values = "&ArgHashingScheme::variants()",
            case_insensitive = "true"
        )
    )]
    hashing_scheme: ArgHashingScheme,

    /// Memory type with which to allocate hash table.
    //   unified: CUDA Unified memory (default)
    //   numa: NUMA-local memory on node specified with hash-table-location
    #[structopt(
        long = "hash-table-mem-type",
        default_value = "Device",
        raw(possible_values = "&ArgMemType::variants()", case_insensitive = "true")
    )]
    hash_table_mem_type: ArgMemType,

    #[structopt(
        long = "hash-table-location",
        default_value = "0",
        raw(require_delimiter = "true")
    )]
    /// Allocate memory for hash table on NUMA nodes (e.g.: 0,1,2) or GPU (See numactl -H and CUDA device list)
    hash_table_location: Vec<u16>,

    #[structopt(
        long = "hash-table-proportions",
        default_value = "100",
        raw(require_delimiter = "true")
    )]
    /// Proportions with with the hash table is allocate on multiple nodes in percent (e.g.: 20,60,20)
    hash_table_proportions: Vec<usize>,

    #[structopt(long = "inner-rel-location", default_value = "0")]
    /// Allocate memory for inner relation on CPU or GPU (See numactl -H and CUDA device list)
    inner_rel_location: u16,

    #[structopt(long = "outer-rel-location", default_value = "0")]
    /// Allocate memory for outer relation on CPU or GPU (See numactl -H and CUDA device list)
    outer_rel_location: u16,

    /// Use a pre-defined or custom data set.
    //   blanas: Blanas et al. "Main memory hash join algorithms for multi-core CPUs"
    //   blanas4mb: Blanas, but with a 4 MiB inner relation
    //   kim: Kim et al. "Sort vs. hash revisited"
    //   test: A small data set for testing on the laptop
    #[structopt(
        short = "s",
        long = "data-set",
        default_value = "Test",
        raw(possible_values = "&ArgDataSet::variants()", case_insensitive = "true")
    )]
    data_set: ArgDataSet,

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
        raw(
            possible_values = "&ArgTupleBytes::variants()",
            case_insensitive = "true"
        )
    )]
    tuple_bytes: ArgTupleBytes,

    /// Set the inner relation size (tuples); required for `-data-set Custom`
    #[structopt(
        long = "inner-rel-tuples",
        raw(required_if = r#""data_set", "Custom""#)
    )]
    inner_rel_tuples: Option<usize>,

    /// Set the outer relation size (tuples); required for `--data-set Custom`
    #[structopt(
        long = "outer-rel-tuples",
        raw(required_if = r#""data_set", "Custom""#)
    )]
    outer_rel_tuples: Option<usize>,

    /// Execute on device(s) with in-place or streaming-transfer method.
    #[structopt(
        long = "execution-method",
        default_value = "CPU",
        raw(
            possible_values = "&ArgExecutionMethod::variants()",
            case_insensitive = "true"
        )
    )]
    execution_method: ArgExecutionMethod,

    /// Stream data to device using the transfer strategy.
    #[structopt(
        long = "transfer-strategy",
        default_value = "PageableCopy",
        raw(
            possible_values = "&ArgTransferStrategy::variants()",
            case_insensitive = "true"
        )
    )]
    transfer_strategy: ArgTransferStrategy,

    /// Execute stream and transfer with chunk size (bytes)
    #[structopt(long = "chunk-bytes", default_value = "1")]
    chunk_bytes: usize,

    #[structopt(short = "i", long = "device-id", default_value = "0")]
    /// Execute on GPU (See CUDA device list)
    device_id: u16,

    #[structopt(short = "t", long = "threads", default_value = "1")]
    threads: usize,
}

fn args_to_bench<T>(
    cmd: &CmdOpt,
    device: Device,
) -> Result<(Box<dyn FnMut() -> Result<HashJoinPoint>>, DataPoint)>
where
    T: Default
        + DeviceCopy
        + Sync
        + Send
        + hash_join::NullKey
        + hash_join::CudaHashJoinable
        + hash_join::CpuHashJoinable
        + EnsurePhysicallyBacked
        + num_traits::FromPrimitive
        + DeserializeOwned,
{
    // Convert ArgHashingScheme to HashingScheme
    let hashing_scheme = match cmd.hashing_scheme {
        ArgHashingScheme::Perfect => hash_join::HashingScheme::Perfect,
        ArgHashingScheme::LinearProbing => hash_join::HashingScheme::LinearProbing,
    };

    // Device tuning
    let cuda_cores = device.cores()?;
    let warp_size = device.get_attribute(DeviceAttribute::WarpSize)? as u32;
    let warp_overcommit_factor = 2;
    let grid_overcommit_factor = 32;
    let hash_table_load_factor = 2;

    let block_size = BlockSize::x(warp_size * warp_overcommit_factor);
    let grid_size = GridSize::x(cuda_cores * grid_overcommit_factor);

    assert_eq!(
        cmd.hash_table_location.len(),
        cmd.hash_table_proportions.len(),
        "Invalid arguments: Each hash table location must have exactly one proportion."
    );

    if cmd.execution_method == ArgExecutionMethod::GpuStream {
        assert!(
            cmd.mem_type != ArgMemType::Device,
            "Invalid memory type. Streaming execution method cannot be used with device memory."
        );
        assert!(
            cmd.transfer_strategy != ArgTransferStrategy::Unified
                || cmd.mem_type == ArgMemType::Unified,
            "If transfer strategy is \"Unified\", then memory type must also be \"Unified\"."
        );
    }

    let mut hjb_builder = HashJoinBenchBuilder::default();
    hjb_builder
        .hashing_scheme(hashing_scheme)
        .hash_table_load_factor(hash_table_load_factor)
        .inner_location(Box::new([NodeRatio {
            node: cmd.inner_rel_location,
            ratio: Ratio::from_integer(1),
        }]))
        .outer_location(Box::new([NodeRatio {
            node: cmd.outer_rel_location,
            ratio: Ratio::from_integer(1),
        }]))
        .inner_mem_type(cmd.mem_type)
        .outer_mem_type(cmd.mem_type);

    // Select the operator to run, depending on the device type
    let exec_method = cmd.execution_method.clone();
    let transfer_strategy = cmd.transfer_strategy.clone();
    let chunk_len = cmd.chunk_bytes / size_of::<T>();
    let mem_type = cmd.hash_table_mem_type;
    let threads = cmd.threads.clone();
    let device_id = cmd.device_id;

    let node_ratios: Box<[NodeRatio]> = cmd
        .hash_table_location
        .iter()
        .zip(cmd.hash_table_proportions.iter())
        .map(|(node, pct)| NodeRatio {
            node: *node,
            ratio: Ratio::new(*pct, 100),
        })
        .collect();

    // Load file or generate data set
    let (mut hjb, malloc_time, data_gen_time) =
        if let (Some(inner_rel_path), Some(outer_rel_path)) = (
            cmd.inner_rel_file.as_ref().and_then(|p| p.to_str()),
            cmd.outer_rel_file.as_ref().and_then(|p| p.to_str()),
        ) {
            hjb_builder.build_with_files(inner_rel_path, outer_rel_path)?
        } else {
            let (inner_relation_len, outer_relation_len, data_gen) =
                data_gen_fn::<_>(cmd.data_set, cmd.inner_rel_tuples, cmd.outer_rel_tuples);
            hjb_builder
                .inner_len(inner_relation_len)
                .outer_len(outer_relation_len)
                .build_with_data_gen(data_gen)?
        };

    // Construct data point template for CSV
    let dp = DataPoint::new()?
        .fill_from_cmd_options(cmd)?
        .fill_from_hash_join_bench(&hjb)
        .set_init_time(malloc_time, data_gen_time);

    // Create closure that wraps a hash join benchmark function
    let hjc: Box<dyn FnMut() -> Result<HashJoinPoint>> = match exec_method {
        ArgExecutionMethod::Cpu => Box::new(move || {
            let ht_alloc = allocator::Allocator::deref_mem_alloc_fn::<T>(
                ArgMemTypeHelper {
                    mem_type,
                    node_ratios: node_ratios.clone(),
                }
                .into(),
            );
            hjb.cpu_hash_join(threads, ht_alloc)
        }),
        ArgExecutionMethod::Gpu => Box::new(move || {
            let ht_alloc = allocator::Allocator::mem_alloc_fn::<T>(
                ArgMemTypeHelper {
                    mem_type,
                    node_ratios: node_ratios.clone(),
                }
                .into(),
            );
            hjb.cuda_hash_join(
                ht_alloc,
                (grid_size.clone(), block_size.clone()),
                (grid_size.clone(), block_size.clone()),
            )
        }),
        ArgExecutionMethod::GpuStream if transfer_strategy == ArgTransferStrategy::Unified => {
            Box::new(move || {
                let ht_alloc = allocator::Allocator::mem_alloc_fn::<T>(
                    ArgMemTypeHelper {
                        mem_type,
                        node_ratios: node_ratios.clone(),
                    }
                    .into(),
                );
                hjb.cuda_streaming_unified_hash_join(
                    ht_alloc,
                    (grid_size.clone(), block_size.clone()),
                    (grid_size.clone(), block_size.clone()),
                    chunk_len,
                )
            })
        }
        ArgExecutionMethod::GpuStream => Box::new(move || {
            let ht_alloc = allocator::Allocator::mem_alloc_fn::<T>(
                ArgMemTypeHelper {
                    mem_type,
                    node_ratios: node_ratios.clone(),
                }
                .into(),
            );
            hjb.cuda_streaming_hash_join(
                ht_alloc,
                (grid_size.clone(), block_size.clone()),
                (grid_size.clone(), block_size.clone()),
                transfer_strategy.into(),
                chunk_len,
            )
        }),
        ArgExecutionMethod::Het => Box::new(move || {
            let ht_alloc = allocator::Allocator::mem_alloc_fn::<T>(
                ArgMemTypeHelper {
                    mem_type,
                    node_ratios: node_ratios.clone(),
                }
                .into(),
            );
            hjb.hetrogeneous_hash_join(
                ht_alloc,
                (0..threads as u16).into_iter().collect(),
                vec![device_id],
                (grid_size.clone(), block_size.clone()),
                (grid_size.clone(), block_size.clone()),
                chunk_len,
            )
        }),
    };

    Ok((hjc, dp))
}

type DataGenFn<T> = Box<dyn FnMut(&mut [T], &mut [T]) -> Result<()>>;

fn data_gen_fn<T>(
    description: ArgDataSet,
    inner_rel_tuples: Option<usize>,
    outer_rel_tuples: Option<usize>,
) -> (usize, usize, DataGenFn<T>)
where
    T: Copy + Send + num_traits::FromPrimitive,
{
    match description {
        ArgDataSet::Blanas => (
            datagen::popular::Blanas::primary_key_len(),
            datagen::popular::Blanas::foreign_key_len(),
            Box::new(|pk_rel, fk_rel| {
                datagen::popular::Blanas::gen(pk_rel, fk_rel).map_err(|e| e.into())
            }),
        ),
        ArgDataSet::Kim => (
            datagen::popular::Kim::primary_key_len(),
            datagen::popular::Kim::foreign_key_len(),
            Box::new(|pk_rel, fk_rel| {
                datagen::popular::Kim::gen(pk_rel, fk_rel).map_err(|e| e.into())
            }),
        ),
        ArgDataSet::Blanas4MB => {
            let gen = |pk_rel: &mut [_], fk_rel: &mut [_]| {
                datagen::relation::UniformRelation::gen_primary_key_par(pk_rel)?;
                datagen::relation::UniformRelation::gen_attr_par(fk_rel, 1..=pk_rel.len())?;
                Ok(())
            };

            (512 * 2_usize.pow(10), 256 * 2_usize.pow(20), Box::new(gen))
        }
        ArgDataSet::Test => {
            let gen = |pk_rel: &mut [_], fk_rel: &mut [_]| {
                datagen::relation::UniformRelation::gen_primary_key(pk_rel)?;
                datagen::relation::UniformRelation::gen_foreign_key_from_primary_key(
                    fk_rel, pk_rel,
                );
                Ok(())
            };

            (1000, 1000, Box::new(gen))
        }
        ArgDataSet::Lutz2Gv32G => {
            let gen = |pk_rel: &mut [_], fk_rel: &mut [_]| {
                datagen::relation::UniformRelation::gen_primary_key_par(pk_rel)?;
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
            let gen = |pk_rel: &mut [_], fk_rel: &mut [_]| {
                datagen::relation::UniformRelation::gen_primary_key_par(pk_rel)?;
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
            let gen = |pk_rel: &mut [_], fk_rel: &mut [_]| {
                datagen::relation::UniformRelation::gen_primary_key_par(pk_rel)?;
                datagen::relation::UniformRelation::gen_attr_par(fk_rel, 1..=pk_rel.len())?;
                Ok(())
            };

            (
                inner_rel_tuples.expect(
                    "Couldn't find inner relation size. Did you specify --inner-rel-tuples?",
                ),
                outer_rel_tuples.expect(
                    "Couldn't find outer relation size. Did you specify --outer-rel-tuples?",
                ),
                Box::new(gen),
            )
        }
    }
}
