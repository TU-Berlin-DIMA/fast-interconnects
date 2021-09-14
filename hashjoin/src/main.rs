/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

mod error;
mod measurement;
mod types;

use crate::error::{ErrorKind, Result};
use crate::measurement::data_point::DataPoint;
use crate::measurement::harness;
use crate::measurement::hash_join_bench::{HashJoinBenchBuilder, HashJoinPoint};
use crate::types::*;
use data_store::join_data::{JoinDataBuilder, JoinDataGenFn};
use datagen::relation::KeyAttribute;
use num_rational::Ratio;
use num_traits::cast::AsPrimitive;
use numa_gpu::runtime::allocator;
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::dispatcher::{MorselSpec, WorkerCpuAffinity};
use numa_gpu::runtime::hw_info::NvidiaDriverInfo;
use numa_gpu::runtime::linux_wrapper;
use numa_gpu::runtime::numa::{self, NodeRatio};
use rustacuda::context::CurrentContext;
use rustacuda::device::DeviceAttribute;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::DeviceCopy;
use rustacuda::prelude::*;
use serde::de::DeserializeOwned;
use sql_ops::join::{no_partitioning_join, HashingScheme, HtEntry};
use std::mem::size_of;
use std::os::raw::c_uint;
use std::path::PathBuf;
use structopt::StructOpt;

fn main() -> Result<()> {
    // Parse commandline arguments
    let mut cmd = CmdOpt::from_args();

    // Initialize CUDA
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(cmd.device_id.into())?;
    let _context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let cache_node = device.numa_node().ok();
    let overflow_node = match device.numa_memory_affinity() {
        Ok(numa_node) => numa_node,
        Err(e) => {
            eprintln!("Warning: {}; Falling back to node = 0", e);
            0
        }
    };
    cmd.set_spill_hash_table(cache_node, overflow_node)?;

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

    /// Memory type with which to allocate hash table.
    //   unified: CUDA Unified memory (default)
    //   numa: NUMA-local memory on node specified with hash-table-location
    #[structopt(
        long = "hash-table-mem-type",
        default_value = "Device",
        possible_values = &ArgMemType::variants(),
        case_insensitive = true
    )]
    hash_table_mem_type: ArgMemType,

    #[structopt(
        long = "hash-table-location",
        default_value = "0",
        require_delimiter = true
    )]
    /// Allocate memory for hash table on NUMA nodes (e.g.: 0,1,2) or GPU (See numactl -H and CUDA device list)
    hash_table_location: Vec<u16>,

    #[structopt(
        long = "hash-table-proportions",
        default_value = "100",
        require_delimiter = true
    )]
    /// Proportions with with the hash table is allocate on multiple nodes in percent (e.g.: 20,60,20)
    hash_table_proportions: Vec<usize>,

    /// Device memory used to cache hash table (upper limit, in MiB) [Default: All device memory]
    #[structopt(
        long,
        conflicts_with = "hash-table-proportions",
        requires = "spill-hash-table"
    )]
    max_hash_table_cache_size: Option<usize>,

    /// Cache the hash table in GPU memory and spill to the nearest CPU memory node
    ///
    /// This option only works with NVLink 2.0, and sets `--hash-table-mem-type DistributedNuma`
    #[structopt(long)]
    spill_hash_table: Option<bool>,

    #[structopt(long = "inner-rel-location", default_value = "0")]
    /// Allocate memory for inner relation on CPU or GPU (See numactl -H and CUDA device list)
    inner_rel_location: u16,

    #[structopt(long = "outer-rel-location", default_value = "0")]
    /// Allocate memory for outer relation on CPU or GPU (See numactl -H and CUDA device list)
    outer_rel_location: u16,

    /// Page type with with to allocate memory
    #[structopt(
        long = "page-type",
        default_value = "Default",
        possible_values = &ArgPageType::variants(),
        case_insensitive = true
    )]
    page_type: ArgPageType,

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

    /// Execute on device(s) with in-place or streaming-transfer method.
    #[structopt(
        long = "execution-method",
        default_value = "CPU",
        possible_values = &ArgExecutionMethod::variants(),
        case_insensitive = true
    )]
    execution_method: ArgExecutionMethod,

    /// Stream data to device using the transfer strategy.
    #[structopt(
        long = "transfer-strategy",
        default_value = "PageableCopy",
        possible_values = &ArgTransferStrategy::variants(),
        case_insensitive = true
    )]
    transfer_strategy: ArgTransferStrategy,

    #[structopt(long = "cpu-morsel-bytes", default_value = "16384")]
    cpu_morsel_bytes: usize,

    #[structopt(long = "gpu-morsel-bytes", default_value = "33554432")]
    gpu_morsel_bytes: usize,

    #[structopt(short = "i", long = "device-id", default_value = "0")]
    /// Execute on GPU (See CUDA device list)
    device_id: u16,

    #[structopt(short = "t", long = "threads", default_value = "1")]
    threads: usize,

    /// Path to CPU affinity map file for CPU workers
    #[structopt(long = "cpu-affinity", parse(from_os_str))]
    cpu_affinity: Option<PathBuf>,

    /// Path to CPU affinity map file for GPU workers
    #[structopt(long = "gpu-affinity", parse(from_os_str))]
    gpu_affinity: Option<PathBuf>,
}

impl CmdOpt {
    fn set_spill_hash_table(
        &mut self,
        cache_location: Option<u16>,
        overflow_location: u16,
    ) -> Result<()> {
        if self.spill_hash_table == Some(true) {
            let cache_location = cache_location.ok_or_else(|| {
                ErrorKind::RuntimeError(
                    "Failed to set the cache NUMA location. Are you using PCI-e?".to_string(),
                )
            })?;

            self.hash_table_mem_type = ArgMemType::DistributedNuma;
            self.hash_table_location = vec![cache_location, overflow_location];
            self.hash_table_proportions = vec![0, 0];
        }

        Ok(())
    }
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
) -> Result<(Box<dyn FnMut() -> Result<HashJoinPoint>>, DataPoint)>
where
    T: Default
        + AsPrimitive<c_uint>
        + Copy
        + DeviceCopy
        + Sync
        + Send
        + KeyAttribute
        + no_partitioning_join::CudaHashJoinable
        + no_partitioning_join::CpuHashJoinable
        + num_traits::FromPrimitive
        + DeserializeOwned,
{
    // Convert ArgHashingScheme to HashingScheme
    let (hashing_scheme, hash_table_load_factor) = match cmd.hashing_scheme {
        ArgHashingScheme::Perfect => (HashingScheme::Perfect, 1),
        ArgHashingScheme::LinearProbing => (HashingScheme::LinearProbing, 2),
    };

    // Device tuning
    let multiprocessors = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
    let warp_size = device.get_attribute(DeviceAttribute::WarpSize)? as u32;
    let warp_overcommit_factor = 4;
    let grid_overcommit_factor = 2;

    let block_size = BlockSize::x(warp_size * warp_overcommit_factor);
    let grid_size = GridSize::x(multiprocessors * grid_overcommit_factor);

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

    let mut data_builder = JoinDataBuilder::default();
    data_builder
        .mlock(true)
        .inner_mem_type(
            ArgMemTypeHelper {
                mem_type: cmd.mem_type,
                node_ratios: Box::new([NodeRatio {
                    node: cmd.inner_rel_location,
                    ratio: Ratio::from_integer(1),
                }]),
                page_type: cmd.page_type,
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
                page_type: cmd.page_type,
            }
            .into(),
        );

    // Select the operator to run, depending on the device type
    let exec_method = cmd.execution_method.clone();
    let transfer_strategy = cmd.transfer_strategy.clone();
    let mem_type = cmd.hash_table_mem_type;
    let spill_hash_table = cmd.spill_hash_table;
    let max_hash_table_cache_bytes = cmd.max_hash_table_cache_size.map(|s| s * 1024 * 1024); // convert MiB to bytes
    let threads = cmd.threads.clone();
    let device_id = cmd.device_id;
    let page_type = cmd.page_type;

    let morsel_spec = MorselSpec {
        cpu_morsel_bytes: cmd.cpu_morsel_bytes,
        gpu_morsel_bytes: cmd.gpu_morsel_bytes,
    };

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

    let mut hjb_builder = HashJoinBenchBuilder::default();
    let hjb = hjb_builder
        .hashing_scheme(hashing_scheme)
        .is_selective(cmd.selectivity != 100)
        .hash_table_load_factor(hash_table_load_factor)
        .build(join_data.build_relation_key.len())?;

    // Construct data point template for CSV
    let dp = DataPoint::new()?
        .fill_from_cmd_options(cmd)?
        .fill_from_join_data(&join_data)
        .fill_from_hash_join_bench(&hjb)
        .set_init_time(malloc_time, data_gen_time);

    let worker_cpu_affinity = {
        let cpu_workers = if let Some(ref cpu_affinity_file) = cmd.cpu_affinity {
            CpuAffinity::from_file(cpu_affinity_file.as_path())?
        } else {
            CpuAffinity::default()
        };
        let gpu_workers = if let Some(ref gpu_affinity_file) = cmd.gpu_affinity {
            CpuAffinity::from_file(gpu_affinity_file.as_path())?
        } else {
            CpuAffinity::default()
        };
        WorkerCpuAffinity {
            cpu_workers,
            gpu_workers,
        }
    };

    // Bind main thread to the CPU node closest to the GPU. This improves NVLink latency.
    match exec_method {
        ArgExecutionMethod::Gpu
        | ArgExecutionMethod::GpuStream
        | ArgExecutionMethod::Het
        | ArgExecutionMethod::GpuBuildHetProbe => {
            let device = CurrentContext::get_device()?;
            if let Ok(local_cpu_node) = device.numa_memory_affinity() {
                linux_wrapper::numa_run_on_node(local_cpu_node).expect(&format!(
                    "Failed to bind main thread to CPU node {}",
                    local_cpu_node
                ));
            } else {
                eprintln!(
                    "Warning: Couldn't bind main thread to the CPU closest to GPU {}. This may
                        cause additional latency in measurements.",
                    device_id
                );
            }
        }
        _ => {}
    };

    // Create closure that wraps a hash join benchmark function
    let hjc: Box<dyn FnMut() -> Result<HashJoinPoint>> = match exec_method {
        ArgExecutionMethod::Cpu => Box::new(move || {
            let ht_alloc = allocator::Allocator::deref_mem_alloc_fn::<HtEntry<T, T>>(
                ArgMemTypeHelper {
                    mem_type,
                    node_ratios: node_ratios.clone(),
                    page_type,
                }
                .into(),
            );
            hjb.cpu_hash_join(
                &mut join_data,
                threads,
                &worker_cpu_affinity.cpu_workers,
                ht_alloc,
            )
        }),
        ArgExecutionMethod::Gpu => Box::new(move || {
            let (cache_and_spill, cache_node) = if spill_hash_table == Some(true) {
                let (cache_node, spill_node) = if let [cache_node, spill_node] = *node_ratios {
                    Ok((cache_node.node, spill_node.node))
                } else {
                    Err(ErrorKind::InvalidArgument(
                        "Hash table memory type must define exactly two NUMA nodes".to_string(),
                    ))
                }?;

                let cache_spill_type = allocator::CacheSpillType::CacheAndSpill {
                    cache_node,
                    spill_node,
                    page_type: page_type.into(),
                };

                (cache_spill_type, cache_node)
            } else {
                let cache_spill_type: allocator::CacheSpillType =
                    allocator::MemType::from(ArgMemTypeHelper {
                        mem_type,
                        node_ratios: node_ratios.clone(),
                        page_type,
                    })
                    .into();
                let cache_node: u16 = 0;

                (cache_spill_type, cache_node)
            };

            let (ht_alloc_fn, cache_bytes_future) =
                allocator::Allocator::mem_spill_alloc_fn::<HtEntry<T, T>>(cache_and_spill);

            hjb.cuda_hash_join(
                &mut join_data,
                ht_alloc_fn,
                cache_node,
                max_hash_table_cache_bytes,
                cache_bytes_future,
                (grid_size.clone(), block_size.clone()),
                (grid_size.clone(), block_size.clone()),
            )
        }),
        ArgExecutionMethod::GpuStream if transfer_strategy == ArgTransferStrategy::Unified => {
            Box::new(move || {
                let ht_alloc = allocator::Allocator::mem_alloc_fn::<HtEntry<T, T>>(
                    ArgMemTypeHelper {
                        mem_type,
                        node_ratios: node_ratios.clone(),
                        page_type,
                    }
                    .into(),
                );
                hjb.cuda_streaming_unified_hash_join(
                    &mut join_data,
                    ht_alloc,
                    (grid_size.clone(), block_size.clone()),
                    (grid_size.clone(), block_size.clone()),
                    morsel_spec.gpu_morsel_bytes,
                )
            })
        }
        ArgExecutionMethod::GpuStream => Box::new(move || {
            let ht_alloc = allocator::Allocator::mem_alloc_fn::<HtEntry<T, T>>(
                ArgMemTypeHelper {
                    mem_type,
                    node_ratios: node_ratios.clone(),
                    page_type,
                }
                .into(),
            );
            hjb.cuda_streaming_hash_join(
                &mut join_data,
                ht_alloc,
                (grid_size.clone(), block_size.clone()),
                (grid_size.clone(), block_size.clone()),
                transfer_strategy.into(),
                morsel_spec.gpu_morsel_bytes,
                threads,
                &worker_cpu_affinity.cpu_workers,
            )
        }),
        ArgExecutionMethod::Het => Box::new(move || {
            let ht_alloc = allocator::Allocator::mem_alloc_fn::<HtEntry<T, T>>(
                ArgMemTypeHelper {
                    mem_type,
                    node_ratios: node_ratios.clone(),
                    page_type,
                }
                .into(),
            );
            hjb.hetrogeneous_hash_join(
                &mut join_data,
                ht_alloc,
                threads,
                &worker_cpu_affinity,
                vec![device_id],
                (grid_size.clone(), block_size.clone()),
                (grid_size.clone(), block_size.clone()),
                &morsel_spec,
            )
        }),
        ArgExecutionMethod::GpuBuildHetProbe => Box::new(move || {
            // Allocate CPU memory on NUMA node of thread 0
            let cpu_node = numa::node_of_cpu(
                worker_cpu_affinity
                    .cpu_workers
                    .thread_to_cpu(0)
                    .expect("Couldn't map thread to a core"),
            )?;
            let cpu_ht_alloc = allocator::Allocator::mem_alloc_fn::<HtEntry<T, T>>(
                ArgMemTypeHelper {
                    mem_type: ArgMemType::Numa,
                    node_ratios: Box::new([NodeRatio {
                        node: cpu_node,
                        ratio: Ratio::from_integer(0),
                    }]),
                    page_type,
                }
                .into(),
            );

            // Allocate GPU memory as specified on the commandline
            let gpu_ht_alloc = allocator::Allocator::mem_alloc_fn::<HtEntry<T, T>>(
                ArgMemTypeHelper {
                    mem_type,
                    node_ratios: node_ratios.clone(),
                    page_type,
                }
                .into(),
            );

            hjb.gpu_build_heterogeneous_probe(
                &mut join_data,
                cpu_ht_alloc,
                gpu_ht_alloc,
                threads,
                &worker_cpu_affinity,
                vec![device_id],
                (grid_size.clone(), block_size.clone()),
                (grid_size.clone(), block_size.clone()),
                &morsel_spec,
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
            Box::new(move |pk_rel, _, fk_rel, _| {
                datagen::popular::Blanas::gen(pk_rel, fk_rel, selectivity).map_err(|e| e.into())
            }),
        ),
        ArgDataSet::Kim => (
            datagen::popular::Kim::primary_key_len(),
            datagen::popular::Kim::foreign_key_len(),
            Box::new(move |pk_rel, _, fk_rel, _| {
                datagen::popular::Kim::gen(pk_rel, fk_rel, selectivity).map_err(|e| e.into())
            }),
        ),
        ArgDataSet::Blanas4MB => {
            let gen = move |pk_rel: &mut [_], _: &mut [_], fk_rel: &mut [_], _: &mut [_]| {
                datagen::relation::UniformRelation::gen_primary_key_par(pk_rel, selectivity)?;
                datagen::relation::UniformRelation::gen_attr_par(fk_rel, 0..pk_rel.len())?;
                Ok(())
            };

            (512 * 2_usize.pow(10), 256 * 2_usize.pow(20), Box::new(gen))
        }
        ArgDataSet::Test => {
            let gen = move |pk_rel: &mut [_], _: &mut [_], fk_rel: &mut [_], _: &mut [_]| {
                datagen::relation::UniformRelation::gen_primary_key(pk_rel, selectivity)?;
                datagen::relation::UniformRelation::gen_foreign_key_from_primary_key(
                    fk_rel, pk_rel,
                );
                Ok(())
            };

            (1000, 1000, Box::new(gen))
        }
        ArgDataSet::Lutz2Gv32G => {
            let gen = move |pk_rel: &mut [_], _: &mut [_], fk_rel: &mut [_], _: &mut [_]| {
                datagen::relation::UniformRelation::gen_primary_key_par(pk_rel, selectivity)?;
                datagen::relation::UniformRelation::gen_attr_par(fk_rel, 0..pk_rel.len())?;
                Ok(())
            };

            (
                2 * 2_usize.pow(30) / (2 * size_of::<T>()),
                32 * 2_usize.pow(30) / (2 * size_of::<T>()),
                Box::new(gen),
            )
        }
        ArgDataSet::Lutz32Gv32G => {
            let gen = move |pk_rel: &mut [_], _: &mut [_], fk_rel: &mut [_], _: &mut [_]| {
                datagen::relation::UniformRelation::gen_primary_key_par(pk_rel, selectivity)?;
                datagen::relation::UniformRelation::gen_attr_par(fk_rel, 0..pk_rel.len())?;
                Ok(())
            };

            (
                32 * 2_usize.pow(30) / (2 * size_of::<T>()),
                32 * 2_usize.pow(30) / (2 * size_of::<T>()),
                Box::new(gen),
            )
        }
        ArgDataSet::Custom => {
            let uniform_gen = Box::new(
                move |pk_rel: &mut [_], _: &mut [_], fk_rel: &mut [_], _: &mut [_]| {
                    datagen::relation::UniformRelation::gen_primary_key_par(pk_rel, selectivity)?;
                    datagen::relation::UniformRelation::gen_attr_par(fk_rel, 0..pk_rel.len())?;
                    Ok(())
                },
            );

            let gen: JoinDataGenFn<T> = match data_distribution {
                DataDistribution::Uniform => uniform_gen,
                DataDistribution::Zipf(exp) if !(exp > 0.0) => uniform_gen,
                DataDistribution::Zipf(exp) => Box::new(
                    move |pk_rel: &mut [_], _: &mut [_], fk_rel: &mut [_], _: &mut [_]| {
                        datagen::relation::UniformRelation::gen_primary_key_par(
                            pk_rel,
                            selectivity,
                        )?;
                        datagen::relation::ZipfRelation::gen_attr_par(fk_rel, pk_rel.len(), exp)?;
                        Ok(())
                    },
                ),
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
