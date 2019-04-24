/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

#[macro_use]
extern crate average;
#[macro_use]
extern crate clap;
extern crate core; // Required by average::concatenate!{} macro
extern crate csv;
#[macro_use]
extern crate error_chain;
extern crate hostname;
extern crate num_traits;
extern crate numa_gpu;
extern crate rayon;
#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_repr;
extern crate structopt;

use average::{Estimate, Max, Min, Quantile, Variance};

use numa_gpu::datagen;
use numa_gpu::error::{ErrorKind, Result};
use numa_gpu::operators::hash_join;
use numa_gpu::runtime::allocator;
use numa_gpu::runtime::cuda::{
    CudaTransferStrategy, IntoCudaIterator, IntoCudaIteratorWithStrategy,
};
use numa_gpu::runtime::cuda_wrapper::prefetch_async;
use numa_gpu::runtime::hw_info::{cpu_codename, CudaDeviceInfo};
use numa_gpu::runtime::memory::*;
use numa_gpu::runtime::utils::EnsurePhysicallyBacked;

use rustacuda::device::DeviceAttribute;
use rustacuda::event::{Event, EventFlags};
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::DeviceCopy;
use rustacuda::prelude::*;

use serde_repr::Serialize_repr;

use std::collections::vec_deque::VecDeque;
use std::mem::size_of;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use structopt::StructOpt;

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq)]
    enum ArgDataSet {
        Blanas,
        Kim,
        Test,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgMemType {
        System,
        Numa,
        NumaLazyPinned,
        Pinned,
        Unified,
        Device,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgExecutionMethod {
        Cpu,
        Gpu,
        GpuStream,
        Het,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgTransferStrategy {
        PageableCopy,
        PinnedCopy,
        LazyPinnedCopy,
        Unified,
        Coherence,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgHashingScheme {
        Perfect,
        LinearProbing,
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

#[derive(Debug)]
pub struct ArgMemTypeHelper {
    mem_type: ArgMemType,
    location: u16,
}

impl From<ArgMemTypeHelper> for allocator::MemType {
    fn from(ArgMemTypeHelper { mem_type, location }: ArgMemTypeHelper) -> Self {
        match mem_type {
            ArgMemType::System => allocator::MemType::SysMem,
            ArgMemType::Numa => allocator::MemType::NumaMem(location),
            ArgMemType::NumaLazyPinned => allocator::MemType::NumaMem(location),
            ArgMemType::Pinned => allocator::MemType::CudaPinnedMem,
            ArgMemType::Unified => allocator::MemType::CudaUniMem,
            ArgMemType::Device => allocator::MemType::CudaDevMem,
        }
    }
}

impl From<ArgMemTypeHelper> for allocator::DerefMemType {
    fn from(ArgMemTypeHelper { mem_type, location }: ArgMemTypeHelper) -> Self {
        match mem_type {
            ArgMemType::System => allocator::DerefMemType::SysMem,
            ArgMemType::Numa => allocator::DerefMemType::NumaMem(location),
            ArgMemType::NumaLazyPinned => allocator::DerefMemType::NumaMem(location),
            ArgMemType::Pinned => allocator::DerefMemType::CudaPinnedMem,
            ArgMemType::Unified => allocator::DerefMemType::CudaUniMem,
            ArgMemType::Device => panic!("Error: Device memory not supported in this context!"),
        }
    }
}

impl From<ArgTransferStrategy> for CudaTransferStrategy {
    fn from(asm: ArgTransferStrategy) -> Self {
        match asm {
            ArgTransferStrategy::PageableCopy => CudaTransferStrategy::PageableCopy,
            ArgTransferStrategy::PinnedCopy => CudaTransferStrategy::PinnedCopy,
            ArgTransferStrategy::LazyPinnedCopy => CudaTransferStrategy::LazyPinnedCopy,
            ArgTransferStrategy::Unified => {
                panic!("Error: Unified memory cannot be handled by CudaTransferStrategy!")
            }
            ArgTransferStrategy::Coherence => CudaTransferStrategy::Coherence,
        }
    }
}

impl From<ArgHashingScheme> for hash_join::HashingScheme {
    fn from(ahs: ArgHashingScheme) -> Self {
        match ahs {
            ArgHashingScheme::Perfect => hash_join::HashingScheme::Perfect,
            ArgHashingScheme::LinearProbing => hash_join::HashingScheme::LinearProbing,
        }
    }
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

    #[structopt(long = "hash-table-location", default_value = "0")]
    /// Allocate memory for hash table on CPU or GPU (See numactl -H and CUDA device list)
    hash_table_location: u16,

    #[structopt(long = "inner-rel-location", default_value = "0")]
    /// Allocate memory for inner relation on CPU or GPU (See numactl -H and CUDA device list)
    inner_rel_location: u16,

    #[structopt(long = "outer-rel-location", default_value = "0")]
    /// Allocate memory for outer relation on CPU or GPU (See numactl -H and CUDA device list)
    outer_rel_location: u16,

    /// Use a pre-defined data set.
    //   blanas: Blanas et al. "Main memory hash join algorithms for multi-core CPUs"
    //   kim: Kim et al. "Sort vs. hash revisited"
    //   test: A small data set for testing on the laptop
    #[structopt(
        short = "s",
        long = "data-set",
        default_value = "Test",
        raw(possible_values = "&ArgDataSet::variants()", case_insensitive = "true")
    )]
    data_set: ArgDataSet,

    #[structopt(
        long = "tuple-bytes",
        default_value = "Bytes8",
        raw(
            possible_values = "&ArgTupleBytes::variants()",
            case_insensitive = "true"
        )
    )]
    tuple_bytes: ArgTupleBytes,

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

#[derive(Clone, Debug, Default, Serialize)]
pub struct DataPoint {
    pub hostname: String,
    pub execution_method: Option<ArgExecutionMethod>,
    pub device_codename: Option<String>,
    pub transfer_strategy: Option<ArgTransferStrategy>,
    pub chunk_bytes: Option<usize>,
    pub threads: Option<usize>,
    pub hashing_scheme: Option<ArgHashingScheme>,
    pub hash_table_memory_type: Option<ArgMemType>,
    pub hash_table_memory_node: Option<u16>,
    pub hash_table_bytes: Option<usize>,
    pub tuple_bytes: Option<ArgTupleBytes>,
    pub relation_memory_type: Option<ArgMemType>,
    pub inner_relation_memory_location: Option<u16>,
    pub outer_relation_memory_location: Option<u16>,
    pub build_tuples: Option<usize>,
    pub build_bytes: Option<usize>,
    pub probe_tuples: Option<usize>,
    pub probe_bytes: Option<usize>,
    pub warm_up: Option<bool>,
    pub build_ns: Option<f64>,
    pub probe_ns: Option<f64>,
}

impl DataPoint {
    fn new() -> Result<DataPoint> {
        let hostname = hostname::get_hostname().ok_or_else(|| "Couldn't get hostname")?;

        let dp = DataPoint {
            hostname,
            ..DataPoint::default()
        };

        Ok(dp)
    }

    fn fill_from_cmd_options(&self, cmd: &CmdOpt) -> Result<DataPoint> {
        // Get device information
        let dev_codename_str = match cmd.execution_method {
            ArgExecutionMethod::Cpu => cpu_codename(),
            ArgExecutionMethod::Gpu | ArgExecutionMethod::GpuStream => {
                let device = Device::get_device(cmd.device_id.into())?;
                device.name()?
            }
            ArgExecutionMethod::Het => unimplemented!(),
        };

        let dp = DataPoint {
            execution_method: Some(cmd.execution_method),
            device_codename: Some(dev_codename_str),
            transfer_strategy: Some(cmd.transfer_strategy),
            chunk_bytes: Some(cmd.chunk_bytes),
            threads: if cmd.execution_method == ArgExecutionMethod::Cpu {
                Some(cmd.threads)
            } else {
                None
            },
            hashing_scheme: Some(cmd.hashing_scheme),
            hash_table_memory_type: Some(cmd.hash_table_mem_type),
            hash_table_memory_node: Some(cmd.hash_table_location),
            tuple_bytes: Some(cmd.tuple_bytes),
            relation_memory_type: Some(cmd.mem_type),
            inner_relation_memory_location: Some(cmd.inner_rel_location),
            outer_relation_memory_location: Some(cmd.outer_rel_location),
            ..self.clone()
        };

        Ok(dp)
    }

    fn fill_from_hash_join_bench<T: DeviceCopy>(&self, hjb: &HashJoinBench<T>) -> DataPoint {
        DataPoint {
            hash_table_bytes: Some(hjb.hash_table_len * size_of::<T>()),
            build_tuples: Some(hjb.build_relation_key.len()),
            build_bytes: Some(hjb.build_relation_key.len() * size_of::<T>()),
            probe_tuples: Some(hjb.probe_relation_key.len()),
            probe_bytes: Some(hjb.probe_relation_key.len() * size_of::<T>()),
            ..self.clone()
        }
    }
}

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
            measure("hash_join_kim", cmd.repeat, cmd.csv, dp, hjc)?;
        }
        ArgTupleBytes::Bytes16 => {
            let (hjc, dp) = args_to_bench::<i64>(&cmd, device)?;
            measure("hash_join_kim", cmd.repeat, cmd.csv, dp, hjc)?;
        }
    };

    Ok(())
}

fn args_to_bench<T>(
    cmd: &CmdOpt,
    device: Device,
) -> Result<(Box<FnMut() -> Result<(f64, f64)>>, DataPoint)>
where
    T: Default
        + DeviceCopy
        + Sync
        + Send
        + hash_join::NullKey
        + hash_join::CudaHashJoinable
        + hash_join::CpuHashJoinable
        + EnsurePhysicallyBacked
        + num_traits::FromPrimitive,
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

    if cmd.execution_method == ArgExecutionMethod::GpuStream {
        assert!(
            cmd.mem_type == ArgMemType::System || cmd.mem_type == ArgMemType::Numa || cmd.mem_type == ArgMemType::Unified,
            "Invalid memory type. Streaming execution method must be either System, Numa, or Unified."
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
        .inner_location(cmd.inner_rel_location)
        .outer_location(cmd.outer_rel_location)
        .inner_mem_type(cmd.mem_type)
        .outer_mem_type(cmd.mem_type);

    // Select the operator to run, depending on the device type
    let exec_method = cmd.execution_method.clone();
    let transfer_strategy = cmd.transfer_strategy.clone();
    let chunk_len = cmd.chunk_bytes / size_of::<T>();
    let mem_type = cmd.hash_table_mem_type;
    let location = cmd.hash_table_location;
    let threads = cmd.threads.clone();

    // Select data set
    let (inner_relation_len, outer_relation_len, data_gen) = data_gen_fn::<_>(cmd.data_set);
    let mut hjb = hjb_builder
        .inner_len(inner_relation_len)
        .outer_len(outer_relation_len)
        .build_with_data_gen(data_gen)?;

    // Construct data point template for CSV
    let dp = DataPoint::new()?
        .fill_from_cmd_options(cmd)?
        .fill_from_hash_join_bench(&hjb);

    // Create closure that wraps a hash join benchmark function
    let hjc: Box<FnMut() -> Result<(f64, f64)>> = match exec_method {
        ArgExecutionMethod::Cpu => Box::new(move || {
            let ht_alloc = allocator::Allocator::deref_mem_alloc_fn::<T>(
                ArgMemTypeHelper { mem_type, location }.into(),
            );
            hjb.cpu_hash_join(threads, ht_alloc)
        }),
        ArgExecutionMethod::Gpu => Box::new(move || {
            let ht_alloc = allocator::Allocator::mem_alloc_fn::<T>(
                ArgMemTypeHelper { mem_type, location }.into(),
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
                    ArgMemTypeHelper { mem_type, location }.into(),
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
                ArgMemTypeHelper { mem_type, location }.into(),
            );
            hjb.cuda_streaming_hash_join(
                ht_alloc,
                (grid_size.clone(), block_size.clone()),
                (grid_size.clone(), block_size.clone()),
                transfer_strategy.into(),
                chunk_len,
            )
        }),
        ArgExecutionMethod::Het => unimplemented!(),
    };

    Ok((hjc, dp))
}

type DataGenFn<T> = Box<FnMut(&mut [T], &mut [T]) -> Result<()>>;

fn data_gen_fn<T>(description: ArgDataSet) -> (usize, usize, DataGenFn<T>)
where
    T: Copy + num_traits::FromPrimitive,
{
    match description {
        ArgDataSet::Blanas => (
            datagen::popular::Blanas::primary_key_len(),
            datagen::popular::Blanas::foreign_key_len(),
            Box::new(|pk_rel, fk_rel| datagen::popular::Blanas::gen(pk_rel, fk_rel)),
        ),
        ArgDataSet::Kim => (
            datagen::popular::Kim::primary_key_len(),
            datagen::popular::Kim::foreign_key_len(),
            Box::new(|pk_rel, fk_rel| datagen::popular::Kim::gen(pk_rel, fk_rel)),
        ),
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
    }
}

fn measure(
    name: &str,
    repeat: u32,
    out_file_name: Option<PathBuf>,
    template: DataPoint,
    mut func: Box<FnMut() -> Result<(f64, f64)>>,
) -> Result<()> {
    let measurements = (0..repeat)
        .map(|_| {
            func().map(|(build_ns, probe_ns)| DataPoint {
                warm_up: Some(false),
                build_ns: Some(build_ns),
                probe_ns: Some(probe_ns),
                ..template.clone()
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if let Some(ofn) = out_file_name {
        let csv_file = std::fs::File::create(ofn)?;
        let mut csv = csv::Writer::from_writer(csv_file);
        ensure!(
            measurements
                .iter()
                .try_for_each(|row| csv.serialize(row))
                .is_ok(),
            "Couldn't write serialized measurements"
        );
    }

    concatenate!(
        Estimator,
        [Variance, variance, mean, error],
        [Quantile, quantile, quantile],
        [Min, min, min],
        [Max, max, max]
    );

    let time_stats: Estimator = measurements
        .iter()
        .filter_map(|row| row.probe_ns)
        .map(|probe_ns| probe_ns / 10_f64.powf(6.0))
        .collect();

    let tput_stats: Estimator = measurements
        .iter()
        .filter_map(|row| {
            row.probe_bytes
                .and_then(|bytes| row.probe_ns.and_then(|ns| Some((bytes, ns))))
        })
        .map(|(probe_bytes, probe_ns)| (probe_bytes as f64, probe_ns))
        .map(|(bytes, ms)| bytes / ms / 2.0_f64.powf(30.0) * 10.0_f64.powf(9.0))
        .collect();

    println!(
        r#"Bench: {}
Sample size: {}
               Time            Throughput
                ms              GiB/s
Mean:          {:6.2}          {:6.2}
Stddev:        {:6.2}          {:6.2}
Median:        {:6.2}          {:6.2}
Min:           {:6.2}          {:6.2}
Max:           {:6.2}          {:6.2}"#,
        name.replace("_", " "),
        measurements.len(),
        time_stats.mean(),
        tput_stats.mean(),
        time_stats.error(),
        tput_stats.error(),
        time_stats.quantile(),
        tput_stats.quantile(),
        time_stats.min(),
        tput_stats.min(),
        time_stats.max(),
        tput_stats.max(),
    );

    Ok(())
}

struct HashJoinBench<T: DeviceCopy> {
    hashing_scheme: hash_join::HashingScheme,
    hash_table_len: usize,
    build_relation_key: Mem<T>,
    build_relation_payload: Mem<T>,
    probe_relation_key: Mem<T>,
    probe_relation_payload: Mem<T>,
}

struct HashJoinBenchBuilder {
    hash_table_load_factor: usize,
    hash_table_elems_per_entry: usize,
    inner_len: usize,
    outer_len: usize,
    inner_location: u16,
    outer_location: u16,
    inner_mem_type: ArgMemType,
    outer_mem_type: ArgMemType,
    hashing_scheme: hash_join::HashingScheme,
}

impl Default for HashJoinBenchBuilder {
    fn default() -> HashJoinBenchBuilder {
        HashJoinBenchBuilder {
            hash_table_load_factor: 2,
            hash_table_elems_per_entry: 2, // FIXME: replace constant with an HtEntry type
            inner_len: 1,
            outer_len: 1,
            inner_location: 0,
            outer_location: 0,
            inner_mem_type: ArgMemType::System,
            outer_mem_type: ArgMemType::System,
            hashing_scheme: hash_join::HashingScheme::LinearProbing,
        }
    }
}

impl HashJoinBenchBuilder {
    fn hash_table_load_factor(&mut self, hash_table_load_factor: usize) -> &mut Self {
        self.hash_table_load_factor = hash_table_load_factor;
        self
    }

    fn inner_len(&mut self, inner_len: usize) -> &mut Self {
        self.inner_len = inner_len;
        self
    }

    fn outer_len(&mut self, outer_len: usize) -> &mut Self {
        self.outer_len = outer_len;
        self
    }

    fn inner_location(&mut self, inner_location: u16) -> &mut Self {
        self.inner_location = inner_location;
        self
    }

    fn outer_location(&mut self, outer_location: u16) -> &mut Self {
        self.outer_location = outer_location;
        self
    }

    fn inner_mem_type(&mut self, inner_mem_type: ArgMemType) -> &mut Self {
        self.inner_mem_type = inner_mem_type;
        self
    }

    fn outer_mem_type(&mut self, outer_mem_type: ArgMemType) -> &mut Self {
        self.outer_mem_type = outer_mem_type;
        self
    }

    fn hashing_scheme(&mut self, hashing_scheme: hash_join::HashingScheme) -> &mut Self {
        self.hashing_scheme = hashing_scheme;
        self
    }

    fn build_with_data_gen<T: Copy + Default + DeviceCopy>(
        &mut self,
        mut data_gen_fn: DataGenFn<T>,
    ) -> Result<HashJoinBench<T>> {
        // Allocate memory for data sets
        let mut memory: VecDeque<_> = [
            (self.inner_len, self.inner_mem_type, self.inner_location),
            (self.inner_len, self.inner_mem_type, self.inner_location),
            (self.outer_len, self.outer_mem_type, self.outer_location),
            (self.outer_len, self.outer_mem_type, self.outer_location),
        ]
        .iter()
        .map(|&(len, mem_type, location)| {
            let mut mem = allocator::Allocator::alloc_deref_mem(
                ArgMemTypeHelper { mem_type, location }.into(),
                len,
            );
            match (mem_type, &mut mem) {
                (ArgMemType::NumaLazyPinned, DerefMem::NumaMem(lazy_pinned_mem)) => lazy_pinned_mem
                    .page_lock()
                    .expect("Failed to lazily pin memory"),
                _ => {}
            };
            mem
        })
        .collect();

        let mut inner_key = memory.pop_front().ok_or_else(|| {
            ErrorKind::LogicError(
                "Failed to get primary key relation. Is it allocated?".to_string(),
            )
        })?;
        let inner_payload = memory.pop_front().ok_or_else(|| {
            ErrorKind::LogicError(
                "Failed to get primary key relation. Is it allocated?".to_string(),
            )
        })?;
        let mut outer_key = memory.pop_front().ok_or_else(|| {
            ErrorKind::LogicError(
                "Failed to get foreign key relation. Is it allocated?".to_string(),
            )
        })?;
        let outer_payload = memory.pop_front().ok_or_else(|| {
            ErrorKind::LogicError(
                "Failed to get foreign key relation. Is it allocated?".to_string(),
            )
        })?;

        // Generate dataset
        data_gen_fn(inner_key.as_mut_slice(), outer_key.as_mut_slice())?;

        // Calculate hash table length
        let hash_table_len = self
            .inner_len
            .checked_next_power_of_two()
            .and_then(|x| {
                x.checked_mul(self.hash_table_load_factor * self.hash_table_elems_per_entry)
            })
            .ok_or_else(|| {
                ErrorKind::IntegerOverflow("Failed to compute hash table length".to_string())
            })?;

        Ok(HashJoinBench {
            hashing_scheme: self.hashing_scheme,
            hash_table_len: hash_table_len,
            build_relation_key: inner_key.into(),
            build_relation_payload: inner_payload.into(),
            probe_relation_key: outer_key.into(),
            probe_relation_payload: outer_payload.into(),
        })
    }
}

impl<T> HashJoinBench<T>
where
    T: Default
        + DeviceCopy
        + Sync
        + Send
        + hash_join::NullKey
        + hash_join::CudaHashJoinable
        + hash_join::CpuHashJoinable
        + EnsurePhysicallyBacked,
{
    fn cuda_hash_join(
        &mut self,
        hash_table_alloc: allocator::MemAllocFn<T>,
        build_dim: (GridSize, BlockSize),
        probe_dim: (GridSize, BlockSize),
    ) -> Result<(f64, f64)> {
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // FIXME: specify load factor as argument
        let hash_table_mem = hash_table_alloc(self.hash_table_len);
        let hash_table = hash_join::HashTable::new_on_gpu(hash_table_mem, self.hash_table_len)?;
        let mut result_counts = allocator::Allocator::alloc_mem(
            allocator::MemType::CudaUniMem,
            (probe_dim.0.x * probe_dim.1.x) as usize,
        );

        // Initialize counts
        if let CudaUniMem(ref mut c) = result_counts {
            c.iter_mut().map(|count| *count = 0).for_each(drop);
        }

        // Tune memory locations
        [
            &mut self.build_relation_key,
            &mut self.probe_relation_key,
            &mut self.build_relation_payload,
            &mut self.probe_relation_payload,
        ]
        .iter_mut()
        .filter_map(|mem| {
            if let CudaUniMem(m) = mem {
                Some(m)
            } else {
                None
            }
        })
        .map(|mem| prefetch_async(mem.as_unified_ptr(), mem.len(), &stream))
        .collect::<Result<()>>()?;

        stream.synchronize()?;

        let hj_op = hash_join::CudaHashJoinBuilder::<T>::default()
            .hashing_scheme(self.hashing_scheme)
            .build_dim(build_dim.0.clone(), build_dim.1.clone())
            .probe_dim(probe_dim.0.clone(), probe_dim.1.clone())
            .hash_table(hash_table)
            .build()?;

        let start_event = Event::new(EventFlags::DEFAULT)?;
        let stop_event = Event::new(EventFlags::DEFAULT)?;

        start_event.record(&stream)?;
        hj_op.build(
            self.build_relation_key.as_launchable_slice(),
            self.build_relation_payload.as_launchable_slice(),
            &stream,
        )?;

        stop_event.record(&stream)?;
        stop_event.synchronize()?;
        let build_millis = stop_event.elapsed_time_f32(&start_event)?;

        start_event.record(&stream)?;
        hj_op.probe_count(
            self.probe_relation_key.as_launchable_slice(),
            self.probe_relation_payload.as_launchable_slice(),
            &mut result_counts,
            &stream,
        )?;

        stop_event.record(&stream)?;
        stop_event.synchronize()?;
        let probe_millis = stop_event.elapsed_time_f32(&start_event)?;

        stream.synchronize()?;
        Ok((
            build_millis as f64 * 10_f64.powf(6.0),
            probe_millis as f64 * 10_f64.powf(6.0),
        ))
    }

    fn cuda_streaming_hash_join(
        &mut self,
        hash_table_alloc: allocator::MemAllocFn<T>,
        build_dim: (GridSize, BlockSize),
        probe_dim: (GridSize, BlockSize),
        transfer_strategy: CudaTransferStrategy,
        chunk_len: usize,
    ) -> Result<(f64, f64)> {
        let hash_table_mem = hash_table_alloc(self.hash_table_len);
        let hash_table = hash_join::HashTable::new_on_gpu(hash_table_mem, self.hash_table_len)?;
        let mut result_counts = allocator::Allocator::alloc_mem(
            allocator::MemType::CudaUniMem,
            (probe_dim.0.x * probe_dim.1.x) as usize,
        );

        // Initialize counts
        if let CudaUniMem(ref mut c) = result_counts {
            c.iter_mut().map(|count| *count = 0).for_each(drop);
        }

        let build_rel_key = match self.build_relation_key {
            Mem::CudaUniMem(ref mut m) => m.as_mut_slice(),
            Mem::SysMem(ref mut m) => m.as_mut_slice(),
            Mem::NumaMem(ref mut m) => m.as_mut_slice(),
            Mem::CudaPinnedMem(ref mut m) => m.as_mut_slice(),
            Mem::CudaDevMem(_) => panic!("Can't use CUDA device memory on CPU!"),
        };
        let build_rel_pay = match self.build_relation_payload {
            Mem::CudaUniMem(ref mut m) => m.as_mut_slice(),
            Mem::SysMem(ref mut m) => m.as_mut_slice(),
            Mem::NumaMem(ref mut m) => m.as_mut_slice(),
            Mem::CudaPinnedMem(ref mut m) => m.as_mut_slice(),
            Mem::CudaDevMem(_) => panic!("Can't use CUDA device memory on CPU!"),
        };
        let probe_rel_key = match self.probe_relation_key {
            Mem::CudaUniMem(ref mut m) => m.as_mut_slice(),
            Mem::SysMem(ref mut m) => m.as_mut_slice(),
            Mem::NumaMem(ref mut m) => m.as_mut_slice(),
            Mem::CudaPinnedMem(ref mut m) => m.as_mut_slice(),
            Mem::CudaDevMem(_) => panic!("Can't use CUDA device memory on CPU!"),
        };
        let probe_rel_pay = match self.probe_relation_payload {
            Mem::CudaUniMem(ref mut m) => m.as_mut_slice(),
            Mem::SysMem(ref mut m) => m.as_mut_slice(),
            Mem::NumaMem(ref mut m) => m.as_mut_slice(),
            Mem::CudaPinnedMem(ref mut m) => m.as_mut_slice(),
            Mem::CudaDevMem(_) => panic!("Can't use CUDA device memory on CPU!"),
        };

        let hj_op = hash_join::CudaHashJoinBuilder::<T>::default()
            .hashing_scheme(self.hashing_scheme)
            .build_dim(build_dim.0.clone(), build_dim.1.clone())
            .probe_dim(probe_dim.0.clone(), probe_dim.1.clone())
            .hash_table(hash_table)
            .build()?;

        let mut build_relation = (build_rel_key, build_rel_pay);
        let mut probe_relation = (probe_rel_key, probe_rel_pay);

        let mut timer = Instant::now();

        build_relation
            .into_cuda_iter_with_strategy(transfer_strategy, chunk_len)?
            .fold(|(key, val), stream| hj_op.build(key, val, stream))?;

        let mut dur = timer.elapsed();
        let build_nanos = dur.as_secs() * 10_u64.pow(9) + dur.subsec_nanos() as u64;

        timer = Instant::now();

        probe_relation
            .into_cuda_iter_with_strategy(transfer_strategy, chunk_len)?
            .fold(|(key, val), stream| hj_op.probe_count(key, val, &result_counts, stream))?;

        dur = timer.elapsed();
        let probe_nanos = dur.as_secs() * 10_u64.pow(9) + dur.subsec_nanos() as u64;

        Ok((build_nanos as f64, probe_nanos as f64))
    }

    fn cuda_streaming_unified_hash_join(
        &mut self,
        hash_table_alloc: allocator::MemAllocFn<T>,
        build_dim: (GridSize, BlockSize),
        probe_dim: (GridSize, BlockSize),
        chunk_len: usize,
    ) -> Result<(f64, f64)> {
        let hash_table_mem = hash_table_alloc(self.hash_table_len);
        let hash_table = hash_join::HashTable::new_on_gpu(hash_table_mem, self.hash_table_len)?;
        let mut result_counts = allocator::Allocator::alloc_mem(
            allocator::MemType::CudaUniMem,
            (probe_dim.0.x * probe_dim.1.x) as usize,
        );

        // Initialize counts
        if let CudaUniMem(ref mut c) = result_counts {
            c.iter_mut().map(|count| *count = 0).for_each(drop);
        }

        let build_rel_key = match self.build_relation_key {
            Mem::CudaUniMem(ref mut m) => m,
            _ => unreachable!(),
        };
        let build_rel_pay = match self.build_relation_payload {
            Mem::CudaUniMem(ref mut m) => m,
            _ => unreachable!(),
        };
        let probe_rel_key = match self.probe_relation_key {
            Mem::CudaUniMem(ref mut m) => m,
            _ => unreachable!(),
        };
        let probe_rel_pay = match self.probe_relation_payload {
            Mem::CudaUniMem(ref mut m) => m,
            _ => unreachable!(),
        };

        let hj_op = hash_join::CudaHashJoinBuilder::<T>::default()
            .hashing_scheme(self.hashing_scheme)
            .build_dim(build_dim.0.clone(), build_dim.1.clone())
            .probe_dim(probe_dim.0.clone(), probe_dim.1.clone())
            .hash_table(hash_table)
            .build()?;

        let mut build_relation = (build_rel_key, build_rel_pay);
        let mut probe_relation = (probe_rel_key, probe_rel_pay);

        let mut timer = Instant::now();

        build_relation
            .into_cuda_iter(chunk_len)?
            .fold(|(key, val), stream| hj_op.build(key, val, stream))?;

        let mut dur = timer.elapsed();
        let build_nanos = dur.as_secs() * 10_u64.pow(9) + dur.subsec_nanos() as u64;

        timer = Instant::now();

        probe_relation
            .into_cuda_iter(chunk_len)?
            .fold(|(key, val), stream| hj_op.probe_count(key, val, &result_counts, stream))?;

        dur = timer.elapsed();
        let probe_nanos = dur.as_secs() * 10_u64.pow(9) + dur.subsec_nanos() as u64;

        Ok((build_nanos as f64, probe_nanos as f64))
    }

    fn cpu_hash_join(
        &self,
        threads: usize,
        hash_table_alloc: allocator::DerefMemAllocFn<T>,
    ) -> Result<(f64, f64)> {
        let mut hash_table_mem = hash_table_alloc(self.hash_table_len);
        T::ensure_physically_backed(hash_table_mem.as_mut_slice());
        let hash_table = hash_join::HashTable::new_on_cpu(hash_table_mem, self.hash_table_len)?;
        let mut result_counts = vec![0; threads];

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|_| ErrorKind::RuntimeError("Failed to create thread pool".to_string()))?;
        let build_chunk_size = (self.build_relation_key.len() + threads - 1) / threads;
        let probe_chunk_size = (self.probe_relation_key.len() + threads - 1) / threads;
        let build_rel_chunks: Vec<_> = match self.build_relation_key {
            Mem::CudaUniMem(ref m) => m.chunks(build_chunk_size),
            Mem::SysMem(ref m) => m.chunks(build_chunk_size),
            Mem::NumaMem(ref m) => m.as_slice().chunks(build_chunk_size),
            Mem::CudaPinnedMem(ref m) => m.chunks(build_chunk_size),
            Mem::CudaDevMem(_) => panic!("Can't use CUDA device memory on CPU!"),
        }
        .collect();
        let build_pay_chunks: Vec<_> = match self.build_relation_payload {
            Mem::CudaUniMem(ref m) => m.chunks(build_chunk_size),
            Mem::SysMem(ref m) => m.chunks(build_chunk_size),
            Mem::NumaMem(ref m) => m.as_slice().chunks(build_chunk_size),
            Mem::CudaPinnedMem(ref m) => m.chunks(build_chunk_size),
            Mem::CudaDevMem(_) => panic!("Can't use CUDA device memory on CPU!"),
        }
        .collect();
        let probe_rel_chunks: Vec<_> = match self.probe_relation_key {
            Mem::CudaUniMem(ref m) => m.chunks(probe_chunk_size),
            Mem::SysMem(ref m) => m.chunks(probe_chunk_size),
            Mem::NumaMem(ref m) => m.as_slice().chunks(probe_chunk_size),
            Mem::CudaPinnedMem(ref m) => m.chunks(probe_chunk_size),
            Mem::CudaDevMem(_) => panic!("Can't use CUDA device memory on CPU!"),
        }
        .collect();
        let probe_pay_chunks: Vec<_> = match self.probe_relation_payload {
            Mem::CudaUniMem(ref m) => m.chunks(probe_chunk_size),
            Mem::SysMem(ref m) => m.chunks(probe_chunk_size),
            Mem::NumaMem(ref m) => m.as_slice().chunks(probe_chunk_size),
            Mem::CudaPinnedMem(ref m) => m.chunks(probe_chunk_size),
            Mem::CudaDevMem(_) => panic!("Can't use CUDA device memory on CPU!"),
        }
        .collect();
        let result_count_chunks: Vec<_> = result_counts.chunks_mut(threads).collect();

        let hj_builder = hash_join::CpuHashJoinBuilder::default()
            .hashing_scheme(self.hashing_scheme)
            .hash_table(Arc::new(hash_table));

        let mut timer = Instant::now();

        thread_pool.scope(|s| {
            for ((_tid, rel), pay) in (0..threads).zip(build_rel_chunks).zip(build_pay_chunks) {
                let mut hj_op = hj_builder.build();
                s.spawn(move |_| {
                    hj_op.build(rel, pay).expect("Couldn't build hash table");
                });
            }
        });

        let mut dur = timer.elapsed();
        let build_nanos = dur.as_secs() * 10_u64.pow(9) + dur.subsec_nanos() as u64;

        timer = Instant::now();

        thread_pool.scope(|s| {
            for (((_tid, rel), pay), res) in (0..threads)
                .zip(probe_rel_chunks)
                .zip(probe_pay_chunks)
                .zip(result_count_chunks)
            {
                let mut hj_op = hj_builder.build();
                s.spawn(move |_| {
                    hj_op
                        .probe_count(rel, pay, &mut res[0])
                        .expect("Couldn't execute hash table probe");
                });
            }
        });

        dur = timer.elapsed();
        let probe_nanos = dur.as_secs() * 10_u64.pow(9) + dur.subsec_nanos() as u64;

        Ok((build_nanos as f64, probe_nanos as f64))
    }
}
