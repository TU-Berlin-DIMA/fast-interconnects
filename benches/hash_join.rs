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
extern crate numa_gpu;
extern crate rayon;
#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate structopt;

use average::{Estimate, Max, Min, Quantile, Variance};

use numa_gpu::datagen;
use numa_gpu::error::{Result, ErrorKind};
use numa_gpu::operators::hash_join;
use numa_gpu::runtime::allocator;
use numa_gpu::runtime::backend::*;
use numa_gpu::runtime::cuda_wrapper::prefetch_async;
use numa_gpu::runtime::memory::*;
use numa_gpu::runtime::utils::ensure_physically_backed;
use numa_gpu::runtime::backend::CudaDeviceInfo;

use rustacuda::prelude::*;
use rustacuda::device::DeviceAttribute;
use rustacuda::function::{GridSize, BlockSize};
use rustacuda::event::{Event, EventFlags};

use std::mem::size_of;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use std::collections::vec_deque::VecDeque;

use structopt::StructOpt;

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq)]
    enum ArgDataSet {
        Alb,
        Kim,
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
    pub enum ArgDeviceType {
        CPU,
        GPU,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgHashingScheme {
        Perfect,
        LinearProbing,
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

    /// Output path for measurement files (defaults to current directory)
    #[structopt(short = "o", long = "out-dir", parse(from_os_str), default_value = ".")]
    out_dir: PathBuf,

    /// Memory type with which to allocate data.
    //   unified: CUDA Unified memory (default)
    //   numa: NUMA-local memory on node specified with [inner,outer]-rel-location
    #[structopt(
        short = "m",
        long = "mem-type",
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
        short = "m",
        long = "mem-type",
        default_value = "Unified",
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
    //   alb: Albutiu et al. Massively parallel sort-merge joins"
    //   kim: Kim et al. "Sort vs. hash revisited"
    #[structopt(
        short = "s",
        long = "data-set",
        raw(possible_values = "&ArgDataSet::variants()", case_insensitive = "true")
    )]
    #[allow(dead_code)]
    data_set: Option<ArgDataSet>,

    /// Type of the device.
    #[structopt(
        short = "d",
        long = "device-type",
        default_value = "CPU",
        raw(
            possible_values = "&ArgDeviceType::variants()",
            case_insensitive = "true"
        )
    )]
    device_type: ArgDeviceType,

    #[structopt(short = "i", long = "device-id", default_value = "0")]
    /// Execute on GPU (See CUDA device list)
    device_id: u16,

    #[structopt(short = "t", long = "threads", default_value = "1")]
    threads: usize,
}

#[derive(Debug, Serialize)]
pub struct DataPoint<'h, 'c> {
    pub hostname: &'h str,
    pub device_type: ArgDeviceType,
    pub device_codename: &'c str,
    pub threads: Option<usize>,
    pub hashing_scheme: ArgHashingScheme,
    pub hash_table_memory_type: ArgMemType,
    pub hash_table_memory_node: u16,
    pub hash_table_bytes: usize,
    pub relation_memory_type: ArgMemType,
    pub inner_relation_memory_location: u16,
    pub outer_relation_memory_location: u16,
    pub build_tuples: usize,
    pub build_bytes: usize,
    pub probe_tuples: usize,
    pub probe_bytes: usize,
    pub warm_up: bool,
    pub build_ns: f64,
    pub probe_ns: f64,
}

// FIXME: Support for i32 data type
fn main() -> Result<()> {
    // Parse commandline arguments
    let cmd = CmdOpt::from_args();

    // Initialize CUDA
    rustacuda::init(CudaFlags::empty())?;

    // Allocate memory for data sets
    let mut memory: VecDeque<_> = [
        (
            datagen::popular::Kim::primary_key_len(),
            cmd.mem_type,
            cmd.inner_rel_location,
        ),
        (
            datagen::popular::Kim::primary_key_len(),
            cmd.mem_type,
            cmd.inner_rel_location,
        ),
        (
            datagen::popular::Kim::foreign_key_len(),
            cmd.mem_type,
            cmd.outer_rel_location,
        ),
        (
            datagen::popular::Kim::foreign_key_len(),
            cmd.mem_type,
            cmd.outer_rel_location,
        ),
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
    let mut rel_pk_key = memory.pop_front().ok_or_else(|| ErrorKind::LogicError("Failed to get primary key relation. Is it allocated?".to_string()))?;
    let rel_pk_payload = memory.pop_front().ok_or_else(|| ErrorKind::LogicError("Failed to get primary key relation. Is it allocated?".to_string()))?;
    let mut rel_fk_key = memory.pop_front().ok_or_else(|| ErrorKind::LogicError("Failed to get foreign key relation. Is it allocated?".to_string()))?;
    let rel_fk_payload = memory.pop_front().ok_or_else(|| ErrorKind::LogicError("Failed to get foreign key relation. Is it allocated?".to_string()))?;

    // Generate Kim dataset
    datagen::popular::Kim::gen(rel_pk_key.as_mut_slice(), rel_fk_key.as_mut_slice())
        .expect("Failed to generate Kim data set");

    // Convert ArgHashingScheme to HashingScheme
    let hashing_scheme = match cmd.hashing_scheme {
        ArgHashingScheme::Perfect => hash_join::HashingScheme::Perfect,
        ArgHashingScheme::LinearProbing => hash_join::HashingScheme::LinearProbing,
    };

    // Device tuning
    let device = Device::get_device(cmd.device_id.into())?;
    let cuda_cores = device.cores()?;
    let warp_size = device.get_attribute(DeviceAttribute::WarpSize)? as u32;
    let warp_overcommit_factor = 2;
    let grid_overcommit_factor = 32;

    let block_size = BlockSize::x(warp_size * warp_overcommit_factor);
    let grid_size = GridSize::x(cuda_cores * grid_overcommit_factor);

    // Construct benchmark
    let hjb = HashJoinBench {
        hashing_scheme,
        hash_table_len: 4 * 128 * 2_usize.pow(20),
        build_relation_key: rel_pk_key.into(),
        build_relation_payload: rel_pk_payload.into(),
        probe_relation_key: rel_fk_key.into(),
        probe_relation_payload: rel_fk_payload.into(),
        build_dim: (grid_size.clone(), block_size.clone()),
        probe_dim: (grid_size.clone(), block_size.clone()),
    };

    // Get device information
    let dev_codename_str = match cmd.device_type {
        ArgDeviceType::CPU => cpu_codename(),
        ArgDeviceType::GPU => device.name()?,
    };

    // Construct data point template for CSV
    let dp = DataPoint {
        hostname: "",
        device_type: cmd.device_type,
        device_codename: dev_codename_str.as_str(),
        threads: if cmd.device_type == ArgDeviceType::CPU {
            Some(cmd.threads)
        } else {
            None
        },
        hashing_scheme: cmd.hashing_scheme,
        hash_table_memory_type: cmd.hash_table_mem_type,
        hash_table_memory_node: cmd.hash_table_location,
        hash_table_bytes: hjb.hash_table_len * size_of::<i64>(),
        relation_memory_type: cmd.mem_type,
        inner_relation_memory_location: cmd.inner_rel_location,
        outer_relation_memory_location: cmd.outer_rel_location,
        build_tuples: hjb.build_relation_key.len(),
        build_bytes: hjb.build_relation_key.len() * size_of::<i64>(),
        probe_tuples: hjb.probe_relation_key.len(),
        probe_bytes: hjb.probe_relation_key.len() * size_of::<i64>(),
        warm_up: false,
        build_ns: 0.0,
        probe_ns: 0.0,
    };

    // Select the operator to run, depending on the device type
    let dev_type = cmd.device_type.clone();
    let mem_type = cmd.hash_table_mem_type;
    let location = cmd.hash_table_location;
    let threads = cmd.threads;
    let hjc = || match dev_type {
        ArgDeviceType::CPU => {
            let ht_alloc = allocator::Allocator::deref_mem_alloc_fn::<i64>(
                ArgMemTypeHelper { mem_type, location }.into(),
            );
            hjb.cpu_hash_join(threads, ht_alloc)
        }
        ArgDeviceType::GPU => {
            let ht_alloc = allocator::Allocator::mem_alloc_fn::<i64>(
                ArgMemTypeHelper { mem_type, location }.into(),
            );
            hjb.cuda_hash_join(device, ht_alloc)
        }
    };

    // Run experiment
    measure("hash_join_kim", cmd.repeat, cmd.out_dir, dp, hjc)
        .expect("Failure: hash join benchmark");
    Ok(())
}

fn measure<F>(name: &str, repeat: u32, out_dir: PathBuf, template: DataPoint, func: F) -> Result<()>
where
    F: Fn() -> Result<(f64, f64)>,
{
    let hostname = &hostname::get_hostname().ok_or_else(|| "Couldn't get hostname")?;

    let measurements = (0..repeat)
        .map(|_| {
            func().map(|(build_ns, probe_ns)| DataPoint {
                hostname,
                warm_up: false,
                build_ns,
                probe_ns,
                ..template
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let csv_path = out_dir.with_file_name(name).with_extension("csv");

    let csv_file = std::fs::File::create(csv_path)?;

    let mut csv = csv::Writer::from_writer(csv_file);
    ensure!(
        measurements
            .iter()
            .try_for_each(|row| csv.serialize(row))
            .is_ok(),
        "Couldn't write serialized measurements"
    );

    concatenate!(
        Estimator,
        [Variance, variance, mean, error],
        [Quantile, quantile, quantile],
        [Min, min, min],
        [Max, max, max]
    );

    let time_stats: Estimator = measurements
        .iter()
        .map(|row| row.probe_ns / 10_f64.powf(6.0))
        .collect();

    let tput_stats: Estimator = measurements
        .iter()
        .map(|row| (row.probe_bytes as f64, row.probe_ns))
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

#[allow(dead_code)]
struct HashJoinBench {
    hashing_scheme: hash_join::HashingScheme,
    hash_table_len: usize,
    build_relation_key: Mem<i64>,
    build_relation_payload: Mem<i64>,
    probe_relation_key: Mem<i64>,
    probe_relation_payload: Mem<i64>,
    build_dim: (GridSize, BlockSize),
    probe_dim: (GridSize, BlockSize),
}

impl HashJoinBench {
    fn cuda_hash_join(&self, device: Device, hash_table_alloc: allocator::MemAllocFn<i64>) -> Result<(f64, f64)> {
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // FIXME: specify load factor as argument
        let hash_table_mem = hash_table_alloc(self.hash_table_len);
        let hash_table = hash_join::HashTable::new_on_gpu(hash_table_mem, self.hash_table_len)?;
        let mut result_counts = allocator::Allocator::alloc_mem(
            allocator::MemType::CudaUniMem,
            (self.probe_dim.0.x * self.probe_dim.1.x) as usize,
        );

        // Initialize counts
        if let CudaUniMem(ref mut c) = result_counts {
            c.iter_mut().map(|count| *count = 0).for_each(drop);
        }

        // Tune memory locations
        [
            &self.build_relation_key,
            &self.probe_relation_key,
            &self.build_relation_payload,
            &self.probe_relation_payload,
        ]
        .iter()
        .filter_map(|mem| {
            if let CudaUniMem(m) = mem {
                Some(m)
            } else {
                None
            }
        })
        .map(|mem| prefetch_async(mem, 0, unsafe { std::mem::zeroed() }))
        .collect::<Result<()>>()?;

        stream.synchronize()?;

        let mut hj_op = hash_join::CudaHashJoinBuilder::default()
            .hashing_scheme(self.hashing_scheme)
            .build_dim(self.build_dim.0.clone(), self.build_dim.1.clone())
            .probe_dim(self.probe_dim.0.clone(), self.probe_dim.1.clone())
            .hash_table(hash_table)
            .build()?;

        let start_event = Event::new(EventFlags::DEFAULT)?;
        let stop_event = Event::new(EventFlags::DEFAULT)?;

        start_event.record(&stream)?;
        hj_op.build(&self.build_relation_key, &self.build_relation_payload, &stream)?;

        stop_event.record(&stream).and_then(|e| e.synchronize())?;
        let build_millis = stop_event.elapsed_time(&start_event)?;

        start_event.record(&stream)?;
        hj_op.probe_count(
            &self.probe_relation_key,
            &self.probe_relation_payload,
            &mut result_counts,
            &stream,
        )?;

        stop_event.record(&stream).and_then(|e| e.synchronize())?;
        let probe_millis = stop_event.elapsed_time(&start_event)?;

        stream.synchronize()?;
        Ok((
            build_millis as f64 * 10_f64.powf(6.0),
            probe_millis as f64 * 10_f64.powf(6.0),
        ))
    }

    fn cpu_hash_join(
        &self,
        threads: usize,
        hash_table_alloc: allocator::DerefMemAllocFn<i64>,
    ) -> Result<(f64, f64)> {
        let mut hash_table_mem = hash_table_alloc(self.hash_table_len);
        ensure_physically_backed(&mut hash_table_mem);
        let hash_table = hash_join::HashTable::new_on_cpu(hash_table_mem, self.hash_table_len)?;
        let mut result_counts = vec![0; threads];

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("Couldn't create thread pool");
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
