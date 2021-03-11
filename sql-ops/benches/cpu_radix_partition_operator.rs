/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use datagen::relation::{KeyAttribute, UniformRelation, ZipfRelation};
use itertools::{iproduct, izip};
use num_rational::Ratio;
use num_traits::cast::FromPrimitive;
use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemType};
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::dispatcher::{
    HetMorselExecutorBuilder, IntoHetMorselIterator, MorselSpec, WorkerCpuAffinity,
};
use numa_gpu::runtime::hw_info;
use numa_gpu::runtime::linux_wrapper;
use numa_gpu::runtime::memory::{DerefMem, MemLock};
use numa_gpu::runtime::numa::{NodeRatio, PageType};
use rustacuda::context::{Context, ContextFlags, CurrentContext};
use rustacuda::device::Device;
use rustacuda::memory::{AsyncCopyDestination, DeviceBuffer, DeviceCopy};
use rustacuda::stream::{Stream, StreamFlags};
use serde_derive::Serialize;
use serde_repr::Serialize_repr;
use sql_ops::partition::cpu_radix_partition::{
    CpuHistogramAlgorithm, CpuRadixPartitionAlgorithm, CpuRadixPartitionable, CpuRadixPartitioner,
};
use sql_ops::partition::{
    PartitionOffsets, PartitionedRelation, RadixPartitionInputChunkable, Tuple,
};
use std::convert::TryInto;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::mem;
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::sync::Arc;
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
    }
}

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

#[derive(Debug)]
pub struct ArgMemTypeHelper {
    pub mem_type: ArgMemType,
    pub node_ratios: Box<[NodeRatio]>,
    pub page_type: PageType,
}

impl From<ArgMemTypeHelper> for DerefMemType {
    fn from(
        ArgMemTypeHelper {
            mem_type,
            node_ratios,
            page_type,
        }: ArgMemTypeHelper,
    ) -> Self {
        match mem_type {
            ArgMemType::System => DerefMemType::SysMem,
            ArgMemType::Numa => DerefMemType::NumaMem {
                node: node_ratios[0].node,
                page_type,
            },
            ArgMemType::NumaLazyPinned => DerefMemType::NumaPinnedMem {
                node: node_ratios[0].node,
                page_type,
            },
            ArgMemType::DistributedNuma => DerefMemType::DistributedNumaMem {
                nodes: node_ratios,
                page_type,
            },
            ArgMemType::Pinned => DerefMemType::CudaPinnedMem,
            ArgMemType::Unified => DerefMemType::CudaUniMem,
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
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgExecutionMethod {
        CpuRadixPartition,
        CpuRadixPartitionWithTransfer,
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

    /// Select the execution strategies to run.
    #[structopt(
        long,
        default_value = "CpuRadixPartition",
        possible_values = &ArgExecutionMethod::variants(),
        case_insensitive = true,
        require_delimiter = true
    )]
    execution_methods: Vec<ArgExecutionMethod>,

    #[structopt(long = "cpu-morsel-bytes", default_value = "33554432")]
    cpu_morsel_bytes: usize,

    /// No effect (passed by Cargo to run only benchmarks instead of unit tests)
    #[structopt(long, hidden = true)]
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

    /// Transfer data to CUDA device with ID
    #[structopt(long = "device-id", default_value = "0")]
    device_id: u16,

    #[structopt(long = "threads")]
    threads: Option<usize>,

    /// Path to CPU affinity map file for CPU workers
    #[structopt(long = "cpu-affinity", parse(from_os_str))]
    cpu_affinity: Option<PathBuf>,

    /// Memory type with which to allocate input relation
    #[structopt(
        long = "input-mem-type",
        default_value = "Numa",
        possible_values = &ArgMemType::variants(),
        case_insensitive = true
    )]
    input_mem_type: ArgMemType,

    /// Memory type with which to allocate output relation
    #[structopt(
        long = "output-mem-type",
        default_value = "Numa",
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
    pub hostname: String,
    pub histogram_algorithm: Option<ArgHistogramAlgorithm>,
    pub partition_algorithm: Option<ArgRadixPartitionAlgorithm>,
    pub execution_method: Option<ArgExecutionMethod>,
    pub device_codename: Option<String>,
    pub cpu_morsel_bytes: Option<usize>,
    pub threads: Option<usize>,
    pub grid_size: Option<u32>,
    pub block_size: Option<u32>,
    pub input_mem_type: Option<ArgMemType>,
    pub output_mem_type: Option<ArgMemType>,
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

type BenchFn<T, W> = fn(
    CpuHistogramAlgorithm,
    CpuRadixPartitionAlgorithm,
    &[u32],
    &mut (DerefMem<T>, DerefMem<T>),
    &MemType,
    usize,
    &CpuAffinity,
    &MorselSpec,
    u32,
    &DataPoint,
    &mut csv::Writer<W>,
) -> Result<(), Box<dyn Error>>;

fn cpu_radix_partition_benchmark<T, W>(
    prefix_sum_algorithm: CpuHistogramAlgorithm,
    partition_algorithm: CpuRadixPartitionAlgorithm,
    radix_bits_list: &[u32],
    input_data: &mut (DerefMem<T>, DerefMem<T>),
    output_mem_type: &MemType,
    threads: usize,
    cpu_affinity: &CpuAffinity,
    _morsel_spec: &MorselSpec,
    repeat: u32,
    template: &DataPoint,
    csv_writer: &mut csv::Writer<W>,
) -> Result<(), Box<dyn Error>>
where
    T: Clone + Default + Send + Sync + FromPrimitive + CpuRadixPartitionable,
    W: Write,
{
    let tuples = input_data.0.len();

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

fn cpu_radix_partition_and_transfer_benchmark<T, W>(
    prefix_sum_algorithm: CpuHistogramAlgorithm,
    partition_algorithm: CpuRadixPartitionAlgorithm,
    radix_bits_list: &[u32],
    input_data: &mut (DerefMem<T>, DerefMem<T>),
    output_mem_type: &MemType,
    threads: usize,
    cpu_affinity: &CpuAffinity,
    morsel_spec: &MorselSpec,
    repeat: u32,
    template: &DataPoint,
    csv_writer: &mut csv::Writer<W>,
) -> Result<(), Box<dyn Error>>
where
    T: Copy + Clone + Default + Send + Sync + FromPrimitive + CpuRadixPartitionable,
    W: Write,
{
    let worker_cpu_affinity = WorkerCpuAffinity {
        cpu_workers: cpu_affinity.clone(),
        gpu_workers: CpuAffinity::default(),
    };

    let morsel_len = morsel_spec.cpu_morsel_bytes / mem::size_of::<Tuple<T, T>>();
    let mut executor = HetMorselExecutorBuilder::new()
        .cpu_threads(threads)
        .worker_cpu_affinity(worker_cpu_affinity)
        .morsel_spec(morsel_spec.clone())
        .build()?;

    radix_bits_list
        .iter()
        .map(|&radix_bits| {
            let mut data_ref = (input_data.0.as_mut_slice(), input_data.1.as_mut_slice());
            let mut morsel_iter = data_ref.into_het_morsel_iter(&mut executor).with_state(
                |tid| {
                    const PARTITION_CHUNKS: u32 = 1;
                    const PIPELINE_STAGES: usize = 2;

                    let cpu_id = cpu_affinity
                        .thread_to_cpu(tid)
                        .expect("Failed to map thread ID to CPU ID");
                    let local_node = linux_wrapper::numa_node_of_cpu(cpu_id)
                        .expect("Failed to map CPU to NUMA node");

                    let radix_prnr = CpuRadixPartitioner::new(
                        prefix_sum_algorithm,
                        partition_algorithm,
                        radix_bits,
                        DerefMemType::AlignedSysMem{align_bytes: sql_ops::CPU_CACHE_LINE_SIZE as usize},
                    );

                    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

                    let partition_offsets = PartitionOffsets::new(
                        prefix_sum_algorithm.into(),
                        PARTITION_CHUNKS,
                        radix_bits,
                        Allocator::mem_alloc_fn(
                        MemType::NumaMem {
                            node: local_node,
                            page_type: output_mem_type.page_type(),
                        },
                            ),
                    );

                    let partitioned_relations: [_; PIPELINE_STAGES] = [PartitionedRelation::new(
                        morsel_len,
                        prefix_sum_algorithm.into(),
                        radix_bits,
                        PARTITION_CHUNKS,
                        Allocator::mem_alloc_fn(output_mem_type.clone()),
                        Allocator::mem_alloc_fn(output_mem_type.clone()),
                    ), PartitionedRelation::new(
                        morsel_len,
                        prefix_sum_algorithm.into(),
                        radix_bits,
                        PARTITION_CHUNKS,
                        Allocator::mem_alloc_fn(
                            MemType::NumaPinnedMem {
                                node: local_node,
                                page_type: output_mem_type.page_type(),
                            }
                            ),
                        Allocator::mem_alloc_fn(
                        MemType::NumaMem {
                            node: local_node,
                            page_type: output_mem_type.page_type(),
                        }),
                    )];

                    let device_relation: DeviceBuffer<Tuple<T, T>> =
                        unsafe { DeviceBuffer::uninitialized(partitioned_relations[0].padded_len())? };

                    Ok((
                        radix_prnr,
                        stream,
                        partition_offsets,
                        partitioned_relations,
                        device_relation,
                    ))
                },
                |_, _| Ok(()),
            )?;

            let result: Result<(), Box<dyn Error>> = (0..repeat)
                .zip(std::iter::once(true).chain(std::iter::repeat(false)))
                .try_for_each(|(_, warm_up)| {

                    let partition_timer = Instant::now();

                morsel_iter.fold(
                |(key_morsel, pay_morsel), thread_state| {
                    let (
                        radix_prnr,
                        stream,
                        partition_offsets,
                        partitioned_relations,
                        device_relation,
                    ) = thread_state;
                    let key_chunks = key_morsel
                        .input_chunks::<T>(1)
                        .expect("Failed to get input chunk");
                    let pay_chunks = pay_morsel
                        .input_chunks::<T>(1)
                        .expect("Failed to get input chunk");
                    let offsets_chunk = partition_offsets.chunks_mut().nth(0).unwrap();
                    let partitioned_chunk = partitioned_relations[0].chunks_mut().nth(0).unwrap();

                    radix_prnr
                        .prefix_sum(key_chunks[0].clone(), offsets_chunk)
                        .expect("Failed to prefix sum the data");

                    let offsets_chunk = partition_offsets.chunks_mut().nth(0).unwrap();

                    radix_prnr
                        .partition(
                            key_chunks[0].clone(),
                            pay_chunks[0].clone(),
                            offsets_chunk,
                            partitioned_chunk,
                        )
                        .expect("Failed to partition the data");

                    let relation_ref: &[_] = (&partitioned_relations[0].relation)
                        .try_into()
                        .map_err(|_| {})
                        .expect(
                            "Partitioned relation is a device buffer, failed to get a slice reference",
                        );

                    unsafe {
                        device_relation.async_copy_from(relation_ref, stream)?;
                    }

                    // Swap the relations so that the CPU partitioning and DMA
                    // transfer can take place concurrently
                    partitioned_relations.swap(0, 1);

                    Ok(())
                },
                |_, _, _| Ok(()),
            )?;

            CurrentContext::synchronize()?;

            let partition_time = partition_timer.elapsed();

            let dp = DataPoint {
                radix_bits: Some(radix_bits),
                threads: Some(threads),
                warm_up: Some(warm_up),
                prefix_sum_ns: Some(0),
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

    // FIXME
    let morsel_spec = MorselSpec {
        cpu_morsel_bytes: options.cpu_morsel_bytes,
        gpu_morsel_bytes: 0,
    };

    // Initialize CUDA
    rustacuda::init(rustacuda::CudaFlags::empty())?;
    let device = Device::get_device(options.device_id.into())?;
    let _context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    if let Some(parent) = options.csv.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    let csv_file = std::fs::File::create(&options.csv)?;
    let mut csv_writer = csv::Writer::from_writer(csv_file);

    let input_mem_type: DerefMemType = ArgMemTypeHelper {
        mem_type: options.input_mem_type,
        node_ratios: Box::new([NodeRatio {
            node: options.input_location,
            ratio: Ratio::from_integer(0),
        }]),
        page_type: options.page_type.into(),
    }
    .into();

    let output_mem_type: DerefMemType = ArgMemTypeHelper {
        mem_type: options.output_mem_type,
        node_ratios: Box::new([NodeRatio {
            node: options.output_location,
            ratio: Ratio::from_integer(0),
        }]),
        page_type: options.page_type.into(),
    }
    .into();
    let output_mem_type: MemType = output_mem_type.into();

    let template = DataPoint {
        hostname: hostname::get()?
            .into_string()
            .expect("Couldn't convert hostname into UTF-8 string"),
        device_codename: Some(hw_info::cpu_codename()?),
        input_mem_type: Some(options.input_mem_type),
        cpu_morsel_bytes: Some(options.cpu_morsel_bytes),
        output_mem_type: Some(options.output_mem_type),
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
            let mut input_data = alloc_and_gen(
                options.tuples,
                &input_mem_type,
                options.data_distribution,
                options.zipf_exponent,
            )?;

            for (prefix_sum_algorithm, partition_algorithm, execution_method) in iproduct!(
                options.prefix_sum_algorithms,
                options.partition_algorithms,
                options.execution_methods
            ) {
                let f: BenchFn<i32, _> = match execution_method {
                    ArgExecutionMethod::CpuRadixPartition => {
                        cpu_radix_partition_benchmark::<i32, _>
                    }
                    ArgExecutionMethod::CpuRadixPartitionWithTransfer => {
                        cpu_radix_partition_and_transfer_benchmark::<i32, _>
                    }
                };

                let template = DataPoint {
                    histogram_algorithm: Some(prefix_sum_algorithm),
                    partition_algorithm: Some(partition_algorithm),
                    execution_method: Some(execution_method),
                    ..template.clone()
                };

                f(
                    prefix_sum_algorithm.into(),
                    partition_algorithm.into(),
                    &options.radix_bits,
                    &mut input_data,
                    &output_mem_type,
                    threads,
                    &cpu_affinity,
                    &morsel_spec,
                    options.repeat,
                    &template,
                    &mut csv_writer,
                )?;
            }
        }
        ArgTupleBytes::Bytes16 => {
            let mut input_data = alloc_and_gen(
                options.tuples,
                &input_mem_type,
                options.data_distribution,
                options.zipf_exponent,
            )?;

            for (prefix_sum_algorithm, partition_algorithm, execution_method) in iproduct!(
                options.prefix_sum_algorithms,
                options.partition_algorithms,
                options.execution_methods
            ) {
                let f: BenchFn<i64, _> = match execution_method {
                    ArgExecutionMethod::CpuRadixPartition => {
                        cpu_radix_partition_benchmark::<i64, _>
                    }
                    ArgExecutionMethod::CpuRadixPartitionWithTransfer => {
                        cpu_radix_partition_and_transfer_benchmark::<i64, _>
                    }
                };

                let template = DataPoint {
                    histogram_algorithm: Some(prefix_sum_algorithm),
                    partition_algorithm: Some(partition_algorithm),
                    execution_method: Some(execution_method),
                    ..template.clone()
                };

                f(
                    prefix_sum_algorithm.into(),
                    partition_algorithm.into(),
                    &options.radix_bits,
                    &mut input_data,
                    &output_mem_type,
                    threads,
                    &cpu_affinity,
                    &morsel_spec,
                    options.repeat,
                    &template,
                    &mut csv_writer,
                )?;
            }
        }
    }

    Ok(())
}
