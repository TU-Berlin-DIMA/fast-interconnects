// Copyright 2019-2022 Clemens Lutz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use datagen::relation::{KeyAttribute, UniformRelation, ZipfRelation};
use itertools::iproduct;
use num_rational::Ratio;
use num_traits::cast::FromPrimitive;
use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemType};
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::hw_info::NvidiaDriverInfo;
use numa_gpu::runtime::linux_wrapper;
use numa_gpu::runtime::memory::{Mem, MemLock};
use numa_gpu::runtime::numa::{NodeRatio, PageType};
use numa_gpu::utils::DeviceType;
use rustacuda::context::{CacheConfig, Context, ContextFlags, CurrentContext, SharedMemoryConfig};
use rustacuda::device::{Device, DeviceAttribute};
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::DeviceBuffer;
use rustacuda::memory::DeviceCopy;
use rustacuda::stream::{Stream, StreamFlags};
use serde::Serializer;
use serde_derive::Serialize;
use serde_repr::Serialize_repr;
use sql_ops::partition::cpu_radix_partition::{
    CpuHistogramAlgorithm, CpuRadixPartitionAlgorithm, CpuRadixPartitionable, CpuRadixPartitioner,
};
use sql_ops::partition::gpu_radix_partition::{
    GpuHistogramAlgorithm, GpuRadixPartitionAlgorithm, GpuRadixPartitionable, GpuRadixPartitioner,
};
use sql_ops::partition::{
    PartitionOffsets, PartitionedRelation, RadixPartitionInputChunkable, RadixPass,
};
use std::convert::TryInto;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::ops::Range;
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
        Device,
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

impl From<ArgMemTypeHelper> for MemType {
    fn from(
        ArgMemTypeHelper {
            mem_type,
            node_ratios,
            page_type,
        }: ArgMemTypeHelper,
    ) -> Self {
        match mem_type {
            ArgMemType::System => MemType::SysMem,
            ArgMemType::Numa => MemType::NumaMem {
                node: node_ratios[0].node,
                page_type,
            },
            ArgMemType::NumaLazyPinned => MemType::NumaPinnedMem {
                node: node_ratios[0].node,
                page_type,
            },
            ArgMemType::DistributedNuma => MemType::DistributedNumaMem {
                nodes: node_ratios,
                page_type,
            },
            ArgMemType::Pinned => MemType::CudaPinnedMem,
            ArgMemType::Unified => MemType::CudaUniMem,
            ArgMemType::Device => MemType::CudaDevMem,
        }
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgHistogramAlgorithm {
        CpuChunked,
        CpuChunkedSimd,
        GpuChunked,
        GpuContiguous,
    }
}

impl Into<DeviceType<CpuHistogramAlgorithm, GpuHistogramAlgorithm>> for ArgHistogramAlgorithm {
    fn into(self) -> DeviceType<CpuHistogramAlgorithm, GpuHistogramAlgorithm> {
        match self {
            Self::CpuChunked => DeviceType::Cpu(CpuHistogramAlgorithm::Chunked),
            Self::CpuChunkedSimd => DeviceType::Cpu(CpuHistogramAlgorithm::ChunkedSimd),
            Self::GpuChunked => DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
            Self::GpuContiguous => DeviceType::Gpu(GpuHistogramAlgorithm::Contiguous),
        }
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgRadixPartitionAlgorithm {
        NC,
        LASWWC,
        SSWWCv2,
        SSWWCv2G,
        HSSWWCv4,
    }
}

impl Into<GpuRadixPartitionAlgorithm> for ArgRadixPartitionAlgorithm {
    fn into(self) -> GpuRadixPartitionAlgorithm {
        match self {
            Self::NC => GpuRadixPartitionAlgorithm::NC,
            Self::LASWWC => GpuRadixPartitionAlgorithm::LASWWC,
            Self::SSWWCv2 => GpuRadixPartitionAlgorithm::SSWWCv2,
            Self::SSWWCv2G => GpuRadixPartitionAlgorithm::SSWWCv2G,
            Self::HSSWWCv4 => GpuRadixPartitionAlgorithm::HSSWWCv4,
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
    #[structopt(long, hidden = true)]
    #[allow(dead_code)]
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

    /// Device memory buffer sizes per partition per thread block for HSSWWC variants (in KiB)
    #[structopt(long, default_value = "8", require_delimiter = true)]
    dmem_buffer_sizes: Vec<usize>,

    #[structopt(long = "threads")]
    threads: Option<usize>,

    /// Path to CPU affinity map file for CPU workers
    #[structopt(long = "cpu-affinity", parse(from_os_str))]
    cpu_affinity: Option<PathBuf>,

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

    /// NUMA nodes on which the output data is allocated
    ///
    /// NUMA nodes are specified as a list (e.g.: 0,1,2). See numactl -H for
    /// the available NUMA nodes.
    ///
    /// The NUMA node list is only used for the `Numa` and `DistributedNuma`
    /// memory types. Multiple nodes are only valid for the `DistributedNuma`
    /// memory type.
    #[structopt(
        long = "output-location",
        default_value = "0",
        require_delimiter = true
    )]
    output_location: Vec<u16>,

    /// Proportions with which the output data are allocated on multiple nodes
    ///
    /// Given as a list of percentages (e.g.: 20,60,20), that should add up to
    /// 100%.
    ///
    /// The proportions are used only for the `DistributedNuma` memory type and
    /// have no effect on other memory types.
    #[structopt(long, default_value = "100", require_delimiter = true)]
    output_proportions: Vec<usize>,

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
    #[structopt(long, default_value = "target/bench/gpu_radix_partition_operator.csv")]
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
    pub device_codename: Option<String>,
    pub threads: Option<usize>,
    pub grid_size: Option<u32>,
    pub block_size: Option<u32>,
    pub dmem_buffer_bytes: Option<usize>,
    pub input_mem_type: Option<ArgMemType>,
    pub output_mem_type: Option<ArgMemType>,
    pub input_location: Option<u16>,
    #[serde(serialize_with = "serialize_vec")]
    pub output_location: Option<Vec<u16>>,
    #[serde(serialize_with = "serialize_vec")]
    pub output_proportions: Option<Vec<usize>>,
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

fn gpu_radix_partition_benchmark<T, W>(
    histogram_algorithm: DeviceType<CpuHistogramAlgorithm, GpuHistogramAlgorithm>,
    partition_algorithm: GpuRadixPartitionAlgorithm,
    radix_bits_list: &[u32],
    input_data: &(Mem<T>, Mem<T>),
    output_mem_type: &MemType,
    grid_size_hint: &Option<GridSize>,
    dmem_buffer_bytes: usize,
    threads: usize,
    cpu_affinity: &CpuAffinity,
    repeat: u32,
    template: &DataPoint,
    csv_writer: &mut csv::Writer<W>,
) -> Result<(), Box<dyn Error>>
where
    T: Clone
        + Default
        + Send
        + Sync
        + DeviceCopy
        + FromPrimitive
        + CpuRadixPartitionable
        + GpuRadixPartitionable,
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
            let mut radix_prnr = GpuRadixPartitioner::new(
                histogram_algorithm.gpu_or_else(|cpu| cpu.into()),
                partition_algorithm,
                radix_bits.into(),
                &grid_size,
                &block_size,
                dmem_buffer_bytes,
            )?;

            let mut partitioned_relation = PartitionedRelation::new(
                input_data.0.len(),
                histogram_algorithm.either(|cpu| cpu.into(), |gpu| gpu.into()),
                radix_bits.into(),
                grid_size.x,
                Allocator::mem_alloc_fn(output_mem_type.clone()),
                Allocator::mem_alloc_fn(MemType::CudaPinnedMem),
            );
            partitioned_relation.mlock()?;

            let result: Result<(), Box<dyn Error>> = (0..repeat)
                .zip(std::iter::once(true).chain(std::iter::repeat(false)))
                .try_for_each(|(_, warm_up)| {
                    let mut partition_offsets = PartitionOffsets::new(
                        histogram_algorithm.either(|cpu| cpu.into(), |gpu| gpu.into()),
                        grid_size.x,
                        radix_bits,
                        Allocator::mem_alloc_fn(MemType::CudaPinnedMem),
                    );
                    partition_offsets.mlock()?;

                    let prefix_sum_timer = Instant::now();

                    match histogram_algorithm {
                        DeviceType::Cpu(histogram_algorithm) => {
                            let key_slice: &[T] = (&input_data.0)
                                .try_into()
                                .map_err(|_| "Failed to run CPU prefix sum on device memory")?;
                            let key_chunks = key_slice.input_chunks::<T>(grid_size.x)?;
                            let out_chunks = partition_offsets.chunks_mut();

                            thread_pool.scope(|s| {
                                for (input, output) in key_chunks.into_iter().zip(out_chunks) {
                                    s.spawn(move |_| {
                                        let align_bytes = sql_ops::CPU_CACHE_LINE_SIZE as usize;
                                        let mut radix_prnr = CpuRadixPartitioner::new(
                                            histogram_algorithm,
                                            CpuRadixPartitionAlgorithm::NC,
                                            radix_bits,
                                            DerefMemType::AlignedSysMem { align_bytes },
                                        );
                                        radix_prnr
                                            .prefix_sum(input, output)
                                            .expect("Failed to run CPU prefix sum");
                                    })
                                }
                            });
                        }
                        DeviceType::Gpu(_) => {
                            radix_prnr.prefix_sum(
                                RadixPass::First,
                                input_data.0.as_launchable_slice(),
                                &mut partition_offsets,
                                &stream,
                            )?;
                        }
                    }
                    stream.synchronize()?;

                    let prefix_sum_time = prefix_sum_timer.elapsed();
                    let partition_timer = Instant::now();

                    radix_prnr.partition(
                        RadixPass::First,
                        input_data.0.as_launchable_slice(),
                        input_data.1.as_launchable_slice(),
                        &mut partition_offsets,
                        &mut partitioned_relation,
                        &stream,
                    )?;
                    stream.synchronize()?;

                    let partition_time = partition_timer.elapsed();

                    let dp = DataPoint {
                        radix_bits: Some(radix_bits),
                        grid_size: Some(grid_size.x),
                        block_size: Some(block_size.x),
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
    mem_type: &MemType,
    data_distribution: ArgDataDistribution,
    zipf_exponent: Option<f64>,
) -> Result<(Mem<T>, Mem<T>), Box<dyn Error>>
where
    T: Clone + Default + Send + DeviceCopy + FromPrimitive + KeyAttribute,
{
    let key_range = tuples;
    const PAYLOAD_RANGE: Range<usize> = 0..10000;

    let host_alloc = match mem_type.clone().try_into() {
        Err(_) => Allocator::deref_mem_alloc_fn::<T>(DerefMemType::SysMem),
        Ok(mt) => Allocator::deref_mem_alloc_fn::<T>(mt),
    };
    let mut host_data_key = host_alloc(tuples);
    let mut host_data_pay = host_alloc(tuples);

    host_data_key.mlock()?;
    host_data_pay.mlock()?;

    match data_distribution {
        ArgDataDistribution::Unique => {
            UniformRelation::gen_primary_key_par(host_data_key.as_mut_slice(), None)?;
        }
        ArgDataDistribution::Uniform => {
            UniformRelation::gen_attr_par(host_data_key.as_mut_slice(), 0..key_range)?;
        }
        ArgDataDistribution::Zipf if !(zipf_exponent.unwrap() > 0.0) => {
            UniformRelation::gen_attr_par(host_data_key.as_mut_slice(), 0..key_range)?;
        }
        ArgDataDistribution::Zipf => {
            ZipfRelation::gen_attr_par(
                host_data_key.as_mut_slice(),
                key_range,
                zipf_exponent.unwrap(),
            )?;
        }
    }

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

    // Bind main thread to the CPU node closest to the GPU. This improves NVLink latency.
    if let Ok(local_cpu_node) = device.numa_memory_affinity() {
        linux_wrapper::numa_run_on_node(local_cpu_node).expect(&format!(
            "Failed to bind main thread to CPU node {}",
            local_cpu_node
        ));
        linux_wrapper::numa_set_preferred(local_cpu_node);
    } else {
        eprintln!(
            "Warning: Couldn't bind main thread to the CPU closest to GPU {}. This may
                        cause additional latency in measurements.",
            options.device_id
        );
    }

    let csv_file = std::fs::File::create(&options.csv)?;
    let mut csv_writer = csv::Writer::from_writer(csv_file);

    let input_mem_type: MemType = ArgMemTypeHelper {
        mem_type: options.input_mem_type,
        node_ratios: Box::new([NodeRatio {
            node: options.input_location,
            ratio: Ratio::from_integer(0),
        }]),
        page_type: options.page_type.into(),
    }
    .into();

    let node_ratios: Box<[NodeRatio]> = options
        .output_location
        .iter()
        .zip(options.output_proportions.iter())
        .map(|(node, pct)| NodeRatio {
            node: *node,
            ratio: Ratio::new(*pct, 100),
        })
        .collect();

    let output_mem_type: MemType = ArgMemTypeHelper {
        mem_type: options.output_mem_type,
        node_ratios,
        page_type: options.page_type.into(),
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
        output_location: Some(options.output_location.clone()),
        output_proportions: Some(options.output_proportions.clone()),
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
            for (histogram_algorithm, partition_algorithm, dmem_buffer_size) in iproduct!(
                options.histogram_algorithms,
                options.partition_algorithms,
                options.dmem_buffer_sizes
            ) {
                let template = DataPoint {
                    histogram_algorithm: Some(histogram_algorithm),
                    partition_algorithm: Some(partition_algorithm),
                    dmem_buffer_bytes: Some(dmem_buffer_size),
                    ..template.clone()
                };

                gpu_radix_partition_benchmark::<i32, _>(
                    histogram_algorithm.into(),
                    partition_algorithm.into(),
                    &options.radix_bits,
                    &input_data,
                    &output_mem_type,
                    &grid_size_hint,
                    dmem_buffer_size * 1024,
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
            for (histogram_algorithm, partition_algorithm, dmem_buffer_size) in iproduct!(
                options.histogram_algorithms,
                options.partition_algorithms,
                options.dmem_buffer_sizes
            ) {
                let template = DataPoint {
                    histogram_algorithm: Some(histogram_algorithm),
                    partition_algorithm: Some(partition_algorithm),
                    dmem_buffer_bytes: Some(dmem_buffer_size),
                    ..template.clone()
                };

                gpu_radix_partition_benchmark::<i64, _>(
                    histogram_algorithm.into(),
                    partition_algorithm.into(),
                    &options.radix_bits,
                    &input_data,
                    &output_mem_type,
                    &grid_size_hint,
                    dmem_buffer_size * 1024,
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

/// Serialize `Option<Vec<T>>` by converting it into a `String`.
///
/// This is necessary because the `csv` crate does not support nesting `Vec`
/// instead of flattening it.
fn serialize_vec<S, T>(option: &Option<Vec<T>>, ser: S) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
    T: ToString,
{
    if let Some(vec) = option {
        let record = vec
            .iter()
            .enumerate()
            .map(|(i, e)| {
                if i == 0 {
                    e.to_string()
                } else {
                    ",".to_owned() + &e.to_string()
                }
            })
            .collect::<String>();
        ser.serialize_str(&record)
    } else {
        ser.serialize_none()
    }
}
