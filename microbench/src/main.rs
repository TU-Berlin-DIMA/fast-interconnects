// Copyright 2018-2022 Clemens Lutz
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

mod cuda_memcopy;
mod error;
mod memory_bandwidth;
mod memory_latency;
mod numa_memcopy;
mod tlb_latency;
mod types;

use crate::cuda_memcopy::CudaMemcopy;
use crate::error::Result;
use crate::memory_bandwidth::MemoryBandwidth;
use crate::memory_latency::MemoryLatency;
use crate::numa_memcopy::NumaMemcopy;
use crate::tlb_latency::TlbLatency;
use crate::types::*;
use numa_gpu::runtime::allocator;
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::numa::PageType;
use serde_derive::Serialize;
use std::path::PathBuf;
use structopt::clap::arg_enum;
use structopt::StructOpt;

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgDeviceType {
        CPU,
        GPU,
    }
}

// FIXME: rename to CopyDirection
arg_enum! {
    #[derive(Debug, Clone, Copy, Serialize, Eq, PartialEq)]
    pub enum CopyMethod {
        HostToDevice,
        DeviceToHost,
        Bidirectional,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgPageType {
        Default,
        Small,
        TransparentHuge,
        Huge2MB,
        Huge16MB,
        Huge1GB,
        Huge16GB,
    }
}

impl From<ArgPageType> for PageType {
    fn from(arg_page_type: ArgPageType) -> PageType {
        match arg_page_type {
            ArgPageType::Default => PageType::Default,
            ArgPageType::Small => PageType::Small,
            ArgPageType::TransparentHuge => PageType::TransparentHuge,
            ArgPageType::Huge2MB => PageType::Huge2MB,
            ArgPageType::Huge16MB => PageType::Huge16MB,
            ArgPageType::Huge1GB => PageType::Huge1GB,
            ArgPageType::Huge16GB => PageType::Huge16GB,
        }
    }
}

impl From<PageType> for ArgPageType {
    fn from(page_type: PageType) -> ArgPageType {
        match page_type {
            PageType::Default => ArgPageType::Default,
            PageType::Small => ArgPageType::Small,
            PageType::TransparentHuge => ArgPageType::TransparentHuge,
            PageType::Huge2MB => ArgPageType::Huge2MB,
            PageType::Huge16MB => ArgPageType::Huge16MB,
            PageType::Huge1GB => ArgPageType::Huge1GB,
            PageType::Huge16GB => ArgPageType::Huge16GB,
        }
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgMemType {
        System,
        Numa,
        NumaPinned,
        Pinned,
        Unified,
        Device,
    }
}

#[derive(Debug)]
pub struct ArgMemTypeHelper {
    mem_type: ArgMemType,
    node: u16,
    page_type: ArgPageType,
}

impl From<ArgMemTypeHelper> for allocator::MemType {
    fn from(
        ArgMemTypeHelper {
            mem_type,
            node,
            page_type,
        }: ArgMemTypeHelper,
    ) -> Self {
        match mem_type {
            ArgMemType::System => allocator::MemType::SysMem,
            ArgMemType::Numa => allocator::MemType::NumaMem {
                node,
                page_type: page_type.into(),
            },
            ArgMemType::NumaPinned => allocator::MemType::NumaPinnedMem {
                node,
                page_type: page_type.into(),
            },
            ArgMemType::Pinned => allocator::MemType::CudaPinnedMem,
            ArgMemType::Unified => allocator::MemType::CudaUniMem,
            ArgMemType::Device => allocator::MemType::CudaDevMem,
        }
    }
}

#[derive(StructOpt)]
struct Options {
    #[structopt(long = "csv")]
    /// CSV output file
    csv: Option<String>,

    #[structopt(subcommand)]
    cmd: Command,
}

#[derive(StructOpt)]
enum Command {
    #[structopt(name = "bandwidth")]
    /// Memory bandwidth test with random access pattern based on linear congruent generator
    Bandwidth(CmdBandwidth),

    #[structopt(name = "latency")]
    /// Memory latency test based on loop over buffer with increasing strides
    Latency(CmdLatency),

    #[structopt(name = "tlb-latency")]
    /// TLB latency test based on loop over buffer with increasing strides
    TlbLatency(CmdTlbLatency),

    #[structopt(name = "numacopy")]
    /// NUMA interconnect bandwidth test based on memcpy
    NumaCopy(CmdNumaCopy),

    #[structopt(name = "cudacopy")]
    /// GPU interconnect bandwidth test based on cudaMemcpy
    CudaCopy(CmdCudaCopy),
}

#[derive(StructOpt)]
struct CmdBandwidth {
    #[structopt(short = "s", long = "size", default_value = "1024")]
    /// Size of buffer (MB)
    size: usize,

    /// Type of the device.
    #[structopt(
        short = "d",
        long = "device-type",
        default_value = "CPU",
        possible_values = &ArgDeviceType::variants(),
        case_insensitive = true
    )]
    device_type: ArgDeviceType,

    #[structopt(short = "i", long = "device-id", default_value = "0")]
    /// Execute on CPU or GPU (See numactl -H and CUDA device list)
    device_id: u16,

    /// Memory type with which to allocate data.
    //   unified: CUDA Unified memory (default)
    //   system: System memory allocated with std::vec::Vec
    #[structopt(
        short = "t",
        long = "mem-type",
        default_value = "System",
        possible_values = &ArgMemType::variants(),
        case_insensitive = true
    )]
    mem_type: ArgMemType,

    #[structopt(long = "mem-location", default_value = "0")]
    /// Allocate memory on CPU or GPU (See numactl -H and CUDA device list)
    mem_location: u16,

    /// Page type with with to allocate memory
    #[structopt(
        long = "page-type",
        default_value = "Default",
        possible_values = &ArgPageType::variants(),
        case_insensitive = true
    )]
    page_type: ArgPageType,

    #[structopt(long = "threads", default_value = "1,2", require_delimiter = true)]
    /// Number of CPU threads
    threads: Vec<usize>,

    /// Path to CPU affinity map file for CPU workers
    #[structopt(long = "cpu-affinity", parse(from_os_str))]
    cpu_affinity: Option<PathBuf>,

    #[structopt(long = "grid-sizes", require_delimiter = true)]
    /// The CUDA grid sizes to evaluate
    grid_sizes: Vec<u32>,

    #[structopt(long = "block-sizes", require_delimiter = true)]
    /// The CUDA block sizes to evaluate
    block_sizes: Vec<u32>,

    /// Don't align warp-cooperative accesses to the tile size
    ///
    /// By default, warp-cooperative accesses are aligned to the size of the
    /// tile, e.g., a value between 2 and 32. `--misalign-warp` deliberately
    /// misaligns such accesses by one array item.
    #[structopt(long = "misalign-warp")]
    misalign_warp: Option<bool>,

    #[structopt(long = "loop-length", default_value = "1000")]
    /// Number of memory accesses in between cycle measurements
    loop_length: u32,

    #[structopt(long = "target-cycles", default_value = "100")]
    /// Minimum number of clock cycles to measure before quitting (scaled by 10^6)
    target_cycles: u64,

    #[structopt(short = "r", long = "repeat", default_value = "5")]
    /// Number of times to repeat benchmark
    repeat: u32,
}

#[derive(StructOpt)]
struct CmdLatency {
    /// Type of the device.
    #[structopt(
        short = "d",
        long = "device-type",
        default_value = "CPU",
        possible_values = &ArgDeviceType::variants(),
        case_insensitive = true
    )]
    device_type: ArgDeviceType,

    #[structopt(short = "i", long = "device-id", default_value = "0")]
    /// Execute on CPU or GPU (See numactl -H and CUDA device list)
    device_id: u16,

    /// Memory type with which to allocate data.
    //   unified: CUDA Unified memory (default)
    //   system: System memory allocated with std::vec::Vec
    #[structopt(
        short = "t",
        long = "mem-type",
        default_value = "System",
        possible_values = &ArgMemType::variants(),
        case_insensitive = true
    )]
    mem_type: ArgMemType,

    #[structopt(long = "mem-location", default_value = "0")]
    /// Allocate memory on CPU or GPU (See numactl -H and CUDA device list)
    mem_location: u16,

    /// Page type with with to allocate memory
    #[structopt(
        long = "page-type",
        default_value = "Default",
        possible_values = &ArgPageType::variants(),
        case_insensitive = true
    )]
    page_type: ArgPageType,

    #[structopt(short = "l", long = "range-lower", default_value = "1")]
    /// Smallest buffer size (KB)
    range_lower: usize,

    #[structopt(short = "u", long = "range-upper", default_value = "16")]
    /// Largest buffer size (KB)
    range_upper: usize,

    #[structopt(short = "m", long = "stride-lower", default_value = "64")]
    /// Smallest stride length (Bytes)
    stride_lower: usize,

    #[structopt(short = "n", long = "stride-upper", default_value = "1024")]
    /// Largest stride length (Bytes)
    stride_upper: usize,

    #[structopt(short = "r", long = "repeat", default_value = "100")]
    /// Number of times to repeat benchmark
    repeat: u32,
}

#[derive(StructOpt)]
pub struct CmdTlbLatency {
    #[structopt(short = "i", long = "device-id", default_value = "0")]
    /// GPU ID to run on (See CUDA device list)
    device_id: u16,

    /// Memory type with which to allocate data.
    //   unified: CUDA Unified memory (default)
    //   system: System memory allocated with std::vec::Vec
    #[structopt(
        short = "m",
        long = "mem-type",
        default_value = "Device",
        possible_values = &ArgMemType::variants(),
        case_insensitive = true
    )]
    mem_type: ArgMemType,

    #[structopt(short = "l", long = "mem-location", default_value = "0")]
    /// Allocate memory on CPU or GPU (See numactl -H and CUDA device list)
    mem_location: u16,

    /// Page type with with to allocate memory
    #[structopt(
        long = "page-type",
        default_value = "Default",
        possible_values = &ArgPageType::variants(),
        case_insensitive = true
    )]
    page_type: ArgPageType,

    /// Smallest buffer size (MiB)
    #[structopt(short = "x", long = "range-lower", default_value = "4")]
    range_lower: usize,

    /// Largest buffer size (MiB)
    #[structopt(short = "y", long = "range-upper", default_value = "80")]
    range_upper: usize,

    /// List of stride lengths (KiB)
    #[structopt(
        short = "s",
        long = "strides",
        default_value = "32,2048",
        require_delimiter = true
    )]
    strides: Vec<usize>,

    /// Flush the CPU TLB and IOTLB before each measurement series [default: enabled]
    #[structopt(long)]
    iotlb_flush: Option<bool>,
}

#[derive(StructOpt)]
struct CmdNumaCopy {
    #[structopt(short = "t", long = "threads", default_value = "1")]
    /// Number of threads to run (shouldn't exceed #CPUs of one NUMA node)
    threads: usize,

    #[structopt(short = "s", long = "copy-size", default_value = "1024")]
    /// Size of buffer to copy (MB)
    size: usize,

    #[structopt(long = "cpu-node", default_value = "0")]
    /// CPU NUMA node ID
    cpu_node: u16,

    /// Path to CPU affinity map file for CPU workers
    #[structopt(long = "cpu-affinity", conflicts_with = "cpu-node", parse(from_os_str))]
    cpu_affinity: Option<PathBuf>,

    #[structopt(long = "src-node", default_value = "0")]
    /// Source NUMA node ID
    src_node: u16,

    #[structopt(long = "dst-node", default_value = "0")]
    /// Destination NUMA node ID
    dst_node: u16,

    /// Page type with with to allocate memory
    #[structopt(
        long = "page-type",
        default_value = "Default",
        possible_values = &ArgPageType::variants(),
        case_insensitive = true
    )]
    page_type: ArgPageType,

    /// Number of times to repeat benchmark
    #[structopt(short = "r", long = "repeat", default_value = "100")]
    repeat: u32,
}

#[derive(StructOpt)]
struct CmdCudaCopy {
    /// Direction in which to execute the copy
    #[structopt(
        short = "c",
        long = "copy-direction",
        default_value = "HostToDevice,DeviceToHost,Bidirectional",
        require_delimiter = true
    )]
    copy_direction: Vec<CopyMethod>,

    /// Size of buffer to copy (MB)
    #[structopt(
        short = "s",
        long = "copy-size",
        default_value = "1024",
        require_delimiter = true
    )]
    size: Vec<usize>,

    /// Execute on GPU (See CUDA device list)
    #[structopt(short = "i", long = "device-id", default_value = "0")]
    device_id: u16,

    /// Memory type with which to allocate data
    #[structopt(
        short = "m",
        long = "mem-type",
        default_value = "NumaPinned",
        possible_values = &ArgMemType::variants(),
        case_insensitive = true
    )]
    mem_type: ArgMemType,

    /// Allocate memory on NUMA node (See numactl -H)
    #[structopt(long = "mem-location", default_value = "0")]
    mem_location: u16,

    /// Page type with with to allocate memory
    #[structopt(
        long = "page-type",
        default_value = "Default",
        possible_values = &ArgPageType::variants(),
        case_insensitive = true
    )]
    page_type: ArgPageType,

    /// Number of times to repeat benchmark
    #[structopt(short = "r", long = "repeat", default_value = "100")]
    repeat: u32,
}

fn main() -> Result<()> {
    let options = Options::from_args();

    let kb = 2_usize.pow(10);
    let mb = 2_usize.pow(20);

    let mut csv_file = options
        .csv
        .map(|p| std::fs::File::create(p))
        .map(|r| r.expect("Couldn't create CSV file"));

    match options.cmd {
        Command::Bandwidth(ref bw) => {
            let device = match bw.device_type {
                ArgDeviceType::CPU => DeviceId::Cpu(bw.device_id),
                ArgDeviceType::GPU => DeviceId::Gpu(bw.device_id.into()),
            };

            let mem_type_helper = ArgMemTypeHelper {
                mem_type: bw.mem_type,
                node: bw.mem_location,
                page_type: bw.page_type,
            };

            let cpu_affinity = if let Some(ref cpu_affinity_file) = bw.cpu_affinity {
                CpuAffinity::from_file(cpu_affinity_file.as_path())?
            } else {
                CpuAffinity::default()
            };

            MemoryBandwidth::measure(
                device,
                mem_type_helper.into(),
                bw.size * mb,
                bw.threads
                    .iter()
                    .map(|&t| ThreadCount(t))
                    .collect::<Vec<_>>(),
                cpu_affinity,
                bw.grid_sizes.iter().map(|&gs| Grid(gs)).collect::<Vec<_>>(),
                bw.block_sizes
                    .iter()
                    .map(|&bs| Block(bs))
                    .collect::<Vec<_>>(),
                !bw.misalign_warp.unwrap_or(false),
                bw.loop_length,
                Cycles(bw.target_cycles * 10_u64.pow(6)),
                bw.repeat,
                csv_file.as_mut(),
            );
        }
        Command::Latency(ref lat) => {
            let device = match lat.device_type {
                ArgDeviceType::CPU => DeviceId::Cpu(lat.device_id),
                ArgDeviceType::GPU => DeviceId::Gpu(lat.device_id.into()),
            };

            let mem_type_helper = ArgMemTypeHelper {
                mem_type: lat.mem_type,
                node: lat.mem_location,
                page_type: lat.page_type,
            };

            MemoryLatency::measure(
                device,
                mem_type_helper.into(),
                (lat.range_lower * kb)..=(lat.range_upper * kb),
                (lat.stride_lower)..=(lat.stride_upper),
                lat.repeat,
                csv_file.as_mut(),
            );
        }
        Command::TlbLatency(ref tlb) => {
            let mem_type_helper = ArgMemTypeHelper {
                mem_type: tlb.mem_type,
                node: tlb.mem_location,
                page_type: tlb.page_type,
            };
            let template = tlb_latency::DataPoint::from_cmd_options(tlb)?;
            let strides: Vec<_> = tlb.strides.iter().map(|&s| s * kb).collect();

            TlbLatency::measure(
                tlb.device_id,
                mem_type_helper.into(),
                (tlb.range_lower * mb)..=(tlb.range_upper * mb),
                &strides,
                tlb.iotlb_flush.unwrap_or_else(|| true),
                template,
                csv_file.as_mut(),
            )?;
        }
        Command::NumaCopy(ref ncpy) => {
            let cpu_affinity = ncpy
                .cpu_affinity
                .as_ref()
                .map(|cpu_affinity_file| CpuAffinity::from_file(cpu_affinity_file.as_path()))
                .transpose()?;

            let mut numa_memcopy = NumaMemcopy::new(
                ncpy.size * mb,
                ncpy.src_node,
                ncpy.dst_node,
                ncpy.page_type.into(),
                ncpy.threads,
                ncpy.cpu_node,
                cpu_affinity,
            );

            let parallel = ncpy.threads != 1;
            numa_memcopy.measure(parallel, ncpy.repeat, csv_file.as_mut());
        }
        Command::CudaCopy(ref ccpy) => {
            let mem_type_helper = ArgMemTypeHelper {
                mem_type: ccpy.mem_type,
                node: ccpy.mem_location,
                page_type: ccpy.page_type,
            };

            CudaMemcopy::measure(
                ccpy.device_id.into(),
                mem_type_helper.into(),
                &ccpy.copy_direction,
                &ccpy
                    .size
                    .iter()
                    .map(|&bytes| bytes * mb)
                    .collect::<Vec<_>>(),
                ccpy.repeat,
                csv_file.as_mut(),
            );
        }
    }

    Ok(())
}
