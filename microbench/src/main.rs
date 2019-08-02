pub mod cuda_memcopy;
pub mod memory_bandwidth;
pub mod memory_latency;
pub mod numa_memcopy;
pub mod sync_latency;
pub mod types;

use crate::cuda_memcopy::CudaMemcopy;
use crate::memory_bandwidth::MemoryBandwidth;
use crate::memory_latency::MemoryLatency;
use crate::numa_memcopy::NumaMemcopy;
use crate::sync_latency::uvm_sync_latency;
use crate::types::*;

use clap::arg_enum;
use numa_gpu::runtime::allocator;
use rustacuda::prelude::*;
use structopt::StructOpt;

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq)]
    enum ArgDeviceType {
        CPU,
        GPU,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub enum ArgMemType {
        System,
        Numa,
        Pinned,
        Unified,
        Device,
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

    #[structopt(name = "sync")]
    /// CPU-GPU synchronization test based on value ping-pong
    Sync(CmdSync),

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
        raw(
            possible_values = "&ArgDeviceType::variants()",
            case_insensitive = "true"
        )
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
        raw(possible_values = "&ArgMemType::variants()", case_insensitive = "true")
    )]
    mem_type: ArgMemType,

    #[structopt(long = "mem-location", default_value = "0")]
    /// Allocate memory on CPU or GPU (See numactl -H and CUDA device list)
    mem_location: u16,

    #[structopt(long = "threads-lower", default_value = "1")]
    /// Number of CPU threads (lower bound)
    threads_lower: usize,

    #[structopt(long = "threads-upper", default_value = "4")]
    /// Number of CPU threads (upper bound)
    threads_upper: usize,

    #[structopt(long = "oversub-lower", default_value = "1")]
    /// Work groups per SM (lower bound)
    oversub_ratio_lower: u32,

    #[structopt(long = "oversub-upper", default_value = "4")]
    /// Work groups per SM (upper bound)
    oversub_ratio_upper: u32,

    #[structopt(long = "warpmul-lower", default_value = "1")]
    /// Warp multiplier (lower bound)
    warp_mul_lower: u32,

    #[structopt(long = "warpmul-upper", default_value = "4")]
    /// Warp multiplier (upper bound)
    warp_mul_upper: u32,

    #[structopt(long = "ilp-lower", default_value = "1")]
    /// Instruction level parallelism (lower bound)
    ilp_lower: u32,

    #[structopt(long = "ilp-upper", default_value = "4")]
    /// Instruction level parallelism (upper bound)
    ilp_upper: u32,

    #[structopt(short = "r", long = "repeat", default_value = "100")]
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
        raw(
            possible_values = "&ArgDeviceType::variants()",
            case_insensitive = "true"
        )
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
        raw(possible_values = "&ArgMemType::variants()", case_insensitive = "true")
    )]
    mem_type: ArgMemType,

    #[structopt(long = "mem-location", default_value = "0")]
    /// Allocate memory on CPU or GPU (See numactl -H and CUDA device list)
    mem_location: u16,

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
struct CmdSync {
    #[structopt(short = "d", long = "device", default_value = "0")]
    /// CUDA device
    device: u32,

    #[structopt(short = "i", long = "iters", default_value = "10")]
    /// Number of times to move value back and forth
    iterations: u32,
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

    #[structopt(long = "src-node", default_value = "0")]
    /// Source NUMA node ID
    src_node: u16,

    #[structopt(long = "dst-node", default_value = "0")]
    /// Destination NUMA node ID
    dst_node: u16,
}

#[derive(StructOpt)]
struct CmdCudaCopy {
    #[structopt(short = "i", long = "device-id", default_value = "0")]
    /// Execute on GPU (See CUDA device list)
    device_id: u16,

    #[structopt(long = "mem-location", default_value = "0")]
    /// Allocate memory on NUMA node (See numactl -H)
    mem_location: u16,

    #[structopt(short = "r", long = "repeat", default_value = "100")]
    /// Number of times to repeat benchmark
    repeat: u32,
}

fn main() {
    let options = Options::from_args();

    rustacuda::init(CudaFlags::empty()).expect("Couldn't initialize CUDA");

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
                location: bw.mem_location,
            };

            MemoryBandwidth::measure(
                device,
                mem_type_helper.into(),
                bw.size * mb,
                ThreadCount(bw.threads_lower)..=ThreadCount(bw.threads_upper),
                OversubRatio(bw.oversub_ratio_lower)..=OversubRatio(bw.oversub_ratio_upper),
                WarpMul(bw.warp_mul_lower)..=WarpMul(bw.warp_mul_upper),
                Ilp(bw.ilp_lower)..=Ilp(bw.ilp_upper),
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
                location: lat.mem_location,
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
        Command::Sync(ref sync) => {
            let r = uvm_sync_latency(sync.device, sync.iterations);
            println!("{:?}", r);
        }
        Command::NumaCopy(ref ncpy) => {
            let mut numa_memcopy = NumaMemcopy::new(
                ncpy.size * mb,
                ncpy.cpu_node,
                ncpy.src_node,
                ncpy.dst_node,
                ncpy.threads,
            );

            let parallel = ncpy.threads != 1;
            numa_memcopy.measure(parallel, csv_file.as_mut());
        }
        Command::CudaCopy(ref ccpy) => {
            CudaMemcopy::measure(
                ccpy.device_id.into(),
                ccpy.mem_location,
                ccpy.repeat,
                csv_file.as_mut(),
            );
        }
    }
}
