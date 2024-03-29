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

use super::gpu_measurement::GpuMeasurementParameters;
use super::{Benchmark, ItemBytes, MemoryOperation, TileSize};
use crate::types::Cycles;
use numa_gpu::runtime::memory::Mem;
use numa_gpu::runtime::nvml::ThrottleReasons;
use nvml_wrapper::device::Device as NvmlDevice;
use nvml_wrapper::enum_wrappers::device::Clock;
use rustacuda::context::CurrentContext;
use rustacuda::event::{Event, EventFlags};
use rustacuda::launch;
use rustacuda::memory::{CopyDestination, DeviceBox};
use rustacuda::module::Module;
use rustacuda::stream::{Stream, StreamFlags};
use std::ffi::CString;
use std::mem;

/// Generate CUDA function bindings that are callable from Rust
///
/// The CUDA memory benchmark functions are named using the scheme:
///
/// - `gpu_{operator}_bandwidth_{benchmark}_{bytes}B`
/// - `gpu_{operator}_bandwidth_{benchmark}_{bytes}B_{tile size}T`
///
/// CUDA provides `cuModuleGetFunction`, that returns a handle to a function within a module by
/// searching for the function name. Thus, `gen_cuda_fucntions` constructs the function name as a
/// string.
///
/// # Usage
///
/// The macro takes a tuple `(Benchmark, MemoryOperation, ItemBytes, TileSize)` and a list of cases
/// and returns a `&'static str` string. Each case is encoded as `case(bench, op, bytes,
/// tile_size)`.
///
/// Benchmarks:
/// - Seq
/// - Lcg
///
/// MemoryOperations:
/// - Read
/// - Write
/// - CompareAndSwap
///
/// Bytes:
/// - Bytes4
/// - Bytes8
/// - Bytes16
///
/// TileSize:
/// - Threads1
/// - Threads2
/// - Threads4
/// - Threads8
/// - Threads16
/// - Threads32
///
/// # Example
///
/// ```rust,no_run
/// let cuda_function = gen_cuda_functions!(
///   (bench, op, item_bytes, tile_size),
///   case(Seq, Read, Bytes4, Threads1),
///   case(Lcg, Write, Bytes16, Threads16)
/// );
/// ```
macro_rules! gen_cuda_functions {
    (@as_bench_ident Seq) => {Benchmark::Sequential};
    (@as_bench_ident Lcg) => {Benchmark::LinearCongruentialGenerator};

    (@as_bench_str Seq) => {"seq_"};
    (@as_bench_str Lcg) => {"lcg_"};

    (@as_op_str Read) => {"read_"};
    (@as_op_str Write) => {"write_"};
    (@as_op_str CompareAndSwap) => {"cas_"};

    (@as_bytes_str Bytes4) => {"4B"};
    (@as_bytes_str Bytes8) => {"8B"};
    (@as_bytes_str Bytes16) => {"16B"};

    (@as_threads_str Threads1) => {""};
    (@as_threads_str Threads2) => {"_2T"};
    (@as_threads_str Threads4) => {"_4T"};
    (@as_threads_str Threads8) => {"_8T"};
    (@as_threads_str Threads16) => {"_16T"};
    (@as_threads_str Threads32) => {"_32T"};

    (@gen_pattern ($benchmark:ident, $operation:ident, $bytes:ident, $threads:ident)) => {
        (
            gen_cuda_functions!(@as_bench_ident $benchmark),
            MemoryOperation::$operation,
            ItemBytes::$bytes,
            TileSize::$threads,
        )
    };

    (@gen_function_str ($benchmark:ident, $operation:ident, $bytes:ident, $threads:ident)) => {
            Some(concat!(
                    "gpu_",
                    gen_cuda_functions!(@as_op_str $operation),
                    "bandwidth_",
                    gen_cuda_functions!(@as_bench_str $benchmark),
                    gen_cuda_functions!(@as_bytes_str $bytes),
                    gen_cuda_functions!(@as_threads_str $threads)
            ))
    };

    // FIXME: handle `None` cases explicitly instead of with catch-all
    ($obj:expr, $(case($benchmark:ident, $operation:ident, $bytes:ident, $threads:ident)),*) => {
        match $obj {
            $(gen_cuda_functions!(@gen_pattern ($benchmark, $operation, $bytes, $threads)) =>
              gen_cuda_functions!(@gen_function_str ($benchmark, $operation, $bytes, $threads))),*,
            _ => None,
        }
    };
}

#[derive(Debug)]
pub(super) struct GpuMemoryBandwidth {
    buffer_len: usize,
    warp_misalignment: usize,
    loop_length: u32,
    target_cycles: Cycles,
    stream: Stream,
}

impl GpuMemoryBandwidth {
    pub(super) fn new(
        buffer_len: usize,
        warp_misalignment: usize,
        loop_length: u32,
        target_cycles: Cycles,
    ) -> Self {
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).expect("Couldn't create CUDA stream");

        Self {
            buffer_len,
            warp_misalignment,
            loop_length,
            target_cycles,
            stream,
        }
    }

    pub(super) fn run(
        bench: Benchmark,
        op: MemoryOperation,
        item_bytes: ItemBytes,
        tile_size: TileSize,
        state: &mut Self,
        mem: &Mem<u32>,
        mp: &GpuMeasurementParameters,
        nvml_device: &NvmlDevice,
    ) -> Option<(u32, Option<ThrottleReasons>, u64, Cycles, u64)> {
        assert!(
            state.buffer_len.is_power_of_two(),
            "Data size must be a power of two!"
        );

        // Get GPU clock rate that applications run at
        #[cfg(not(target_arch = "aarch64"))]
        let clock_rate_mhz = nvml_device
            .clock_info(Clock::SM)
            .expect("Couldn't get clock rate with NVML");

        // FIXME: remove ARM-specific code paths and replace with portable code
        #[cfg(target_arch = "aarch64")]
        let clock_rate_mhz = CurrentContext::get_device()
            .expect("Couldn't get CUDA device")
            .clock_rate()
            .expect("Couldn't get clock rate");

        // FIXME: load the module lazy globally
        let module_path = CString::new(env!("CUDAUTILS_PATH"))
            .expect("Failed to load CUDA module, check your CUDAUTILS_PATH");
        let module = Module::load_from_file(&module_path).expect("Failed to load CUDA module");
        let stream = &state.stream;

        let mut memory_accesses_device =
            DeviceBox::new(&0_u64).expect("Couldn't allocate device memory");
        let mut measured_cycles = DeviceBox::new(&0_u64).expect("Couldn't allocate device memory");

        let timer_begin = Event::new(EventFlags::DEFAULT).expect("Couldn't create CUDA event");
        let timer_end = Event::new(EventFlags::DEFAULT).expect("Couldn't create CUDA event");
        timer_begin
            .record(&state.stream)
            .expect("Couldn't record CUDA event");

        let function_name = gen_cuda_functions!(
            (bench, op, item_bytes, tile_size),
            case(Seq, Read, Bytes4, Threads1),
            case(Seq, Read, Bytes8, Threads1),
            case(Seq, Read, Bytes16, Threads1),
            case(Seq, Write, Bytes4, Threads1),
            case(Seq, Write, Bytes8, Threads1),
            case(Seq, Write, Bytes16, Threads1),
            case(Seq, CompareAndSwap, Bytes4, Threads1),
            case(Seq, CompareAndSwap, Bytes8, Threads1),
            case(Lcg, Read, Bytes4, Threads1),
            case(Lcg, Read, Bytes8, Threads1),
            case(Lcg, Read, Bytes16, Threads1),
            case(Lcg, Read, Bytes4, Threads2),
            case(Lcg, Read, Bytes8, Threads2),
            case(Lcg, Read, Bytes16, Threads2),
            case(Lcg, Read, Bytes4, Threads4),
            case(Lcg, Read, Bytes8, Threads4),
            case(Lcg, Read, Bytes16, Threads4),
            case(Lcg, Read, Bytes4, Threads8),
            case(Lcg, Read, Bytes8, Threads8),
            case(Lcg, Read, Bytes16, Threads8),
            case(Lcg, Read, Bytes4, Threads16),
            case(Lcg, Read, Bytes8, Threads16),
            case(Lcg, Read, Bytes16, Threads16),
            case(Lcg, Read, Bytes4, Threads32),
            case(Lcg, Read, Bytes8, Threads32),
            case(Lcg, Read, Bytes16, Threads32),
            case(Lcg, Write, Bytes4, Threads1),
            case(Lcg, Write, Bytes8, Threads1),
            case(Lcg, Write, Bytes16, Threads1),
            case(Lcg, Write, Bytes4, Threads2),
            case(Lcg, Write, Bytes8, Threads2),
            case(Lcg, Write, Bytes16, Threads2),
            case(Lcg, Write, Bytes4, Threads4),
            case(Lcg, Write, Bytes8, Threads4),
            case(Lcg, Write, Bytes16, Threads4),
            case(Lcg, Write, Bytes4, Threads8),
            case(Lcg, Write, Bytes8, Threads8),
            case(Lcg, Write, Bytes16, Threads8),
            case(Lcg, Write, Bytes4, Threads16),
            case(Lcg, Write, Bytes8, Threads16),
            case(Lcg, Write, Bytes16, Threads16),
            case(Lcg, Write, Bytes4, Threads32),
            case(Lcg, Write, Bytes8, Threads32),
            case(Lcg, Write, Bytes16, Threads32),
            case(Lcg, CompareAndSwap, Bytes4, Threads1),
            case(Lcg, CompareAndSwap, Bytes8, Threads1),
            case(Lcg, CompareAndSwap, Bytes4, Threads2),
            case(Lcg, CompareAndSwap, Bytes8, Threads2),
            case(Lcg, CompareAndSwap, Bytes4, Threads4),
            case(Lcg, CompareAndSwap, Bytes8, Threads4),
            case(Lcg, CompareAndSwap, Bytes4, Threads8),
            case(Lcg, CompareAndSwap, Bytes8, Threads8),
            case(Lcg, CompareAndSwap, Bytes4, Threads16),
            case(Lcg, CompareAndSwap, Bytes8, Threads16),
            case(Lcg, CompareAndSwap, Bytes4, Threads32),
            case(Lcg, CompareAndSwap, Bytes8, Threads32)
        );

        let function_name = if let Some(n) = function_name {
            n
        } else {
            return None;
        };

        let c_name =
            CString::new(function_name).expect("Failed to convert Rust string into C string");
        let function = module
            .get_function(&c_name)
            .expect(format!("Failed to load the GPU function: {}", function_name).as_str());
        unsafe {
            launch!(
                function<<<mp.grid_size.0, mp.block_size.0, 0, stream>>>(
            mem.as_launchable_ptr(),
            (state.buffer_len * mem::size_of::<i32>()) / item_bytes as usize,
            state.warp_misalignment as u32,
            state.loop_length,
            state.target_cycles.0,
            memory_accesses_device.as_device_ptr(),
            measured_cycles.as_device_ptr()
            )
                   )
            .expect("Failed to run GPU kernel");
        }

        timer_end
            .record(&state.stream)
            .expect("Couldn't record CUDA event");

        CurrentContext::synchronize().expect("Couldn't synchronize CUDA context");

        // Check if GPU is running in a throttled state
        #[cfg(not(target_arch = "aarch64"))]
        let throttle_reasons: Option<ThrottleReasons> = Some(
            nvml_device
                .current_throttle_reasons()
                .expect("Couldn't get current throttle reasons with NVML")
                .into(),
        );

        #[cfg(target_arch = "aarch64")]
        let throttle_reasons = None;

        let ms = timer_end
            .elapsed_time_f32(&timer_begin)
            .expect("Couldn't get elapsed time");
        let ns = ms as f64 * 10.0_f64.powf(6.0);

        let mut memory_accesses = 0;
        memory_accesses_device
            .copy_to(&mut memory_accesses)
            .expect("Couldn't transfer result from device");

        let mut cycles = 0;
        measured_cycles
            .copy_to(&mut cycles)
            .expect("Couldn't transfer result from device");

        Some((
            clock_rate_mhz,
            throttle_reasons,
            memory_accesses,
            Cycles(cycles),
            ns as u64,
        ))
    }
}
