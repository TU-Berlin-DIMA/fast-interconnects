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

//! # The SQL Operator Library
//!
//! `sql-ops` is a collection of SQL operators and building blocks for CPUs and
//! GPUs. Currently it includes the operators:
//!
//! - Hash join (no-partitioning and radix-partitioned)
//! - Radix partition
//! - Prefix scan (exclusive)
//!
//! # Tuning parameters
//!
//! Several tuning parameters are defined as constant values. These affect
//! the performance and should be adjusted if necessary.
//!
//! The tuning parameters are set in the `build.rs` file, which exports them to Rust, C++, and
//! CUDA.
//!
//! ## CPU cacheline size
//!
//! `CPU_CACHE_LINE_SIZE` defines the number of bytes used for padding to prevent false sharing,
//! and SWWC radix partitioning buffers. The size is specific to the CPU architecture and set to
//! different values depending on the ISA:
//!
//! - aarch64: 64 bytes
//! - x86_64: 64 bytes
//! - powerpc64: 128 bytes
//!
//! ## GPU cacheline size
//!
//! `GPU_CACHE_LINE_SIZE` serves the same purpose as the CPU cacheline size, but is used in GPU
//! code paths. The size is set to 128 bytes, which is the size used by many Nvidia GPUs (e.g.,
//! Pascal, Volta, Ampere).
//!
//! ## Align bytes
//!
//! `ALIGN_BYTES` defines the alignment of partitions in bytes. This parameter is intended to
//! prevent cache conflict misses. It should be set to a multiple of the cacheline size.
//!
//! Furthermore, cacheline alignment is necessary for:
//!
//! - non-temporal store instructions
//! - vector load and store instructions
//! - perfectly aligned coalesced loads and stores on GPUs
//!
//! ## Padding bytes
//!
//! `PADDING_BYTES` defines the padding size between partitions.  Padding is necessary for
//! partitioning algorithms to align writes. Aligned writes have fixed length and may overwrite the
//! padding space in front of their partition.  For this reason, also the first partition includes
//! padding in front.
//!
//! If no padding is used, aligned writes incur a race condition between threads. Given two
//! partitions, a thread writing to the end of the first partition must write after a different
//! thread writing to the beginning of the second partition, because the written locations may
//! overlap due to aligning the second thread to `PADDING_BYTES`.
//!
//! ## Number of banks
//!
//! `LOG2_NUM_BANKS` defines the number of shared memory banks on GPUs. This parameter is used to
//! avoid bank conflicts.
//!
//! ## LA-SWWC tuples per thread
//!
//! `LASWWC_TUPLES_PER_THREAD` defines the number of tuples processed at a time per thread. More
//! tuples require more shared memory and more registers. Thus, the parameter should be tuned for
//! each GPU architecture.
//!
//! The Stehle and Jacobsen set the value to `3` for a Tesla P100 GPU in their work: [*A Memory
//! Bandwidth-Efficient Hybrid Radix Sort on GPUs*](http://doi.acm.org/10.1145/3035918.3064043). We
//! set the value to `5` for a Tesla V100 GPU.
//!
//! ## Bucket chaining entries
//!
//! `RADIX_JOIN_BUCKET_CHAINING_ENTRIES` defines the number of hash table entries used by the
//! bucket chaining scheme of the radix join.
//!
//! The value must be set to a power of two, and at least 1. No further constraints.
//!
//! # Library initialization
//!
//! GPU operators are compiled as a [CUDA `fatbinary` module][fatbin]. The
//! module must be loaded into the current context before using the
//! [`cuModuleLoad` driver function][cuModuleLoad] before the operator can start
//! executing. Module loading can take up to several hundred milliseconds.
//!
//! To avoid load the module each time an operator is executed, the `sql-ops`
//! library globally loads the module exactly once. The load is lazy and is
//! performed when a GPU operator is executed for the first time. Thus, later
//! executions of any GPU operator use the already-loaded module.
//!
//! **Important:** The CUDA context must be initialized before calling the
//! a GPU operator. *Destroying this context will also destroy the module!*
//!
//! This is usually not a problem in applications that initialize the context
//! once at the start of the program. However, in unit tests, a common pattern
//! is to initialize a context for each test case. Instead, tests should create
//! a singleton instance of the context that is only initialized once. See
//! `sql-ops/tests/test_gpu_radix_partition.rs` as an example.
//!
//! [fatbin]: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#fatbinaries
//! [cuModuleLoad]: https://docs.nvidia.com/cuda/archive/10.2/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3

pub mod error;
pub mod join;
pub mod partition;
pub mod prefix_scan;

use once_cell::sync::Lazy;
use rustacuda::module::Module;
use std::ffi::CString;

#[allow(dead_code)]
pub(crate) mod constants {
    include!(concat!(env!("OUT_DIR"), "/constants.rs"));
}

// Export cache line constants
pub use constants::CACHE_LINE_SIZE as CPU_CACHE_LINE_SIZE;
pub use constants::GPU_CACHE_LINE_SIZE;

static mut MODULE_OWNER: Option<Module> = None;
static MODULE: Lazy<&'static Module> = Lazy::new(|| {
    let module_path = CString::new(env!("CUDAUTILS_PATH"))
        .expect("Failed to load CUDA module, check your CUDAUTILS_PATH");
    let module = Module::load_from_file(&module_path).expect("Failed to load CUDA module");

    unsafe { MODULE_OWNER.get_or_insert(module) }
});
