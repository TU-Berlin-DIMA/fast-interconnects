/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019-2021, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

//! # The SQL Operator Library
//!
//! `sql-ops` is a collection of SQL operators and building blocks for CPUs and
//! GPUs. Currently it includes the operators:
//!
//! - Hash join (no-partitioning and radix-partitioned)
//! - Radix partition
//! - Prefix scan (exclusive)
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

static mut MODULE_OWNER: Option<Module> = None;
static MODULE: Lazy<&'static Module> = Lazy::new(|| {
    let module_path = CString::new(env!("CUDAUTILS_PATH"))
        .expect("Failed to load CUDA module, check your CUDAUTILS_PATH");
    let module = Module::load_from_file(&module_path).expect("Failed to load CUDA module");

    unsafe { MODULE_OWNER.get_or_insert(module) }
});
