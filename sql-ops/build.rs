/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use std::env;
use std::ffi::OsString;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let include_path = Path::new("include");

    let out_dir = env::var("OUT_DIR").unwrap();
    let cpp_compiler = env::var("CXX");

    println!("cargo:rerun-if-env-changed=CXX");

    #[cfg(target_arch = "aarch64")]
    let cache_line_size: u32 = 64;
    #[cfg(target_arch = "x86_64")]
    let cache_line_size: u32 = 64;
    #[cfg(target_arch = "powerpc64")]
    let cache_line_size: u32 = 128;

    // Defines the alignment of each partition in bytes.
    //
    // Typically, alignment should be a multiple of the cache line size. Reasons for this size are:
    //
    // 1. Non-temporal store instructions
    // 2. Vector load and store intructions
    // 3. Coalesced loads and stores on GPUs
    let align_bytes: u32 = 128;

    // Defines the padding bytes between partitions.
    //
    // Padding is necessary for partitioning algorithms to align writes. Aligned writes have fixed
    // length and may overwrite the padding space in front of their partition.  For this reason,
    // also the first partition includes padding in front.
    //
    // # Invariants
    //
    // * The padding length must be equal to or larger than the alignment:
    //   padding_bytes >= align_bytes
    let padding_bytes: u32 = 128;

    // Number of GPU shared memory banks
    //
    // * Nvidia Volta architecture: 32 banks (log2: 5)
    let log2_num_banks: u32 = 5;

    // The number of tuples processed by each thread. This is a tuning parameter for LA-SWWC GPU
    // partitioning variant.
    let laswwc_tuples_per_thread = 5;

    // Generate constants files for Rust and C++
    let cpp_constants_path = Path::new(&out_dir).join("constants.h");
    let rust_constants_path = Path::new(&out_dir).join("constants.rs");
    let mut cpp_constants = File::create(&cpp_constants_path).unwrap();
    let mut rust_constants = File::create(&rust_constants_path).unwrap();

    cpp_constants
        .write_all(
            format!(
                "
    #define CACHE_LINE_SIZE {}U\n\
    #define ALIGN_BYTES {}U\n\
    #define LOG2_NUM_BANKS {}U\n\
    #define LASWWC_TUPLES_PER_THREAD {}U\n\
    ",
                cache_line_size, align_bytes, log2_num_banks, laswwc_tuples_per_thread,
            )
            .as_bytes(),
        )
        .unwrap();

    rust_constants
        .write_all(
            format!(
                "
    pub const CACHE_LINE_SIZE: u32 = {};\n\
    pub const ALIGN_BYTES: u32 = {};\n\
    pub const PADDING_BYTES: u32 = {};\n\
    pub const LOG2_NUM_BANKS: u32 = {};\n\
    ",
                cache_line_size, align_bytes, padding_bytes, log2_num_banks,
            )
            .as_bytes(),
        )
        .unwrap();

    // List of include files for Cargo build script to check if recompile is needed
    let include_files = vec![
        "include/gpu_common.h",
        "include/gpu_radix_partition.h",
        "include/ppc_intrinsics.h",
        "include/prefix_scan.h",
        "include/prefix_scan_state.h",
        "include/ptx_memory.h",
    ];

    // Add CUDA utils
    let cuda_lib_file = format!("{}/cudautils.fatbin", out_dir);
    let cuda_files = vec![
        "cudautils/gpu_common.cu",
        "cudautils/no_partitioning_join.cu",
        "cudautils/radix_join.cu",
        "cudautils/radix_partition.cu",
    ];
    let nvcc_build_args = vec![
        "-rdc=true",
        "--device-c",
        "-std=c++14",
        "--output-directory",
        &out_dir,
    ];
    let nvcc_link_args = vec!["--device-link", "-fatbin", "--output-file", &cuda_lib_file];
    let nvcc_host_compiler_args: Vec<_> = cpp_compiler
        .as_ref()
        .map_or_else(|_| Vec::new(), |cxx| ["-ccbin", cxx.as_str()].into());

    // For gencodes, see: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    let gpu_archs = vec![
        "-gencode",
        "arch=compute_50,code=sm_50", // GTX 940M
        "-gencode",
        "arch=compute_52,code=sm_52", // GTX 980
        "-gencode",
        "arch=compute_53,code=sm_53", // Jetson Nano
        "-gencode",
        "arch=compute_61,code=sm_61", // GTX 1080
        "-gencode",
        "arch=compute_70,code=sm_70", // Tesla V100
    ];
    let nvcc_include = {
        let mut s = OsString::from("-I ");
        s.push(include_path.as_os_str());
        s
    };
    let constants_include = {
        let mut s = OsString::from("-I ");
        s.push(&out_dir);
        s
    };

    let output = Command::new("nvcc")
        .args(cuda_files.as_slice())
        .args(nvcc_host_compiler_args.as_slice())
        .args(nvcc_build_args.as_slice())
        .args(gpu_archs.as_slice())
        .arg(nvcc_include)
        .arg(constants_include)
        .output()
        .expect("Couldn't execute nvcc");

    if !output.status.success() {
        eprintln!("status: {}", output.status);
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!();
    }

    let cuda_object_files: Vec<_> = cuda_files
        .as_slice()
        .iter()
        .map(|f| {
            let p = Path::new(f);
            let mut obj = PathBuf::new();
            obj.push(&out_dir);
            obj.push(p.file_stem().unwrap());
            obj.set_extension("o");
            obj
        })
        .collect();

    let output = Command::new("nvcc")
        .args(cuda_object_files.as_slice())
        .args(nvcc_link_args.as_slice())
        .args(gpu_archs.as_slice())
        .output()
        .expect("Couldn't execute nvcc");

    if !output.status.success() {
        eprintln!("status: {}", output.status);
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!();
    }

    println!(
        "cargo:rustc-env=CUDAUTILS_PATH={}/cudautils.fatbin",
        out_dir
    );
    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    // CPP files
    let cpp_files = vec![
        "cpputils/no_partitioning_join.cpp",
        "cpputils/radix_partition.cpp",
    ];

    // Add CPP utils
    cc::Build::new()
        .include(include_path)
        .include(&out_dir)
        .cpp(true)
        // Note: -march not supported by GCC-7 on Power9, use -mcpu instead
        .flag("-std=c++14")
        .debug(true)
        .flag_if_supported("-mcpu=native")
        .flag_if_supported("-march=native")
        .flag("-mtune=native")
        // .flag("-fopenmp")
        // .flag("-lnuma")
        // Note: Enables x86 intrinsic translations on POWER9
        // See also "Linux on Power Porting Guide - Vector Intrinsics"
        .define("NO_WARN_X86_INTRINSICS", None)
        .pic(true)
        .files(&cpp_files)
        .compile("libcpputils.a");

    vec!["include/", "cpputils/", "cudautils/"]
        .iter()
        .chain(include_files.iter())
        .chain(cuda_files.iter())
        .chain(cpp_files.iter())
        .for_each(|file| println!("cargo:rerun-if-changed={}", file));
}
