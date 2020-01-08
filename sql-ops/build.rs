/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Number of GPU shared memory banks
    let log2_num_banks = "5";

    let include_path = Path::new("include");

    let out_dir = env::var("OUT_DIR").unwrap();

    #[cfg(target_arch = "aarch64")]
    let cache_line_size = 64;
    #[cfg(target_arch = "x86_64")]
    let cache_line_size = 64;
    #[cfg(target_arch = "powerpc64")]
    let cache_line_size = 128;

    // Add CUDA utils
    let cuda_lib_file = format!("{}/cudautils.fatbin", out_dir);
    let cuda_files = vec![
        "cudautils/no_partitioning_join.cu",
        "cudautils/radix_partition.cu",
    ];
    let num_banks_arg = format!("-DLOG2_NUM_BANKS={}", log2_num_banks);
    let nvcc_build_args = vec![
        num_banks_arg.as_str(),
        "-ccbin",
        "/usr/bin/g++-7",
        "--device-c",
        "-std=c++11",
        "--output-directory",
        &out_dir,
    ];
    let nvcc_link_args = vec!["--device-link", "-fatbin", "--output-file", &cuda_lib_file];

    // For gencodes, see: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    let gpu_archs = vec![
        "-gencode",
        "arch=compute_30,code=sm_30", // Tesla K40
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

    let output = Command::new("nvcc")
        .args(cuda_files.as_slice())
        .args(nvcc_build_args.as_slice())
        .args(gpu_archs.as_slice())
        .arg(nvcc_include)
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
        .into_iter()
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

    println!("cargo:rustc-env=LOG2_NUM_BANKS={}", log2_num_banks);
    println!(
        "cargo:rustc-env=CUDAUTILS_PATH={}/cudautils.fatbin",
        out_dir
    );
    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    // Add CPP utils
    cc::Build::new()
        .compiler("gcc-8")
        .include(include_path)
        .cpp(true)
        // Note: -march not supported by GCC-7 on Power9, use -mcpu instead
        .flag("-std=c++11")
        .debug(true)
        .flag_if_supported("-mcpu=native")
        .flag_if_supported("-march=native")
        .flag("-mtune=native")
        // .flag("-fopenmp")
        // .flag("-lnuma")
        .define("CACHE_LINE_SIZE", cache_line_size.to_string().as_str())
        // Note: Enables x86 intrinsic translations on POWER9
        // See also "Linux on Power Porting Guide - Vector Intrinsics"
        .define("NO_WARN_X86_INTRINSICS", None)
        .pic(true)
        .file("cpputils/no_partitioning_join.cpp")
        .file("cpputils/radix_partition.cpp")
        .compile("libcpputils.a");
}
