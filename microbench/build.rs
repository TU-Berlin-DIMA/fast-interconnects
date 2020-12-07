/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2020 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    const TLB_DATA_POINTS: &str = "256";

    let include_path = Path::new("include");
    let out_dir = env::var("OUT_DIR").unwrap();
    let cpp_compiler = env::var("CXX");

    // Add CUDA utils
    let cuda_lib_file = format!("{}/cudautils.fatbin", out_dir);
    let cuda_files = vec!["cudautils/tlb_latency.cu", "cudautils/cuda_clock.cu"];
    let tlb_data_points_arg = format!("-DTLB_DATA_POINTS={}U", TLB_DATA_POINTS);
    let nvcc_build_args = vec![
        tlb_data_points_arg.as_str(),
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

    let output = Command::new("nvcc")
        .args(cuda_files.as_slice())
        .args(nvcc_host_compiler_args.as_slice())
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

    println!("cargo:rustc-env=TLB_DATA_POINTS={}", TLB_DATA_POINTS);
    println!(
        "cargo:rustc-env=CUDAUTILS_PATH={}/cudautils.fatbin",
        out_dir
    );
    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    // Add remaining CUDA utils with CUDA trampoline-style functions
    // FIXME: Transition to calling CUDA kernels directly from Rust
    cc::Build::new()
        .include(include_path)
        .cuda(true)
        .flag("-std=c++11")
        .flag("-cudart=shared")
        .pic(true)
        .flag("-gencode")
        .flag("arch=compute_50,code=sm_50") // GTX 940M
        .flag("-gencode")
        .flag("arch=compute_52,code=sm_52") // GTX 980
        .flag("-gencode")
        .flag("arch=compute_53,code=sm_53") // Jetson Nano
        .flag("-gencode")
        .flag("arch=compute_61,code=sm_61") // GTX 1080
        .flag("-gencode")
        .flag("arch=compute_70,code=sm_70") // Tesla V100
        .file("cudautils/memory_bandwidth.cu")
        .file("cudautils/memory_latency.cu")
        .debug(false) // Debug enabled slows down mem latency by 10x
        .compile("libcudautils.a");

    cc::Build::new()
        .include(include_path)
        // Note: Disable to prevent linking twice with above CUDA
        // .cpp(true)
        // Note: -march not supported by GCC-7 on Power9, use -mcpu instead
        .flag("-std=c++11")
        .flag("-fopenmp")
        .flag_if_supported("-mcpu=native")
        .flag_if_supported("-march=native")
        .flag("-mtune=native")
        // .flag("-lnuma")
        .pic(true)
        // .file("cpputils/numa_utils.cpp")
        .file("cpputils/memory_bandwidth.cpp")
        .file("cpputils/memory_latency.cpp")
        .compile("libcpputils.a");
}
