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

use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let include_path = Path::new("include");
    let out_dir = env::var("OUT_DIR").unwrap();
    let cpp_compiler = env::var("CXX");

    println!("cargo:rerun-if-env-changed=CXX");

    // List of include files for Cargo build script to check if recompile is needed
    let include_files = vec![
        "include/cpu_clock.h",
        "include/cuda_clock.h",
        "include/cuda_vector.h",
        "include/ppc_intrinsics.h",
        "include/timer.hpp",
    ];

    // Add CUDA utils
    let cuda_lib_file = format!("{}/cudautils.fatbin", out_dir);
    let cuda_files = vec![
        "cudautils/memory_bandwidth.cu",
        "cudautils/memory_latency.cu",
        "cudautils/tlb_latency.cu",
        "cudautils/cuda_clock.cu",
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

    println!(
        "cargo:rustc-env=CUDAUTILS_PATH={}/cudautils.fatbin",
        out_dir
    );
    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

    let cpp_files = vec![
        "cpputils/memory_bandwidth.cpp",
        "cpputils/memory_latency.cpp",
    ];

    cc::Build::new()
        .include(include_path)
        .cpp(true)
        // Note: Disable to prevent linking twice with above CUDA
        // .cpp(true)
        // Note: -march not supported by GCC-7 on Power9, use -mcpu instead
        .flag("-std=c++11")
        .flag("-fopenmp")
        .flag_if_supported("-mcpu=native")
        .flag_if_supported("-march=native")
        .flag("-mtune=native")
        // .flag("-msse4.1") // Note: GCC sometimes detects incorrect microarch
        .pic(true)
        // .file("cpputils/numa_utils.cpp")
        .files(&cpp_files)
        .compile("libcpputils.a");

    // Link libatomic for 128-bit compare-and-exchange on x86_64
    println!("cargo:rustc-link-lib=atomic");

    vec!["include/", "cpputils/", "cudautils/"]
        .iter()
        .chain(include_files.iter())
        .chain(cuda_files.iter())
        .chain(cpp_files.iter())
        .for_each(|file| println!("cargo:rerun-if-changed={}", file));
}
