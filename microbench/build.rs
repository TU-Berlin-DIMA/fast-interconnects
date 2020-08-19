use cc;

use std::env;
use std::path::Path;

fn main() {
    let include_path = Path::new("include");

    // For gencodes, see: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    env::set_var("CXX", "g++-7");
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
        .file("cudautils/sync_latency.cu")
        .file("cudautils/memory_bandwidth.cu")
        .file("cudautils/memory_latency.cu")
        .debug(false) // Debug enabled slows down mem latency by 10x
        .compile("libcudautils.a");

    env::remove_var("CXX");
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
        .file("cpputils/sync_latency.cpp")
        .file("cpputils/memory_bandwidth.cpp")
        .file("cpputils/memory_latency.cpp")
        .compile("libcpputils.a");

    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    // println!("cargo:rustc-link-lib=gomp");
}
