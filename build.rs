use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    let include_path = Path::new("include");

    let out_dir = env::var("OUT_DIR").unwrap();

    let args = vec![
        "cudautils/operators.cu",
        "-fatbin",
        "-gencode",
        "arch=compute_30,code=sm_30", // Tesla K40
        "-gencode",
        "arch=compute_50,code=sm_50", // GTX 940M
        "-gencode",
        "arch=compute_52,code=sm_52", // GTX 980
        "-gencode",
        "arch=compute_61,code=sm_61", // GTX 1080
        "-gencode",
        "arch=compute_70,code=sm_70", // Tesla V100
        "-g",
        "-G",
        "-o",
    ];

    let output = Command::new("nvcc")
        .args(args.as_slice())
        .arg(&format!("{}/cudautils.fatbin", out_dir))
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
}
