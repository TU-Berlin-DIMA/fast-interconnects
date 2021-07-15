<meta name="robots" content="noindex">

NUMA-GPU
========

## What is NUMA-GPU?

NUMA-GPU is an on-going research project investigating the use of a fast GPU
interconnect to speed-up data processing on GPUs. Fast interconnects such as
NVLink 2.0 provide GPUs with high-bandwidth, cache-coherenct access to main
memory. Thus, our goal is to unlock higher DBMS query performance with this new
class of hardware.

In this project, we rethink database design to take full advantage of NVLink 2.0.
Especially out-of-core queries, i.e., queries that use data sets larger than
GPU memory, will see speed-ups compared to previous interconnect technologies.

## Structure

NUMA-GPU provides the following applications and libraries:

 * `datagen` is a application and library to generate data with data
   distributions. It is used as a library, e.g., by `hashjoin` and
   `radix-join`.
 * `hashjoin` is an application to execute and benchmark hash joins on CPUs and
   GPUs.
 * `microbench` is a collection of microbenchmarks for CPUs, GPUs, and GPU
   interconnects.
 * `numa-gpu` is a library with abstractions and tools to program GPUs with and
   without NVLink.
 * `radix-join` is an application to execute and benchmark radix joins on CPUs
   and GPUs. The distinction from `hashjoin` enables a specialized API for
   radix joins.
 * `sql-ops` is a library that implements SQL operators. These are used by
   `hashjoin` and `radix-join`.
 * `tpch-bench` is an application to execute and benchmark TPC-H on CPUs and
   GPUs. Currently, Query 6 is implemented.

Detailed documentation is available by running:
```sh
cargo doc --document-private-items --no-deps --open
```

## Platforms

The following platforms are currently tested:

 * `x86_64-unknown-linux-gnu`
 * `powerpc64le-unknown-linux-gnu`
 * `aarch64-unknown-linux-gnu`

### Interconnect Limitations

NVLink between a CPU and a discrete GPU is currently only available on IBM
POWER platforms. However, NUMA-GPU also runs on integrated GPUs and discrete
GPUs connected by PCI-e.

On these GPUs, NUMA-GPU does not support features that require cache-coherence.
Notable examples are GPU access to pageable memory (i.e, non-pinned), CPU-GPU
cooperative exectution, and memory allocations distributed over multiple NUMA
nodes.

### Hardware Monitoring Limitations

The Nvidia Management Library (NVML) does not support ARM64 CPUs, and is
automatically disabled by NUMA-GPU. Thus, reading out the dynamic GPU clock
rate is not supported on ARM64. Instead, NUMA-GPU statically determines the GPU
clock rate via CUDA.

## Installation

### CUDA

Ensure that CUDA 10.2 or greater is installed and in the path. If CUDA is
installed at a custom path, the path must be included, e.g.:
```sh
export PATH="$PATH:/usr/local/cuda/bin"
export C_INCLUDE_PATH="$C_INCLUDE_PATH:/usr/local/cuda/include"
export LIBRARY_PATH="$LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64:/usr/local/cuda/extras/CUPTI/lib64"
```

These environment variables can be made permanent by including `PATH` and
`C_INCLUDE_PATH` in `$HOME/.profile`, and setting up the system linker with
```sh
sudo cat << EOF > /etc/ld.so.conf.d/cuda.conf
/usr/local/cuda/lib64
/usr/local/cuda/nvvm/lib64
/usr/local/cuda/extras/CUPTI/lib64
EOF

sudo ldconfig
```

### GNU C++ compiler

Ensure that G++ 8.3 or greater is installed and in the path. If this version is
not the default C++ compiler, the `CXX` environment variable must be set, e.g.:
```sh
export CXX=/usr/bin/g++-8
```

### Rust

Ensure that a recent version of Rust (version 51.0 or newer) is installed on
your system. You can install an up-to-date version of Rust using
[rustup](https://rustup.rs):
```sh
curl https://sh.rustup.rs -sSf | sh
```

Note that the performance-critical components (i.e., relational operators) are
written in C++ and CUDA. Rust is used to setup the benchmarks and call the
C++/CUDA functions.

The reason for our approach is that GCC and CUDA support more compiler
intrinsics, inline assembly, and are generally more mature frameworks. In
comparison, Rust is easier to refactor and test due to its strong type system.

### Linux settings

NUMA-GPU locks memory with `mlock` to prevent swapping to disk. This requires
permissions to be setup for the user:
```sh
sudo cat << EOF >> /etc/security/limits.conf
username		soft	memlock		unlimited
username		hard	memlock		unlimited
EOF
```

Now, log out and log in again.

### NUMA-GPU

When all dependencies are installed, NUMA-GPU can be built using Cargo:
```sh
cargo build --release
```

### Advanced Guides

Detailed setup guides are provided for the following topics:

 * [Huge Pages](./guides/huge_pages.md)

## Usage

The easiest way to launch NUMA-GPU commands is through `cargo run`. Some
examples are listed here.

### microbench

`microbench` provides memory bandwidth, memory latency, cudacopy (memcpy with
CUDA), and numacopy (parallel memcpy for testing NUMA interconnects) benchmarks.

For example, the memory benchmark supports a parameter grid search to find the
optimal parameters. It evaluates different memory access patterns
(sequential/coalesced, random) and memory access types (read, write,
compare-and-swap).

The bandwidth benchmark can be executed as follows:

```sh
cargo run --release --package microbench -- \
  --csv bandwidth.csv                       \
  bandwidth                                 \
    --device-type Gpu                       \
    --device-id 0                           \
    --grid-sizes 80,160                     \
    --block-sizes 128,256,512,1024          \
    --size 1024                             \
    --mem-type Device                       \
    --repeat 10
```

### hashjoin

`hashjoin` implements a no-partitioning hash join. It supports perfect hashing
and linear probing schemes. Linear probing hashes keys with a multiply-shift
hash function.

`hashjoin` supports different execution method, memory location and type, tuple
size, and data set combinations. For example:

```sh
cargo run --release --package hashjoin --    \
  --execution-method Gpu                     \
  --device-id 0                              \
  --hash-table-mem-type Device               \
  --rel-mem-type NumaLazyPinned              \
  --page-type TransparentHuge                \
  --inner-rel-location 0                     \
  --outer-rel-location 0                     \
  --tuple-bytes Bytes16                      \
  --hashing-scheme Perfect                   \
  --data-set Custom                          \
  --inner-rel-tuples `bc <<< "128 * 10^6"`   \
  --outer-rel-tuples `bc <<< "128 * 10^6"`   \
  --repeat 10                                \
  --csv hashjoin.csv
```

### radix-join

`radix-join` implements three types of radix join:
 * a textbook GPU radix join,
 * a heterogeneous CPU-GPU join that partitions data on the CPU,
 * and a new out-of-core join for GPUs with fast interconnects, called the
   "Triton join".

```sh
cargo run                                          \
  --release                                        \
  --package radix-join                             \
  --                                               \
  --execution-strategy GpuRadixJoinTwoPass         \
  --hashing-scheme BucketChaining                  \
  --histogram-algorithm CpuChunkedSimd             \
  --partition-algorithm HSSWWCv4                   \
  --partition-algorithm-2nd SSWWCv2                \
  --radix-bits 7,9                                 \
  --page-type TransparentHuge                      \
  --dmem-buffer-size 8                             \
  --threads 64                                     \
  --rel-mem-type NumaLazyPinned                    \
  --inner-rel-location 0                           \
  --outer-rel-location 0                           \
  --partitions-mem-type NumaLazyPinned             \
  --partitions-location 0                          \
  --data-set Custom                                \
  --inner-rel-tuples `bc <<< "128 * 10^6"`         \
  --outer-rel-tuples `bc <<< "128 * 10^6"`         \
  --tuple-bytes Bytes16                            \
  --repeat 10                                      \
  --csv radix-join.csv
```

*Important:* `radix-join` is a hardware-conscious join, and thus high
throughput requires "good" parameters. The above settings are intended for
demonstration only and *should be tuned to your machine as well as the data
size*!

## FAQ

### NUMA-GPU fails with: SIGILL (Illegal instruction)

Most likely, you've attempted to run a GPU program with the wrong memory type
(on a non-NVLink machine). Try using `--mem-type device`.

`hashjoin` also
supports `--execution-method gpustream` to stream blocks of `system` and `numa`
memory to the GPU. Alternatively, `--execution-method unified` can be used
togther with `--rel-mem-type unified`.

### NUMA-GPU fails with: SIGBUS (Misaligned address error)

This is usually an out-of-memory condition. Try the following:

 - If you're trying to use huge pages, double-check if you've configured huge
   pages correctly. At least one of these command should show a non-zero value:

   ```sh
   cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
   cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_overcommit_hugepages
   ```

 - If you're trying to benchmark very large data (i.e., near the memory limit
   of a NUMA node), then the OS might not have enough memory for you on the
   requested NUMA node.

   In this case, it helps to [reserve huge pages on early
   boot](guides/huge_pages.md), and then the program with `--page-type
   Huge2MB`.

 - If none of the above apply, you may have found a bug! In this case, let us
   know by opening an issue or pull request.

### NUMA-GPU fails with: CudaError(NoBinaryForGpu)

Your device isn't set as a build target. First, find the correct gencode for your
device in [Arnon Shimoni's helpful blog post][gencodes].
Then, add the gencode to the gencode lists in `sql-ops/build.rs` and
`microbench/build.rs`.

### NUMA-GPU fails with: mlock() failed with ENOMEM

You don't have permission to lock memory with `mlock`. Setup permissions for your
user with:
```sh
sudo cat << EOF >> /etc/security/limits.conf
username		soft	memlock		unlimited
username		hard	memlock		unlimited
EOF
```

Now, log out and log in again.

### NUMA-GPU fails with: Error: Device memory not supported in this context!

You've attempted to run a CPU program with device memory. Try using
`--mem-type system` or `--mem-type numa`.

ATTENTION NVLink users: The CPU does *not* have access to `--mem-type device`!
Instead, use `--mem-type numa --mem-location X`, where `X` is the GPU's NUMA
node identifier. You can find the correct ID with:
```sh
numactl --hardware
```

[gencodes]: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
