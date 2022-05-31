Setup Guide
===========

Cool that you're interested in setting up Project Triton! The setup works as
follows.

## Platforms

The following platforms are currently tested:

 * `x86_64-unknown-linux-gnu`
 * `powerpc64le-unknown-linux-gnu`

We try to also support `aarch64-unknown-linux-gnu` on the Nvidia Jetson Nano
platform. However, we don't regularly test on Arm. This can lead to breakage
because the CUDA NVTX and NVML libraries are not available and require custom
workarounds.

### Interconnect Limitations

NVLink between a CPU and a discrete GPU is currently only available on IBM
POWER platforms. However, Project Triton also runs on integrated GPUs and
discrete GPUs connected by PCI-e.

On these GPUs, Project Triton does not support features that require
cache-coherence.  Notable examples are GPU access to pageable memory (i.e,
non-pinned), CPU-GPU cooperative exectution, and memory allocations distributed
over multiple NUMA nodes.

For example, the Triton join currently requires NUMA-distributed memory
allocations and does not work with PCI-e.

### Hardware Monitoring Limitations

The Nvidia Management Library (NVML) does not support ARM64 CPUs, and is
automatically disabled by Project Triton. Thus, reading out the dynamic GPU
clock rate is not supported on ARM64. Instead, Project Triton statically
determines the GPU clock rate via CUDA.

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

Ensure that a recent version of Rust (version 60.0 or newer) is installed on
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

Project Triton locks memory with `mlock` to prevent swapping to disk. This requires
permissions to be setup for the user:
```sh
sudo cat << EOF >> /etc/security/limits.conf
username		soft	memlock		unlimited
username		hard	memlock		unlimited
EOF
```

Now, log out and log in again.

### Project Triton

When all dependencies are installed, Project Triton can be built using Cargo:
```sh
cargo build --release
```

## Usage

The easiest way to launch Project Triton commands is through `cargo run`. Some
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
