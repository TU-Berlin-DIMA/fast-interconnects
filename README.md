numa-gpu
========

## What is numa-gpu?

numa-gpu is on-going research into using Nvidia's new NVLink 2.0 technology to
accelerate in-memory data analytics. In contrast to PCI-e and NVLink 1.0,
NVLink 2.0 provides hardware address translation services that, in effect, make
the GPU a NUMA node (Non-Uniform Memory Access). This new feature will enable
higher performance, but requires a re-design of the database.

In this project, we rethink database design to take full advantage of NVLink 2.0.
Especially out-of-core queries, i.e., queries that use data sets larger than
GPU memory, will see speed-ups compared to previous interconnect technologies.

## Platforms

The following platforms are currently tested:

 * `x86_64-unknown-linux-gnu`
 * `powerpc64le-unknown-linux-gnu`
 * `aarch64-unknown-linux-gnu`

Note that NVLink is only available on IBM POWER platforms.

## Installation

### CUDA

Ensure that CUDA 10.1 or greater is installed and in the path. If CUDA is
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

### Rust

Ensure that a recent version of Rust is installed on your system. You can install
an up-to-date version of Rust using [rustup](https://rustup.rs):
```sh
curl https://sh.rustup.rs -sSf | sh
```

### Linux settings

numa-gpu locks memory with `mlock` to prevent swapping to disk. This requires
permissions to be setup for the user:
```sh
sudo cat << EOF >> /etc/security/limits.conf
username		soft	memlock		unlimited
username		hard	memlock		unlimited
EOF
```

Now, log out and log in again.

### numa-gpu

When all dependencies are installed, numa-gpu can be built using Cargo:
```sh
cargo build --release
```

Note that on ARM64 CPUs, Nvidia Management Library (NVML) is not supported. Thus,
NVML must be disabled:
```sh
cargo build --release --no-default-features
```

## Usage

numa-gpu provides several applications and libraries:
 * `datagen` is a application and library to generate data with data
   distributions.
 * `hashjoin` is an application to execute and benchmark hash joins on CPUs and
   GPUs.
 * `microbench` is a collection of microbenchmarks for CPUs, GPUs, and GPU
   interconnects.
 * `numa-gpu` is a library with abstractions and tools to program GPUs with and
   without NVLink.
 * `sql-ops` is a library that implements SQL operators.

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
    --device-type gpu                       \
    --device-id 0                           \
    --warpmul-lower 1                       \
    --warpmul-upper 2                       \
    --oversub-lower 4                       \
    --oversub-upper 32                      \
    --mem-type device                       \
    --repeat 10                             \
    --size 128
```

### hashjoin

`hashjoin` currently implements a no-partitioning hash join. It supports perfect
hashing and linear probing schemes. Linear probing uses a multiply-shift hash
function.

`hashjoin` supports different execution method, memory location and type,
tuple size, and data set combinations. For example:

```sh
cargo run --release --package hashjoin --    \
  --execution-method gpustream               \
  --device-id 0                              \
  --hash-table-mem-type device               \
  --rel-mem-type numa                        \
  --inner-rel-location 0                     \
  --outer-rel-location 0                     \
  --tuple-bytes bytes16                      \
  --hashing-scheme perfect                   \
  --data-set Custom                          \
  --inner-rel-tuples `bc <<< "10^6"`         \
  --outer-rel-tuples `bc <<< "10^6"`         \
  --repeat 10                                \
  --csv hashjoin.csv
```

## FAQ

### numa-gpu fails with: SIGILL (Illegal instruction)

Most likely, you've attempted to run a GPU program with the wrong memory type
(on a non-NVLink machine). Try using `--mem-type device`.

`hashjoin` also
supports `--execution-method gpustream` to stream blocks of `system` and `numa`
memory to the GPU. Alternatively, `--execution-method unified` can be used
togther with `--rel-mem-type unified`.

### numa-gpu fails with: CudaError(NoBinaryForGpu)

Your device isn't set as a build target. First, find the correct gencode for your
device in [Arnon Shimoni's helpful blog post][gencodes].
Then, add the gencode to the gencode lists in `sql-ops/build.rs` and
`microbench/build.rs`.

### numa-gpu fails with: mlock() failed with ENOMEM

You don't have permission to lock memory with `mlock`. Setup permissions for your
user with:
```sh
sudo cat << EOF >> /etc/security/limits.conf
username		soft	memlock		unlimited
username		hard	memlock		unlimited
EOF
```

Now, log out and log in again.

### numa-gpu fails with: Error: Device memory not supported in this context!

You've attempted to run a CPU program with device memory. Try using
`--mem-type system` or `--mem-type numa`.

ATTENTION NVLink users: The CPU does *not* have access to `--mem-type device`!
Instead, use `--mem-type numa --mem-location X`, where `X` is the GPU's NUMA
node identifier. You can find the correct ID with:
```sh
numactl --hardware
```

[gencodes]: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
