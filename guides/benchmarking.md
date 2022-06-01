Benchmarking Guide
==================

Correctly benchmarking the joins requires the right parameterization. Here are
some hints on what to look out for.

## Scripts

We provide scripts to benchmark our join implementations:

 * [CPU-partitioned radix
   join](../scripts/benchmark_cpu-partitioned_radix_join.py)

 * [GPU no-partitioning join](../scripts/benchmark_gpu_no-partitioning_join.py)

 * [GPU radix join and the Triton join](../scripts/benchmark_gpu_radix_join.py)

We also provide scripts to benchmark our partitioning algorithm implementations:

 * [GPU radix partitioning
   algorithms](../scripts/benchmark_gpu_radix_partitioning_algorithms.py)

 * [CPU radix partitioning
   algorithms](../scripts/benchmark_cpu_radix_partitioning_algorithms.py)

Finally, we provide scripts to analyze the hardware:

 * [Memory access granularity and
   patterns](../scripts/memory_access_granularity.py)

 * [TLB latency](../scripts/tlb_latency.py)

There are many more tools available for which we don't provide scripts, but we
always provide `--help`. The tools, especially the microbenchmarks, can be
parameterized to explore different aspects of the hardware. We encourage you to
experiment!

## Measurement Results

All tools output measurements as CSV files. To get hold of the CSV, add `--csv
output.csv` as a commandline parameter.

Our tools measure time in nanoseconds. We do our best to isolate the time taken
by different parts of the program (e.g., the prefix sum, the first partitioning
pass, and the GPU join pipeline).

To get the join throughput, we take the input sizes and divide by the sum of
the join phases. E.g., to get giga-tuples per second:

```python
(num_relations * num_tuples) / (prefix_sum_ns + partitioning_ns + join_ns)
```

The time to allocate memory is always reported separately. We don't consider
memory allocation as part of the join, because we would expect a real system to
reuse allocations (e.g., by using a memory pool or an efficient malloc
implementation). Instead, we focus on producing consistent measurements.

To achieve consistent measurements, we do the following:

 * Measuring time with CUDA events instead of host timers. For CPU tasks, we
   measure time with the Linux monotonic `clock_gettime` clock.

 * Lock memory with `mlock`. This also prefaults pages and prevents paging to
   disk. However, it takes the Linux kernel additional time to do this.

 * Using HugeTLB pages instead of transparent huge pages. We have witnessed
   that the transparent huge page allocator sometimes returns small pages,
   despite sufficient huge pages being available.

 * Early-boot page reservation. GPU TLBs compress their TLB entries if pages
   are physically contiguous. This leads to deteriorating performance when huge
   pages are fragmented. Unfortunately, Linux is not aware of this issue, and
   thus we resort to reserving pages and occasionally rebooting to defrag the
   pages. Reach out to us if you have a better solution for allocating
   contiguous huge pages in userspace.

 * Explicit NUMA allocation, instead of relying on a first-touch allocation
   policy. Libraries such as `numa.h` internally call the `mmap` and then
   `mbind` syscalls, which is slower than calling malloc.  Our `numa-gpu`
   library does the same, but adds some extra configuration options.

 * Pinning threads to CPU cores for CPU-based tasks, which has a big effect on
   short-running benchmarks.

 * Pinning the main program thread to the GPU's NUMA-affine CPU socket. This
   increases the measured GPU performance of latency measurements and
   short-running tasks.

 * Setting the GPU clock frequency. This is only necessary for the memory
   latency, TLB latency, and random memory access bandwidth microbenchmarks,
   because these use clock cycle counters. We refrain from setting the
   frequency in join benchmarks, as it complicates the setup and doesn't have a
   big effect.

## NUMA Locality

The big-ticket item is setting the right NUMA node. There are two nodes to be
aware of: the GPU memory node, and the CPU node to which the GPU has an
affinity. NVLink exposes GPU memory to Linux as a NUMA node, but not for PCI-e
GPUs. In contrast, the affine CPU node is relevant for both NVLink and PCI-e
GPUs.

The affine CPU node for each GPU can be determined with:

```sh
nvidia-smi topo --matrix
```

The NUMA node of the GPU memory can be determined with:

```sh
./scripts/gpu_numa_node.sh
```

The machine's NUMA topology can be visualized with:

```sh
sudo apt install hwloc
lstopo machine_topology.png
```

These NUMA nodes should be configured for both the base relations (e.g.,
`--inner-rel-location` and `--outer-rel-location`) as well as the partitions
(e.g., `--partitions-location`). To have an effect, these settings require a
NUMA-aware memory allocator (e.g. `--rel-mem-type numa` and
`--partitions-mem-type numa`).

## CPU Threads and Affinity

By default, CPU tasks use only a single thread. Setting a higher number of
threads and pinning each thread to a core leads to higher performance.

Setting the number of threads is done with, e.g., `--threads 16`.

Pinning threads to cores works by creating an affinity file and providing that
as an input:

```sh
echo "0 8 16 24 32 40 48 56 4 12 20 28 36 44 52 60" > cpu_affinity.txt
cargo run --release --package radix-join -- --cpu-affinity cpu_affinity.txt --threads ...
```

The CPU core IDs and their mapping to NUMA nodes can be determined with:

```sh
numactl --hardware
```

We provide a [detailed guide for tuning a POWER9
CPU](./power9.md#cpu-core-affinity-tuning).

## GPU Thread Blocks and Thread Block Size

The number of thread blocks is automatically set to the number of streaming
multiprocessors, and the thread block size is hard-coded to a good parameter
per kernel. Thus, generally these don't need to be configured.

However, it's possible to explicitly set the number of thread blocks with
`--grid-size`.

## Pinned Memory

PCI-e GPUs require pinned memory for zero-copy transfers to work.

The regular CUDA pinned allocator can be invoked with the `pinned` memory type
(e.g., `--rel-mem-type Pinned`). However, this allocator does not respect the
NUMA node setting.

Instead, it's better to use the `NumaPinned` memory type, which uses a
NUMA-capable allocator and then pins the memory with `cudaHostRegister`.

For NVLink GPUs, pinning is not necessary and it's recommended to use the
`Numa` memory type instead.

## Huge Pages

Random memory access patterns cause TLB misses when the accessed data structure
exceeds the TLB range. Huge pages help to reduce TLB misses.

The page type can be set with the `--page-type` parameter. It's recommended to
preallocate HugeTLB pages and use them with `--page-type Huge2MB`.

Configuring transparent huge pages with `TransparentHuge` or explicitly using
only regular "small" pages with `Small` is also possible.

We provide a [detailed guide for huge
pages](./huge_pages.md#general-huge-pages-guide).

## Radix Bits

Tuning the number of radix bits is important for radix joins. The best setting
depends on the hardware architecture, the inner relation size, and the hashing
scheme.

The radix bits can be set with `--radix-bits {1st pass},{2nd pass}`.

Tuning the radix bits generally requires a grid search over the first and
second pass for each inner relation size. However, in our experience,
performance is more sensitive to the first pass. Thus, we search for a good
second pass configuration and then calculate the minimum number of bits for the
first pass in our benchmark scripts.

## Hierarchical Radix Partitioning Buffer Size

The hierarchical partitioning algorithm uses a GPU device memory buffer, which
can be configured with `--dmem-buffer-size`.

The parameter determines the buffer size per partition and per thread block.
Thus, a higher number of radix bits increases the total memory allocated.

Typically, small buffers lead to more frequent flushes, which causes overhead.
Small buffers also cannot amortize TLB misses that occur for large data sizes.
Conversely, large buffers take time to fill and lead to "pipelining overhead".
Thus, the optimal size varies.

We found that 8 KiB is a good compromise for NVLink 2.0.
