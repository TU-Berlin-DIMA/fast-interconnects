Non-Uniform Memory Access
=========================

The main reasons why your GPU application should be NUMA-aware are:

 - Better measurement accuracy when using CUDA Events.

 - Higher bandwidth when accessing CPU memory.

 - Allocate memory with the same API for CPU memory and GPU memory.

These points are described in this document, along with hints on the APIs
exposed by Linux and the Nvidia GPU driver. The APIs for GPU memory allocation
are specific to NVLink GPUs and are not available on PCI-e machines.

## Measurement accuracy

In our measurements, we observed that the runtimes returned by CUDA Events
(i.e., `cuEventElapsedTime()`) were about 10-20% slower than those returned by
`nvprof`. We first briefly describe our observations, then outline a fix, and
summarize the solution.

This inaccuracy occurs only when multiple tasks are enqueued on a CUDA Stream,
but not when executing only a single kernel.  Affected are pipelines that
overlap transfers with computations, as well as pipelines consisting of
multiple kernels without any interconnect transfers.

Accuracy improves to the same level as `nvprof` when the application is
NUMA-localized. We NUMA-localized the application by configuring the CPU mask
and the memory mask of the main thread to the NUMA node that is closest to the
GPU. The reason why the problem occurs is unclear, as Nvidia does not publish
details on how tasks and CUDA Events are scheduled on CUDA Streams.

The steps to solve the issue are as follows:

 1. Find the NUMA affinity of the GPU by parsing
    `/sys/bus/pci/devices/$PCI_ID/numa_node`.

 2. Set the CPU affinity in `main()` with `sched_setaffinity()` from glibc or
    `numa_run_on_node()` from libnuma.

 3. Set the memory affinity in `main()` with `mbind()` from glibc or
    `numa_set_preferred()` from libnuma.

## CPU memory bandwidth

The default memory allocation policy of Linux interleaves pages across all NUMA
nodes (excluding GPU memory on AC922 systems) in a round-robin pattern.
However, current CPU NUMA interconnects (e.g., IBM X-Bus and Intel UPI) have a
lower bandwidth than fast GPU interconnects (e.g., NVLink). For consistent
measurements, the main memory allocations accessed by the GPU should be
NUMA-localized to the NUMA node closest to the GPU.

GPUs connected via fast interconnects can access pageable system memory. In
principle, you can choose to allocate memory with your favorite Linux memory
allocator.

To consistently run benchmarks, we followed these steps:

 1. Huge pages can be allocated with `mmap()` on Linux. This is mostly useful
    to allocate HugeTLBFS pages. See [the guide on huge pages](./huge_pages.md)
    for more information.

 2. Transparent huge pages can be enable or disabled with `madvise()`.

 3. NUMA affinity on mmap'ed pages can be configured with `mbind()`. In our
    research papers, we have used `mbind` to interleave pages in custom
    patterns (e.g., to build a "hybrid hash table").

 4. To prevent the OS from paging to disk, `mlock()` the allocated memory.
    `mlock()` also prefaults pages, meaning that physical memory backs each
    virtual address.

 5. *For PCI-e:* Pinning the mmap'ed pages can be done by calling
    `cuMemHostRegister()`. In this case, `mlock()`-ing pages is not necessary.
    Pinning memory has no performance effect on NVLink systems.

These steps must be follow in the given order. For an example, see the
[`numa_gpu::runtime::numa::NumaMemory`
struct](../numa-gpu/src/runtime/numa.rs).

## GPU memory allocation

GPU programming tutorials typically point to the `cuMemAlloc()` function to
allocate GPU memory. However, with fast interconnects, the GPU memory is
exposed to Linux as a NUMA node. Thus, it's possible to allocate GPU memory
with your favorite NUMA-aware memory allocator by specifying the GPU's NUMA
identifier.

The steps to allocate GPU memory with a system allocator are:

 1. Get the GPU's PCI ID by retrieving `CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID`,
    `CU_DEVICE_ATTRIBUTE_PCI_BUS_ID`, and `CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID`
    from `cuDeviceGetAttribute()`. The PCI function ID cannot be retrieved, but
    is typically `0`. These integers must be formatted as a string, e.g.,
    `0004:04:00.0`.

    In Rust:
    ```Rust
    let pci_id = format!(
       "{:04x}:{:02x}:{:02x}.{:1x}",
         pci_domain_id, pci_bus_id, pci_device_id, pci_function_id
    );
    ```

 2. Get the GPU's NUMA ID by parsing the
    `/proc/driver/nvidia/gpus/$PCI_ID/numa_status` file. Note that this is only
    available on IBM AC922 machines (and in future possibly other
    NVLink-capable machines).

 3. Find the amount of available GPU memory by parsing the
    `/sys/devices/system/node/node{$GPU_NODE}/meminfo` file. Note that CUDA also
    provides the available memory with `cuMemGetInfo_v2()`. However,
    `cuMemGetInfo_v2()` appears to overestimate the amount available. In our
    experience, the Linux sysfs file is more accurate.

 4. Allocate the memory with `mmap()` as described above, or other Linux APIs
    such as `numa_alloc_onnode()` from libnuma.

Example code is available in the [`numa_gpu::runtime::hw_info`
module](../numa-gpu/src/runtime/hw_info.rs).

### More details

Under the hood, the system allocators differ from `cuMemAlloc()` by:

 - Page type: `cuMemAlloc()` allocates 2 MB huge pages on Volta GPUs (exposed
   by TLB measurements), whereas the default page size of system allocators
   depends on the system configuration.

 - Page table entries: System allocators map the pages in the standard Linux
   page table. In contrast, `cuMemAlloc()` maps pages in a GPU page table
   managed by the Nvidia GPU driver. We have uncovered this behavior by
   measuring the latency of cold TLB misses.

 - Virtual address space: As a result of the separate GPU page table, memory
   allocated by `cuMemAlloc()` is mapped in the GPU's virtual address space,
   but not in the CPU's virtual address space. Thus, this type of GPU memory
   cannot be accessed by the CPU.
