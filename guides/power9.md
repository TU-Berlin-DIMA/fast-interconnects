POWER9 Microarchitecture Tuning
================================

The IBM POWER9 is a fast CPU compared to its contemporaries, the Intel
Skylake-SP and AMD Naples architectures. In particular, the POWER9 has a eight
DDR4 memory channels and NVLink 2.0 support. However, as its a less popular
platform, fewer tuning guides are available Online.

Thus, here are some hints on how to tune your code for the IBM POWER9 CPU based
on our experience implementing a radix partitioner. The covered topics are:

 - General resources and hints

 - Cachelines

 - TLBs and pages

 - SIMD using AltiVec

 - Hardware prefetcher tuning

 - CPU core affinity tuning

 - Non-temporal stores

 - Energy and power measurement

 - Endianness

## General Resources and Hints

IBM provides thorough documentation:

 - The [IBM POWER9 User's
   Manual](https://openpowerfoundation.org/resources/ibmpower9usermanual)
   specifies the POWER9 CPU architectural design. It's useful to gain insights
   into, e.g., the cache subsystem and the four-way SMT.

 - The [POWER9 ISA Version
   3.0B](https://ibm.ent.box.com/s/1hzcwkwf8rbju5h9iyf44wm94amnlcrv) defines
   the available CPU instructions. As we're writing a database and not a
   compiler, it's value is mostly in documenting special purpose registers such
   as the "data stream control register (DSCR)". More on that below in the
   hardware prefetcher section.

 - [POWER9 Performance Monitor User's
   Guide](https://wiki.raptorcs.com/w/images/6/6b/POWER9_PMU_UG_v12_28NOV2018_pub.pdf)
   specifies the performance monitor, e.g., CPU cycle counters and cache miss
   counters. These monitors are accessible via tools such as Linux Perf and
   [Likwid](https://github.com/RRZE-HPC/likwid).

## Cachelines

IBM POWER processors have 128-byte cachelines instead of the 64-byte cachelines
found on x86_64 CPUs. In the L1 data cache, each cacheline consists of two
64-byte sub-cachelines (aka sectors, see POWER9 User's Manual, p. 42). The
Users Manual state that the memory controller can fetch 64 into the cache
instead of 128 bytes "when memory bandwidth utilization is very high" (POWER9
User's Manual, p. 362).

In theory, this would be interesting for radix partitioning with software
write-combining. In practice however, setting the software write-combine
buffers to 64 bytes yielded lower throughput than setting them to 128 bytes.

# TLBs and Pages

The POWER9 has 64 KB pages by default under Linux, instead of 4 KB on x86_64.
However, the huge page sizes are the same at 2 MB and 1 GB. See our [huge pages
guide](./huge_pages.md) for details on using them.

Like on x86_64, page table is stored as a radix tree by default (radix MMU
mode). However, the POWER9 also supports pages table as a hash table (hash MMU
mode). The hash MMU mode has different huge page sizes, i.e., 16 MB and 16 GB.

The hash MMU mode can be enabled by setting `disable_radix` on the Linux boot
commandline in the bootloader (see the [Linux kernel
documentation](https://www.kernel.org/doc/html/latest/admin-guide/kernel-parameters.html)).
If successful, you should see:

```sh
{ 0.000000} hash-mmu: Initializing hash mmu with SLB
```

instead of:

```sh
[ 0.000000] radix-mmu: Initializing Radix MMU
```

when running `dmesg`.

Radix MMU mode and hash MMU mode have different TLB architectures. In radix MMU
mode, there is a "traditional" two-level TLB hierarchy. IBM refers to the L1
TLB as an "ERAT" cache (Effective to Real Address Translation), and to the L2
TLB simply as the "TLB". This terminology stems from the fact that historically
the operating system controlled the TLB, but did not control the ERAT (see
[Peng et al. "The PowerPC Architecture: 64-bit power with 32-bit
compatibility",
COMPCON'95](https://ieeexplore.ieee.org/abstract/document/512400)). In
contrast, the hash MMU mode uses a "segment lookaside buffer" (SLB) in addition
to the ERAT and TLB to cache segment translations (256 MB or 1 TB, see POWER9
Users' Manual, p. 42).

As a side note, the "effective address" (EA) is a virtual address in x86
terminology, and the "real address" is a physical address. Additionally, a
"virtual address" (VA) is a third level in between the EA and the RA. The VA
space was originally meant to unify multiple physical address spaces into a
single global address space, which would then be divided into the per-process
EA spaces (see [Peng et
al.](https://ieeexplore.ieee.org/abstract/document/512400)). However, modern
Linux translates directly from EA to RA when running on bare metal (see [Jann
et al. "IBM POWER9 system software", IBM J. Res. & Dev.
'18](https://ieeexplore.ieee.org/document/8392671)).

## SIMD using AltiVec

POWER CPUs use SIMD instructions provided by the AltiVec instruction set. These
are 128-bit wide and support most common operations. Unfortunately, they don't
support gather/scatter instructions from/to memory, which AVX-512 does provide.

There are two common versions of AltiVec: VMX and VSX. VMX is an MMX-era ISA,
whereas VSX is an SSE-era ISA. We'll focus on VSX, as this is the newer
instruction set.

AltiVec is comparatively easy to use. GCC provides 128-bit vector types (e.g.,
`vector int` and `vector float`) and defines overloaded memory and arithmetic
operators on these (e.g., `=`, `+`).

For example, computing a histogram with VSX could look like this:

```C++
#ifdef __ALTIVEC__
#include <altivec.h>

constexpr size_t VEC_LEN = sizeof(vector int) / sizeof(int);

template <typename T, typename M, typename B>
vector unsigned int key_to_partition_simd(vector int key, unsigned int mask) {
    return reinterpret_cast<vector unsigned int>(key) & mask;
}

// data[0] must be 16-byte aligned, i.e., the cacheline size
int const *const data = ...;

// define mask and set all vector lanes to the scalar's value
unsigned int mask = (1U << radix_bits) - 1U;
const vector unsigned int mask_vsx = vec_splats(mask);

for (size_t i = 0; i < data_size; i += VEC_LEN * 4) {
    int const *const src = &data[i];

    // load a cacheline of data
    vector int key0 = vec_ld(0, base);
    vector int key1 = vec_ld(16, base);
    vector int key2 = vec_ld(32, base);
    vector int key3 = vec_ld(48, base);

    // compute partition indexes
    vector unsigned int p_index0 = key_to_partition_simd(key0, mask_vsx);
    vector unsigned int p_index1 = key_to_partition_simd(key1, mask_vsx);
    vector unsigned int p_index2 = key_to_partition_simd(key2, mask_vsx);
    vector unsigned int p_index3 = key_to_partition_simd(key3, mask_vsx);

    // increment four different histograms to avoid read-write dependencies
    // and facilitate super-scalar execution
    for (size_t v = 0; v < VEC_LEN; ++v) {
        histogram0[p_index0[v]] += 1;
        histogram1[p_index1[v]] += 1;
        histogram2[p_index2[v]] += 1;
        histogram3[p_index3[v]] += 1;
    }
}

// aggregate temporary histograms into a final histogram
for (size_t h = 0; h < histogram_size; ++h) {
    histogram[h] += histogram0[h];
    histogram[h] += histogram1[h];
    histogram[h] += histogram2[h];
    histogram[h] += histogram3[h];
}
#endif
```

### References

See also the [nice tutorial by Sven
K&ouml;hler](https://www.dcl.hpi.uni-potsdam.de/events/POWER2016/assignment3/assignment3_slides.pdf)
and the [Linux on POWER porting guide for vector
intrinsics](http://openpowerfoundation.org/wp-content/uploads/resources/Vector-Intrinsics/Vector-Intrinsics-20180306.pdf)

### Pitfalls

The AltiVec header files redefines the `bool` type. This behavior is documented
on the [GCC bug tracker](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58241)
and the [RedHat bug
tracker](https://bugzilla.redhat.com/show_bug.cgi?id=1394505).

The solution is to undefine the AltiVec definition:

```C++
#ifdef __ALTIVEC__
#ifdef bool
#undef bool
#endif
#endif
```

## Hardware Prefetcher Tuning

Like all modern CPUs, the POWER9 has a hardware prefetcher, as well as software prefetch instructions. For our use case, we found that software prefetching did not show gains, but tuning the hardware prefetcher increased performance by tens of GB/s.

Software prefetching works by using the GCC `__builtin_prefetch` intrinsic (see
the [GCC docs](https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html)). The
intrinsic maps to the `dst` (data stream touch) and `dstst` (data stream touch
for store) instructions, depending on the parameters given.

The hardware prefetcher supports N-stride prefetching, which is enabled by
default. This mode attempts to detect strided accesses and intelligently
prefetch these data (see POWER9 User's Manual, p. 351). However, we discovered
that disabling N-stride prefetching and instead simply doing a linear prefetch
is faster.

There are two ways to disable N-stride prefetching. In your program:

```C++
// Move to special-purpose register
#ifdef __powerpc64__
#define __mtspr(spr, value) \
  __asm__ volatile("mtspr %0,%1" : : "n"(spr), "r"(value))

// Data stream control register
#define PPC_DSCR 3

// Disable strided prefetch and set maximum prefetch depth
#define PPC_TUNE_DSCR 7ULL

...

// Set this once in your function
__mtspr(PPC_DSCR, PPC_TUNE_DSCR);
#endif
```

On the commandline:

```sh
sudo ppc64_cpu --dscr=7 # The default is 16
```

You can also disable hardware prefetching entirely by setting DSCR=1.

### DSCR Values

The values for DSCR are defined in the POWER ISA manual 3.0b on page 837. Note
that in the manual, bits are numbered from left=MSB to right=LSB (see p. 4).
I.e., bit #0 is most significant bit, and bit #63 is least significant bit.

- 16: default value, enables strided prefetching
- 1 :: disables prefetching
- 0 :: sets default prefetch depth, stride-N disabled (i.e., approx. same as DSCR=5)
- 7 :: sets deepest prefetch depth, stride-N disabled

## CPU Core Affinity Tuning

In the POWER9, every two cores (i.e., a "core pair") share their L2 and L3 data
caches.  That's fair and well, but let's shift our focus to maximizing the
memory bandwidth.

The L2 caches interface with the memory controller, and therefore the core pair
also shares the memory load and store queues (see POWER9 User's Manual, p.
157).

In our measurements, we observed that on a 16 core (4-way SMT = 64 threads)
CPU, using only 8 threads (i.e,. one thread per core pair) can result in a
higher memory bandwidth than all other configurations.

Our thread-to-core mapping with `sched_setaffinity` looks like this:

```sh
0 8 16 24 32 40 48 56
```

Of course, this applies only to data-intensive applications with a sequential
memory access pattern. Compute-intensive applications and irregular access
patterns to memory (i.e., data doesn't fit to the cache) perform better with
more cores and possibly benefit from SMT.

## Non-Temporal Stores

A well-known optimization for radix partitioning are non-temporal stores. On
x86_64, these are invoked using the `_mm_stream_si128` (SSE2),
`_mm256_stream_si256` (AVX), and `_mm512_stream_si512` (AVX-512F) instructions.
In contrast, the POWER ISA does not support non-temporal store instructions.
However, there are ways to improve performance.

First of all, the CPU supports hardware write-combining ("store gathering").
Each store queue has 16 x 64-byte store gather stations (see POWER9 User's
Manual, p. 181). Nothing to explicitly configure here, these "just work".
However, we achieved the best results by writing a cacheline-at-a-time using
four `vec_st` VSX instructions, thus optimizing for store gathering:

```C++
vec_st(tmp0, 0, vsx_dst);
vec_st(tmp1, 16, vsx_dst);
vec_st(tmp2, 32, vsx_dst);
vec_st(tmp3, 48, vsx_dst);
```

In other write-heavy scenarios, allocating an empty cacheline might improve
performance. The cacheline is allocated in the cache and filled with zeroes
without reading anything from memory:

```
// Set cacheline to zero
#define __dcbz(base) __asm__ volatile("dcbz 0,%0" ::"r"(base) : "memory")

int *tmp =  ...; // Some array with at least four 16-byte aligned integers
__dcbz(&tmp);
```

Temporal hint instructions do exist, but slowed down our code in practice.
These are the `dcbtt` (data cache block touch - transient) and `dcbtstt` (data
cache block touch for store - transient) instructions (see the POWER ISA
version 3.0b, p. 849 and 850 and POWER9 User's Manual, p. 67).

## Power and Energy Measurement

The POWER9 contains an embedded on-chip controller (OCC) that collects power
usage statistics. Fun fact: The OCC is actually a 400 MHz PPC-405 core (see the
POWER9 User's Manual, p. 298).

The collected measurements can be conveniently accessed using lm-sensors with
the command:

```sh
sensors
```

Note that this does not require any configuration, it "just works". The
relevant sensors return energy in Mega Joules since system boot. This is more
convenient than Watts, because the OCC takes care of periodically measuring the
current power consumption (i.e., Watts). Of course, `sensors` also gives you
the Watts if you prefer.

The relevant sensors are:

 - Chip 0 GPU: The GPUs, per socket

 - Chip 0 Memory: The DIMMs, per socket

 - Chip 0 Vdd: The CPU cores, per socket

 - Chip 0 Vdn: The Nest uncore (i.e., NVLink, CAPI, etc.), per socket

 - System: The total system

### References

See also the [blog post by Stewart
Smith](https://www.flamingspork.com/blog/tag/lm-sensors/) and the [OpenPOWER
OCC tools](https://github.com/open-power/occ/tree/master/src/tools).

## Endianness

The POWER9 can natively operate in both big endian and little endian modes. The
way this works that, in little endian mode, the CPU core flips the bytes when
loads data from the cache (see POWER9 User's Manual, p. 64). Thus, the core
actually always executes in big endian mode.

The question in my mind was: Why is little endian mode supported at all?
[According to Jeff Scheel, IBM Linux on Power Chief
Engineer](https://www.ibm.com/support/pages/just-faqs-about-little-endian), IBM
wanted to make Linux adoption easier, as the ecosystem mainly runs on  x86_64
and thus little endian. However, as Nvidia GPUs are little endian, this makes
the life of a database researcher more simple, too.

### Pitfalls

GCC doesn't generate native 128-bit load/store instructions for `__int128` in
little endian mode, but does in big endian mode. Technically, the POWER ISA
supports 128-bit loads/stores with the `lq` and `stq` instructions. Instead,
GCC generates two 64-bit `ld` or `std` instructions (see the [GCC source
code](https://github.com/gcc-mirror/gcc/blob/d03ca8a6148f55e119b8220a9c65147173b32065/gcc/config/rs6000/rs6000.c#L4019).

A workaround is to force GCC to generate the `lq` or `stq` instructions using
`__atomic_load_n(addr, __ATOMIC_RELAXED)` and `__atomic_store_n(addr, value,
__ATOMIC_RELAXED)`. These intrinsics emit only the `lq` and `stq` instructions
and thus cause no side effects. However, GCC does not loop-unroll when using
the atomic intrinsics.
