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

#include <cuda_clock.h>
#include <cuda_vector.h>

#include <cooperative_groups.h>
#include <cstdint>

// X mod Y, assuming that Y is a power of 2
#define FAST_MODULO(X, Y) ((X) & ((Y)-1U))

namespace cg = cooperative_groups;

/*
 * Test sequential read bandwidth
 *
 * Read #size elements from array.
 *
 * Preconditions:
 *  - `memory_accesses` is 0
 *  - `measured_cycles` is 0
 *
 * Postconditions:
 *  - Aggregate clock cycles are written to `measured_cycles`
 *  - Aggregate number of memory accesses are written to `memory_accesses`
 */
template <typename T>
__device__ void gpu_read_bandwidth_seq_kernel(
    T *const __restrict__ data, size_t const size,
    uint32_t const /* loop_length */, uint64_t const /* target_cycles */,
    unsigned long long *const memory_accesses,
    unsigned long long *const measured_cycles) {
  uint32_t const global_size = gridDim.x * blockDim.x;
  uint32_t const gid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long long sum = 0;
  clock_type start = 0;
  clock_type stop = 0;

  get_clock(start);

  T dummy = {0};
  for (size_t i = gid; i < size; i += global_size) {
    dummy += data[i];
  }

  get_clock(stop);
  sum = stop - start;

  // Write result
  if (threadIdx.x == 0) {
    atomicMax(measured_cycles, sum);
  }
  if (gid == 0) {
    *memory_accesses = size;
  }

  // Prevent compiler optimization
  if (sum == 0) {
    data[1] = dummy;
  }
}

/*
 * Test sequential write bandwidth
 *
 * Write #size elements to array.
 *
 * Preconditions:
 *  - `memory_accesses` is 0
 *  - `measured_cycles` is 0
 *
 * Postconditions:
 *  - Aggregate clock cycles are written to `measured_cycles`
 *  - Aggregate number of memory accesses are written to `memory_accesses`
 *  - All array elements are filled with unspecified data
 */
template <typename T>
__device__ void gpu_write_bandwidth_seq_kernel(
    T *const __restrict__ data, size_t const size,
    uint32_t const /* loop_length */, uint64_t const /* target_cycles */,
    unsigned long long *const memory_accesses,
    unsigned long long *const measured_cycles) {
  uint32_t const global_size = gridDim.x * blockDim.x;
  uint32_t const gid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long long sum = 0;
  clock_type start = 0;
  clock_type stop = 0;

  get_clock(start);

  for (size_t i = gid; i < size; i += global_size) {
    data[i] = make_type<T>(i);
  }

  get_clock(stop);
  sum = stop - start;

  // Write result
  if (threadIdx.x == 0) {
    atomicMax(measured_cycles, sum);
  }
  if (gid == 0) {
    *memory_accesses = size;
  }
}

/*
 * Test sequential CompareAndSwap bandwidth
 *
 * Write #size elements to array.
 *
 * Preconditions:
 *  - `memory_accesses` is 0
 *  - `measured_cycles` is 0
 *
 * Postconditions:
 *  - Aggregate clock cycles are written to `measured_cycles`
 *  - Aggregate number of memory accesses are written to `memory_accesses`
 *  - All array elements are filled with unspecified data
 */
template <typename T>
__device__ void gpu_cas_bandwidth_seq_kernel(
    T *const __restrict__ data, size_t const size,
    uint32_t const /* loop_length */, uint64_t const /* target_cycles */,
    unsigned long long *const memory_accesses,
    unsigned long long *const measured_cycles) {
  uint32_t const global_size = gridDim.x * blockDim.x;
  uint32_t const gid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long long sum = 0;
  clock_type start = 0;
  clock_type stop = 0;

  get_clock(start);

  for (size_t i = gid; i < size; i += global_size) {
    atomicCAS(&data[i], i, i + 1);
  }

  get_clock(stop);
  sum = stop - start;

  // Write result
  if (threadIdx.x == 0) {
    atomicMax(measured_cycles, sum);
  }
  if (gid == 0) {
    *memory_accesses = size;
  }
}

/*
 * Test random read bandwidth
 *
 * Read #size elements to array. Random memory locations are generated using an
 * efficient Linear Congruential Generator.
 *
 * Preconditions:
 *  - size is a power of 2, i.e. 2^x
 *  - `memory_accesses` is 0
 *  - `measured_cycles` is 0
 *
 * Postconditions:
 *  - Aggregate clock cycles are written to `measured_cycles`
 *  - Aggregate number of memory accesses are written to `memory_accesses`
 */
template <typename T>
__device__ void gpu_read_bandwidth_lcg_kernel(
    T *const __restrict__ data, size_t const size, uint32_t const loop_length,
    uint64_t const target_cycles, unsigned long long *const memory_accesses,
    unsigned long long *const measured_cycles) {
  uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long long sum = 0;
  unsigned long long mem_accesses = 0;
  T dummy = {0};
  clock_type start = 0;
  clock_type stop = 0;
  clock_type target_cycles_i = static_cast<clock_type>(target_cycles);

  // Linear congruent generator
  // See: Knuth "The Art of Computer Programming - Volume 2"
  // and: https://en.wikipedia.org/wiki/Linear_congruential_generator
  uint64_t a = 6364136223846793005ULL;
  uint64_t c = 1442695040888963407ULL;
  uint64_t x = 67890ULL + gid;

  get_clock(start);

  // Do measurement
  do {
    for (uint32_t i = 0; i < loop_length; ++i) {
      // Generate next random number with LCG
      // Note: wrap modulo 2^64 is defined by C/C++ standard
      x = a * x + c;

      // Read from a random location within data range
      uint64_t location = FAST_MODULO(x, size);
      dummy += data[location];
    }

    mem_accesses += loop_length;
  } while ((get_clock(stop), stop) - start < target_cycles_i);

  sum = stop - start;

  // Write result
  if (threadIdx.x == 0) {
    atomicMax(measured_cycles, sum);
  }
  atomicAdd(memory_accesses, mem_accesses);

  // Prevent compiler optimization
  if (sum == 0) {
    data[1] = dummy;
  }
}

/*
 * Test random read bandwidth of a cooperative thread block tile
 *
 * Read #size elements to array. Random memory locations are generated using an
 * efficient Linear Congruential Generator.
 *
 * The threads in the same thread block tile read from contiguous offsets to
 * coalesce their accesses.
 *
 * Preconditions:
 *  - size is a power of 2, i.e. 2^x
 *  - `memory_accesses` is 0
 *  - `measured_cycles` is 0
 *
 * Postconditions:
 *  - Aggregate clock cycles are written to `measured_cycles`
 *  - Aggregate number of memory accesses are written to `memory_accesses`
 */
template <typename T, unsigned int TileSize>
__device__ void gpu_read_bandwidth_lcg_tiled_kernel(
    T *const __restrict__ data, size_t const size, uint32_t const misalignment,
    uint32_t const loop_length, uint64_t const target_cycles,
    unsigned long long *const memory_accesses,
    unsigned long long *const measured_cycles) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(block);

  uint32_t const meta_group_size = blockDim.x / TileSize;
  uint32_t const meta_group_rank = threadIdx.x / TileSize;
  uint32_t const tile_id = meta_group_rank + blockIdx.x * meta_group_size;
  size_t const tiled_size = size / tile.size();
  uint32_t const thread_rank = tile.thread_rank() + misalignment;

  unsigned long long sum = 0;
  unsigned long long mem_accesses = 0;
  T dummy = {0};
  clock_type start = 0;
  clock_type stop = 0;
  clock_type target_cycles_i = static_cast<clock_type>(target_cycles);

  // Linear congruent generator
  // See: Knuth "The Art of Computer Programming - Volume 2"
  // and: https://en.wikipedia.org/wiki/Linear_congruential_generator
  uint64_t a = 6364136223846793005ULL;
  uint64_t c = 1442695040888963407ULL;
  uint64_t x = 67890ULL + tile_id;

  get_clock(start);

  // Do measurement
  do {
    for (uint32_t i = 0; i < loop_length; ++i) {
      // Generate next random number with LCG
      // Note: wrap modulo 2^64 is defined by C/C++ standard
      x = a * x + c;

      // Read from a random location within data range
      uint64_t location =
          FAST_MODULO(x, tiled_size) * tile.size() + thread_rank;
      dummy += data[location];
    }

    mem_accesses += loop_length;
  } while (tile.any((get_clock(stop), stop) - start < target_cycles_i));

  sum = stop - start;

  // Write result
  if (threadIdx.x == 0) {
    atomicMax(measured_cycles, sum);
  }
  atomicAdd(memory_accesses, mem_accesses);

  // Prevent compiler optimization
  if (sum == 0) {
    data[1] = dummy;
  }
}

/*
 * Test random write bandwidth
 *
 * Write #size elements to array. Random memory locations are generated using an
 * efficient Linear Congruential Generator.
 *
 * Preconditions:
 *  - size is a power of 2, i.e. 2^x
 *  - `memory_accesses` is 0
 *  - `measured_cycles` is 0
 *
 * Postconditions:
 *  - Aggregate clock cycles are written to `measured_cycles`
 *  - Aggregate number of memory accesses are written to `memory_accesses`
 *  - All array elements are filled with unspecified data
 */
template <typename T>
__device__ void gpu_write_bandwidth_lcg_kernel(
    T *const __restrict__ data, size_t const size, uint32_t const loop_length,
    uint64_t const target_cycles, unsigned long long *const memory_accesses,
    unsigned long long *const measured_cycles) {
  uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long long sum = 0;
  unsigned long long mem_accesses = 0;
  clock_type start = 0;
  clock_type stop = 0;
  clock_type target_cycles_i = static_cast<clock_type>(target_cycles);

  // Linear congruent generator
  // See: Knuth "The Art of Computer Programming - Volume 2"
  // and: https://en.wikipedia.org/wiki/Linear_congruential_generator
  uint64_t a = 6364136223846793005ULL;
  uint64_t c = 1442695040888963407ULL;
  uint64_t x = 67890ULL + gid;

  get_clock(start);

  // Do measurement
  do {
    for (uint32_t i = 0; i < loop_length; ++i) {
      // Generate next random number with LCG
      // Note: wrap modulo 2^64 is defined by C/C++ standard
      x = a * x + c;

      // Write to a random location within data range
      uint64_t location = FAST_MODULO(x, size);
      data[location] = make_type<T>(x);
    }

    mem_accesses += loop_length;
  } while ((get_clock(stop), stop) - start < target_cycles_i);

  sum = stop - start;

  // Write result
  if (threadIdx.x == 0) {
    atomicMax(measured_cycles, sum);
  }
  atomicAdd(memory_accesses, mem_accesses);
}

/*
 * Test random write bandwidth of a cooperative thread block tile
 *
 * Write #size elements to array. Random memory locations are generated using an
 * efficient Linear Congruential Generator.
 *
 * The threads in the same thread block tile write to contiguous offsets to
 * coalesce their accesses.
 *
 * Preconditions:
 *  - size is a power of 2, i.e. 2^x
 *  - `memory_accesses` is 0
 *  - `measured_cycles` is 0
 *
 * Postconditions:
 *  - Aggregate clock cycles are written to `measured_cycles`
 *  - Aggregate number of memory accesses are written to `memory_accesses`
 *  - All array elements are filled with unspecified data
 */
template <typename T, unsigned int TileSize>
__device__ void gpu_write_bandwidth_lcg_tiled_kernel(
    T *const __restrict__ data, size_t const size, uint32_t const misalignment,
    uint32_t const loop_length, uint64_t const target_cycles,
    unsigned long long *const memory_accesses,
    unsigned long long *const measured_cycles) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(block);

  uint32_t const meta_group_size = blockDim.x / TileSize;
  uint32_t const meta_group_rank = threadIdx.x / TileSize;
  uint32_t const tile_id = meta_group_rank + blockIdx.x * meta_group_size;
  size_t const tiled_size = size / tile.size();
  uint32_t const thread_rank = tile.thread_rank() + misalignment;

  unsigned long long sum = 0;
  unsigned long long mem_accesses = 0;
  clock_type start = 0;
  clock_type stop = 0;
  clock_type target_cycles_i = static_cast<clock_type>(target_cycles);

  // Linear congruent generator
  // See: Knuth "The Art of Computer Programming - Volume 2"
  // and: https://en.wikipedia.org/wiki/Linear_congruential_generator
  uint64_t a = 6364136223846793005ULL;
  uint64_t c = 1442695040888963407ULL;
  uint64_t x = 67890ULL + tile_id;

  get_clock(start);

  // Do measurement
  do {
    for (uint32_t i = 0; i < loop_length; ++i) {
      // Generate next random number with LCG
      // Note: wrap modulo 2^64 is defined by C/C++ standard
      x = a * x + c;

      // Write to a random location within data range
      uint64_t location =
          FAST_MODULO(x, tiled_size) * tile.size() + thread_rank;
      data[location] = make_type<T>(x + tile.thread_rank());
    }

    mem_accesses += loop_length;
  } while (tile.any((get_clock(stop), stop) - start < target_cycles_i));

  sum = stop - start;

  // Write result
  if (threadIdx.x == 0) {
    atomicMax(measured_cycles, sum);
  }
  atomicAdd(memory_accesses, mem_accesses);
}

/*
 * Test random CompareAndSwap bandwidth
 *
 * Write #size elements to array. Random memory locations are generated using an
 * efficient Linear Congruential Generator.
 *
 * Preconditions:
 *  - size is a power of 2, i.e. 2^x
 *  - `memory_accesses` is 0
 *  - `measured_cycles` is 0
 *
 * Postconditions:
 *  - Aggregate clock cycles are written to `measured_cycles`
 *  - Aggregate number of memory accesses are written to `memory_accesses`
 *  - All array elements are filled with unspecified data
 */
template <typename T>
__device__ void gpu_cas_bandwidth_lcg_kernel(
    T *const __restrict__ data, size_t const size, uint32_t const loop_length,
    uint64_t const target_cycles, unsigned long long *const memory_accesses,
    unsigned long long *const measured_cycles) {
  uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long long sum = 0;
  unsigned long long mem_accesses = 0;
  clock_type start = 0;
  clock_type stop = 0;
  clock_type target_cycles_i = static_cast<clock_type>(target_cycles);

  // Linear congruent generator
  // See: Knuth "The Art of Computer Programming - Volume 2"
  // and: https://en.wikipedia.org/wiki/Linear_congruential_generator
  uint64_t a = 6364136223846793005ULL;
  uint64_t c = 1442695040888963407ULL;
  uint64_t x = 67890ULL + gid;

  get_clock(start);

  // Do measurement
  do {
    for (uint32_t i = 0; i < loop_length; ++i) {
      // Generate next random number with LCG
      // Note: wrap modulo 2^64 is defined by C/C++ standard
      x = a * x + c;

      // Write to a random location within data range
      uint64_t location = FAST_MODULO(x, size);
      atomicCAS(&data[location], location, x);
    }

    mem_accesses += loop_length;
  } while ((get_clock(stop), stop) - start < target_cycles_i);

  sum = stop - start;

  // Write result
  if (threadIdx.x == 0) {
    atomicMax(measured_cycles, sum);
  }
  atomicAdd(memory_accesses, mem_accesses);
}

/*
 * Test random CompareAndSwap bandwidth of a cooperative thread block tile
 *
 * Write #size elements to array. Random memory locations are generated using an
 * efficient Linear Congruential Generator.
 *
 * The threads in the same thread block tile perform a CAS on contiguous
 * offsets to coalesce their accesses.
 *
 * Preconditions:
 *  - size is a power of 2, i.e. 2^x
 *  - `memory_accesses` is 0
 *  - `measured_cycles` is 0
 *
 * Postconditions:
 *  - Aggregate clock cycles are written to `measured_cycles`
 *  - Aggregate number of memory accesses are written to `memory_accesses`
 *  - All array elements are filled with unspecified data
 */
template <typename T, unsigned int TileSize>
__device__ void gpu_cas_bandwidth_lcg_tiled_kernel(
    T *const __restrict__ data, size_t const size, uint32_t const misalignment,
    uint32_t const loop_length, uint64_t const target_cycles,
    unsigned long long *const memory_accesses,
    unsigned long long *const measured_cycles) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(block);

  uint32_t const meta_group_size = blockDim.x / TileSize;
  uint32_t const meta_group_rank = threadIdx.x / TileSize;
  uint32_t const tile_id = meta_group_rank + blockIdx.x * meta_group_size;
  size_t const tiled_size = size / tile.size();
  uint32_t const thread_rank = tile.thread_rank() + misalignment;

  unsigned long long sum = 0;
  unsigned long long mem_accesses = 0;
  clock_type start = 0;
  clock_type stop = 0;
  clock_type target_cycles_i = static_cast<clock_type>(target_cycles);

  // Linear congruent generator
  // See: Knuth "The Art of Computer Programming - Volume 2"
  // and: https://en.wikipedia.org/wiki/Linear_congruential_generator
  uint64_t a = 6364136223846793005ULL;
  uint64_t c = 1442695040888963407ULL;
  uint64_t x = 67890ULL + tile_id;

  get_clock(start);

  // Do measurement
  do {
    for (uint32_t i = 0; i < loop_length; ++i) {
      // Generate next random number with LCG
      // Note: wrap modulo 2^64 is defined by C/C++ standard
      x = a * x + c;

      // Write to a random location within data range
      uint64_t location =
          FAST_MODULO(x, tiled_size) * tile.size() + thread_rank;
      atomicCAS(&data[location], static_cast<T>(location + thread_rank),
                static_cast<T>(x + thread_rank));
    }

    mem_accesses += loop_length;
  } while (tile.any((get_clock(stop), stop) - start < target_cycles_i));

  sum = stop - start;

  // Write result
  if (threadIdx.x == 0) {
    atomicMax(measured_cycles, sum);
  }
  atomicAdd(memory_accesses, mem_accesses);
}

// ============== Instantiate templates ==============

#define MAKE_BENCHMARK(FUNCTION_NAME, SUFFIX, DATA_TYPE)                       \
  extern "C" __global__ void FUNCTION_NAME##_##SUFFIX(                         \
      DATA_TYPE *const __restrict__ data, size_t const size,                   \
      uint32_t const /* misalignment */, uint32_t const loop_length,           \
      uint64_t const target_cycles, unsigned long long *const memory_accesses, \
      unsigned long long *const measured_cycles) {                             \
    FUNCTION_NAME##_kernel<DATA_TYPE>(data, size, loop_length, target_cycles,  \
                                      memory_accesses, measured_cycles);       \
  }

#define MAKE_BENCHMARK_TILED(FUNCTION_NAME, SUFFIX, DATA_TYPE, TILE_SIZE)      \
  extern "C" __global__ void FUNCTION_NAME##_##SUFFIX##_##TILE_SIZE##T(        \
      DATA_TYPE *const __restrict__ data, size_t const size,                   \
      uint32_t const misalignment, uint32_t const loop_length,                 \
      uint64_t const target_cycles, unsigned long long *const memory_accesses, \
      unsigned long long *const measured_cycles) {                             \
    FUNCTION_NAME##_tiled_kernel<DATA_TYPE, TILE_SIZE>(                        \
        data, size, misalignment, loop_length, target_cycles, memory_accesses, \
        measured_cycles);                                                      \
  }

MAKE_BENCHMARK(gpu_read_bandwidth_seq, 4B, uint32_t)
MAKE_BENCHMARK(gpu_read_bandwidth_seq, 8B, uint64_t)
MAKE_BENCHMARK(gpu_read_bandwidth_seq, 16B, ulonglong2)

MAKE_BENCHMARK(gpu_write_bandwidth_seq, 4B, uint32_t)
MAKE_BENCHMARK(gpu_write_bandwidth_seq, 8B, uint64_t)
MAKE_BENCHMARK(gpu_write_bandwidth_seq, 16B, ulonglong2)

MAKE_BENCHMARK(gpu_cas_bandwidth_seq, 4B, unsigned int)
MAKE_BENCHMARK(gpu_cas_bandwidth_seq, 8B, unsigned long long)

MAKE_BENCHMARK(gpu_read_bandwidth_lcg, 4B, uint32_t)
MAKE_BENCHMARK(gpu_read_bandwidth_lcg, 8B, uint64_t)
MAKE_BENCHMARK(gpu_read_bandwidth_lcg, 16B, ulonglong2)
MAKE_BENCHMARK_TILED(gpu_read_bandwidth_lcg, 4B, uint32_t, 2)
MAKE_BENCHMARK_TILED(gpu_read_bandwidth_lcg, 8B, uint64_t, 2)
MAKE_BENCHMARK_TILED(gpu_read_bandwidth_lcg, 16B, ulonglong2, 2)
MAKE_BENCHMARK_TILED(gpu_read_bandwidth_lcg, 4B, uint32_t, 4)
MAKE_BENCHMARK_TILED(gpu_read_bandwidth_lcg, 8B, uint64_t, 4)
MAKE_BENCHMARK_TILED(gpu_read_bandwidth_lcg, 16B, ulonglong2, 4)
MAKE_BENCHMARK_TILED(gpu_read_bandwidth_lcg, 4B, uint32_t, 8)
MAKE_BENCHMARK_TILED(gpu_read_bandwidth_lcg, 8B, uint64_t, 8)
MAKE_BENCHMARK_TILED(gpu_read_bandwidth_lcg, 16B, ulonglong2, 8)
MAKE_BENCHMARK_TILED(gpu_read_bandwidth_lcg, 4B, uint32_t, 16)
MAKE_BENCHMARK_TILED(gpu_read_bandwidth_lcg, 8B, uint64_t, 16)
MAKE_BENCHMARK_TILED(gpu_read_bandwidth_lcg, 16B, ulonglong2, 16)
MAKE_BENCHMARK_TILED(gpu_read_bandwidth_lcg, 4B, uint32_t, 32)
MAKE_BENCHMARK_TILED(gpu_read_bandwidth_lcg, 8B, uint64_t, 32)
MAKE_BENCHMARK_TILED(gpu_read_bandwidth_lcg, 16B, ulonglong2, 32)

MAKE_BENCHMARK(gpu_write_bandwidth_lcg, 4B, uint32_t)
MAKE_BENCHMARK(gpu_write_bandwidth_lcg, 8B, uint64_t)
MAKE_BENCHMARK(gpu_write_bandwidth_lcg, 16B, ulonglong2)
MAKE_BENCHMARK_TILED(gpu_write_bandwidth_lcg, 4B, uint32_t, 2)
MAKE_BENCHMARK_TILED(gpu_write_bandwidth_lcg, 8B, uint64_t, 2)
MAKE_BENCHMARK_TILED(gpu_write_bandwidth_lcg, 16B, ulonglong2, 2)
MAKE_BENCHMARK_TILED(gpu_write_bandwidth_lcg, 4B, uint32_t, 4)
MAKE_BENCHMARK_TILED(gpu_write_bandwidth_lcg, 8B, uint64_t, 4)
MAKE_BENCHMARK_TILED(gpu_write_bandwidth_lcg, 16B, ulonglong2, 4)
MAKE_BENCHMARK_TILED(gpu_write_bandwidth_lcg, 4B, uint32_t, 8)
MAKE_BENCHMARK_TILED(gpu_write_bandwidth_lcg, 8B, uint64_t, 8)
MAKE_BENCHMARK_TILED(gpu_write_bandwidth_lcg, 16B, ulonglong2, 8)
MAKE_BENCHMARK_TILED(gpu_write_bandwidth_lcg, 4B, uint32_t, 16)
MAKE_BENCHMARK_TILED(gpu_write_bandwidth_lcg, 8B, uint64_t, 16)
MAKE_BENCHMARK_TILED(gpu_write_bandwidth_lcg, 16B, ulonglong2, 16)
MAKE_BENCHMARK_TILED(gpu_write_bandwidth_lcg, 4B, uint32_t, 32)
MAKE_BENCHMARK_TILED(gpu_write_bandwidth_lcg, 8B, uint64_t, 32)
MAKE_BENCHMARK_TILED(gpu_write_bandwidth_lcg, 16B, ulonglong2, 32)

MAKE_BENCHMARK(gpu_cas_bandwidth_lcg, 4B, unsigned int)
MAKE_BENCHMARK(gpu_cas_bandwidth_lcg, 8B, unsigned long long)
MAKE_BENCHMARK_TILED(gpu_cas_bandwidth_lcg, 4B, unsigned int, 2)
MAKE_BENCHMARK_TILED(gpu_cas_bandwidth_lcg, 8B, unsigned long long, 2)
MAKE_BENCHMARK_TILED(gpu_cas_bandwidth_lcg, 4B, unsigned int, 4)
MAKE_BENCHMARK_TILED(gpu_cas_bandwidth_lcg, 8B, unsigned long long, 4)
MAKE_BENCHMARK_TILED(gpu_cas_bandwidth_lcg, 4B, unsigned int, 8)
MAKE_BENCHMARK_TILED(gpu_cas_bandwidth_lcg, 8B, unsigned long long, 8)
MAKE_BENCHMARK_TILED(gpu_cas_bandwidth_lcg, 4B, unsigned int, 16)
MAKE_BENCHMARK_TILED(gpu_cas_bandwidth_lcg, 8B, unsigned long long, 16)
MAKE_BENCHMARK_TILED(gpu_cas_bandwidth_lcg, 4B, unsigned int, 32)
MAKE_BENCHMARK_TILED(gpu_cas_bandwidth_lcg, 8B, unsigned long long, 32)
