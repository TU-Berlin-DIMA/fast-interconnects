/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

#include <cuda_clock.h>

#include <cstdint>

// X mod Y, assuming that Y is a power of 2
#define FAST_MODULO(X, Y) (X & (Y - 1))

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
extern "C" __global__ void gpu_read_bandwidth_seq_kernel(
    uint32_t *const __restrict__ data, size_t const size,
    uint32_t const /* loop_length */, uint64_t const /* target_cycles */,
    unsigned long long *const memory_accesses,
    unsigned long long *const measured_cycles) {
  uint32_t const global_size = gridDim.x * blockDim.x;
  uint32_t const gid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long long sum = 0;
  clock_type start = 0;
  clock_type stop = 0;

  get_clock(start);

  uint32_t dummy = 0;
  for (size_t i = gid; i < size; i += global_size) {
    dummy += data[i];
  }

  get_clock(stop);
  sum = stop - start;

  // Write result
  atomicMax(measured_cycles, sum);
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
extern "C" __global__ void gpu_write_bandwidth_seq_kernel(
    uint32_t *const __restrict__ data, size_t const size,
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
    data[i] = i;
  }

  get_clock(stop);
  sum = stop - start;

  // Write result
  atomicMax(measured_cycles, sum);
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
extern "C" __global__ void gpu_cas_bandwidth_seq_kernel(
    uint32_t *const __restrict__ data, size_t const size,
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
  atomicMax(measured_cycles, sum);
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
extern "C" __global__ void gpu_read_bandwidth_lcg_kernel(
    uint32_t *const __restrict__ data, size_t const size,
    uint32_t const loop_length, uint64_t const target_cycles,
    unsigned long long *const memory_accesses,
    unsigned long long *const measured_cycles) {
  uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long long sum = 0;
  unsigned long long mem_accesses = 0;
  uint32_t dummy = 0;
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
  atomicMax(measured_cycles, sum);
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
extern "C" __global__ void gpu_write_bandwidth_lcg_kernel(
    uint32_t *const __restrict__ data, size_t const size,
    uint32_t const loop_length, uint64_t const target_cycles,
    unsigned long long *const memory_accesses,
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
      data[location] = x;
    }

    mem_accesses += loop_length;
  } while ((get_clock(stop), stop) - start < target_cycles_i);

  sum = stop - start;

  // Write result
  atomicMax(measured_cycles, sum);
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
extern "C" __global__ void gpu_cas_bandwidth_lcg_kernel(
    uint32_t *const __restrict__ data, size_t const size,
    uint32_t const loop_length, uint64_t const target_cycles,
    unsigned long long *const memory_accesses,
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
  atomicMax(measured_cycles, sum);
  atomicAdd(memory_accesses, mem_accesses);
}
