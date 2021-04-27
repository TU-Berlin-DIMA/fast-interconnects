/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

#include <cpu_clock.h>

#include <atomic>
#include <cstdint>

#if defined(__powerpc64__)
#include <ppc_intrinsics.h>
#endif

// Disable strided prefetch and set maximum prefetch depth
#define PPC_TUNE_DSCR 7ULL

// X mod Y, assuming that Y is a power of 2
#define FAST_MODULO(X, Y) (X & (Y - 1))

// FIXME: test POWER9 branch predictor optimization

/*
 * Test sequential read bandwidth
 *
 * Read #size elements from array.
 *
 * Preconditions:
 *  - None
 *
 * Postconditions:
 *  - Clock cycles are written to `measured_cycles`
 *  - Number of memory accesses are written to `memory_accesses`
 */
extern "C" void cpu_read_bandwidth_seq(
    uint32_t *data, std::size_t size, uint32_t const /* loop_length */,
    uint64_t const /* target_cycles */,
    unsigned long long *const memory_accesses,
    unsigned long long *const measured_cycles, std::size_t tid,
    std::size_t num_threads) {
#if defined(__powerpc64__)
  __mtspr(PPC_DSCR, PPC_TUNE_DSCR);
#endif

  std::size_t const chunk_size = (size + num_threads - 1) / num_threads;
  std::size_t const begin = tid * chunk_size;
  std::size_t const end =
      ((tid + 1) * chunk_size - 1 > size) ? size : (tid + 1) * chunk_size - 1;
  uint64_t sum = 0;
  clock_type start = 0;
  clock_type stop = 0;

  get_clock(start);

  uint32_t dummy = 0;
  for (std::size_t i = begin; i < end; ++i) {
    dummy += data[i];
  }

  get_clock(stop);
  sum = stop - start;

  // Write result
  *measured_cycles = sum;
  *memory_accesses = end - begin;

  // Prevent compiler optimization
  if (dummy == 0) {
    data[0] = dummy;
  }
}

/*
 * Test sequential write bandwidth
 *
 * Write #size elements to array.
 *
 * Preconditions:
 *  - None
 *
 * Postconditions:
 *  - Clock cycles are written to `measured_cycles`
 *  - Number of memory accesses are written to `memory_accesses`
 *  - All array elements are filled with unspecified data
 */
extern "C" void cpu_write_bandwidth_seq(
    uint32_t *data, std::size_t size, uint32_t const /* loop_length */,
    uint64_t const /* target_cycles */,
    unsigned long long *const memory_accesses,
    unsigned long long *const measured_cycles, std::size_t tid,
    std::size_t num_threads) {
#if defined(__powerpc64__)
  __mtspr(PPC_DSCR, PPC_TUNE_DSCR);
#endif

  std::size_t const chunk_size = (size + num_threads - 1) / num_threads;
  std::size_t const begin = tid * chunk_size;
  std::size_t const end =
      ((tid + 1) * chunk_size - 1 > size) ? size : (tid + 1) * chunk_size - 1;
  uint64_t sum = 0;
  clock_type start = 0;
  clock_type stop = 0;

  get_clock(start);

  for (std::size_t i = begin; i < end; ++i) {
    data[i] = i;
  }

  get_clock(stop);
  sum = stop - start;

  // Write result
  *measured_cycles = sum;
  *memory_accesses = end - begin;
}

/*
 * Test sequential CompareAndSwap bandwidth
 *
 * Write #size elements to array.
 *
 * Preconditions:
 *  - None
 *
 * Postconditions:
 *  - Clock cycles are written to `measured_cycles`
 *  - Number of memory accesses are written to `memory_accesses`
 *  - All array elements are filled with unspecified data
 */
extern "C" void cpu_cas_bandwidth_seq(uint32_t *data, std::size_t size,
                                      uint32_t const /* loop_length */,
                                      uint64_t const /* target_cycles */,
                                      unsigned long long *const memory_accesses,
                                      unsigned long long *const measured_cycles,
                                      std::size_t tid,
                                      std::size_t num_threads) {
#if defined(__powerpc64__)
  __mtspr(PPC_DSCR, PPC_TUNE_DSCR);
#endif

  std::size_t const chunk_size = (size + num_threads - 1) / num_threads;
  std::size_t const begin = tid * chunk_size;
  std::size_t const end =
      ((tid + 1) * chunk_size - 1 > size) ? size : (tid + 1) * chunk_size - 1;
  uint64_t sum = 0;
  clock_type start = 0;
  clock_type stop = 0;

  get_clock(start);

  for (std::size_t i = begin; i < end; ++i) {
    auto *item = reinterpret_cast<std::atomic<uint32_t> *>(&data[i]);
    uint32_t expected = (uint32_t)i;
    std::atomic_compare_exchange_strong(item, &expected, (uint32_t)i + 1);
  }

  get_clock(stop);
  sum = stop - start;

  // Write result
  *measured_cycles = sum;
  *memory_accesses = end - begin;
}

/*
 * Test random read bandwidth
 *
 * Read #size elements from array. Random memory locations are generated using
 * an efficient Linear Congruential Generator.
 *
 * Preconditions:
 *  - size is a power of 2, i.e. 2^x
 *
 * Postconditions:
 *  - Clock cycles are written to `measured_cycles`
 *  - Number of memory accesses are written to `memory_accesses`
 */
extern "C" void cpu_read_bandwidth_lcg(uint32_t *data, std::size_t size,
                                       uint32_t const loop_length,
                                       uint64_t const target_cycles,
                                       uint64_t *const memory_accesses,
                                       uint64_t *const measured_cycles,
                                       std::size_t tid,
                                       std::size_t /* num_threads */) {
#if defined(__powerpc64__)
  __mtspr(PPC_DSCR, PPC_TUNE_DSCR);
#endif

  uint64_t sum = 0;
  uint64_t mem_accesses = 0;
  uint32_t dummy = 0;
  clock_type start = 0;
  clock_type stop = 0;
  clock_type target_cycles_i = static_cast<clock_type>(target_cycles);

  // Linear congruent generator
  // See: Knuth "The Art of Computer Programming - Volume 2"
  // and: https://en.wikipedia.org/wiki/Linear_congruential_generator
  uint64_t a = 6364136223846793005ULL;
  uint64_t c = 1442695040888963407ULL;
  uint64_t x = 67890ULL + tid;

  get_clock(start);

  // Do measurement
  do {
    for (uint32_t i = 0; i < loop_length; ++i) {
      // Generate next random number with LCG
      // Note: wrap modulo 2^64 is defined by C/C++ standard
      x = a * x + c;

      // Read from a random location within data range
      uint64_t index = FAST_MODULO(x, size);
      dummy += data[index];
    }

    mem_accesses += loop_length;
  } while ((get_clock(stop), stop) - start < target_cycles_i);

  sum = stop - start;

  // Write result
  *measured_cycles = sum;
  *memory_accesses = mem_accesses;

  // Prevent compiler optimization
  if (dummy == 0) {
    data[0] = dummy;
  }
}

/*
 * Test random write bandwidth
 *
 * Write #size elements to array. Random memory locations are generated using
 * an efficient Linear Congruential Generator.
 *
 * Preconditions:
 *  - size is a power of 2, i.e. 2^x
 *
 * Postconditions:
 *  - Clock cycles are written to `measured_cycles`
 *  - Number of memory accesses are written to `memory_accesses`
 *  - All array elements are filled with unspecified data
 */
extern "C" void cpu_write_bandwidth_lcg(uint32_t *data, std::size_t size,
                                        uint32_t const loop_length,
                                        uint64_t const target_cycles,
                                        uint64_t *const memory_accesses,
                                        uint64_t *const measured_cycles,
                                        std::size_t tid,
                                        std::size_t /* num_threads */) {
#if defined(__powerpc64__)
  __mtspr(PPC_DSCR, PPC_TUNE_DSCR);
#endif

  uint64_t sum = 0;
  uint64_t mem_accesses = 0;
  clock_type start = 0;
  clock_type stop = 0;
  clock_type target_cycles_i = static_cast<clock_type>(target_cycles);

  // Linear congruent generator
  // See: Knuth "The Art of Computer Programming - Volume 2"
  // and: https://en.wikipedia.org/wiki/Linear_congruential_generator
  uint64_t a = 6364136223846793005ULL;
  uint64_t c = 1442695040888963407ULL;
  uint64_t x = 67890ULL + tid;

  get_clock(start);

  // Do measurement
  do {
    for (uint32_t i = 0; i < loop_length; ++i) {
      // Generate next random number with LCG
      // Note: wrap modulo 2^64 is defined by C/C++ standard
      x = a * x + c;

      // Write to a random location within data range
      uint64_t index = FAST_MODULO(x, size);
      data[index] = x;
    }

    mem_accesses += loop_length;
  } while ((get_clock(stop), stop) - start < target_cycles_i);

  sum = stop - start;

  // Write result
  *measured_cycles = sum;
  *memory_accesses = mem_accesses;
}

/*
 * Test random CompareAndSwap bandwidth
 *
 * Write #size elements to array. Random memory locations are generated using
 * an efficient Linear Congruential Generator.
 *
 * Preconditions:
 *  - size is a power of 2, i.e. 2^x
 *
 * Postconditions:
 *  - Clock cycles are written to `measured_cycles`
 *  - Number of memory accesses are written to `memory_accesses`
 *  - All array elements are filled with unspecified data
 */
extern "C" void cpu_cas_bandwidth_lcg(uint32_t *data, std::size_t size,
                                      uint32_t const loop_length,
                                      uint64_t const target_cycles,
                                      uint64_t *const memory_accesses,
                                      uint64_t *const measured_cycles,
                                      std::size_t tid,
                                      std::size_t /* num_threads */) {
#if defined(__powerpc64__)
  __mtspr(PPC_DSCR, PPC_TUNE_DSCR);
#endif

  uint64_t sum = 0;
  uint64_t mem_accesses = 0;
  clock_type start = 0;
  clock_type stop = 0;
  clock_type target_cycles_i = static_cast<clock_type>(target_cycles);

  get_clock(start);

  // Linear congruent generator
  // See: Knuth "The Art of Computer Programming - Volume 2"
  // and: https://en.wikipedia.org/wiki/Linear_congruential_generator
  uint64_t a = 6364136223846793005ULL;
  uint64_t c = 1442695040888963407ULL;
  uint64_t x = 67890ULL + tid;

  // Do measurement
  do {
    for (uint32_t i = 0; i < loop_length; ++i) {
      // Generate next random number with LCG
      // Note: wrap modulo 2^64 is defined by C/C++ standard
      x = a * x + c;

      // Write to a random location within data range
      uint64_t index = FAST_MODULO(x, size);
      auto *item = reinterpret_cast<std::atomic<uint32_t> *>(&data[index]);
      uint32_t expected = (uint32_t)index;
      uint32_t new_val = (uint32_t)x;
      std::atomic_compare_exchange_strong(item, &expected, new_val);
    }

    mem_accesses += loop_length;
  } while ((get_clock(stop), stop) - start < target_cycles_i);

  sum = stop - start;

  // Write result
  *measured_cycles = sum;
  *memory_accesses = mem_accesses;
}
