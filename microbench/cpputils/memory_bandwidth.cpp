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

#include <cstdint>

#if defined(__powerpc64__)
#include <ppc_intrinsics.h>
#endif

// Disable strided prefetch and set maximum prefetch depth
#define PPC_TUNE_DSCR 7ULL

// X mod Y, assuming that Y is a power of 2
#define FAST_MODULO(X, Y) (X & (Y - 1))

// FIXME: add const restrict to all data pointers

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
template <typename T>
void cpu_read_bandwidth_seq_kernel(T *data, std::size_t size,
                                   uint32_t const /* loop_length */,
                                   uint64_t const /* target_cycles */,
                                   uint64_t *const memory_accesses,
                                   uint64_t *const measured_cycles,
                                   std::size_t tid, std::size_t num_threads) {
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

  T dummy = {0};
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
template <typename T>
void cpu_write_bandwidth_seq_kernel(T *data, std::size_t size,
                                    uint32_t const /* loop_length */,
                                    uint64_t const /* target_cycles */,
                                    uint64_t *const memory_accesses,
                                    uint64_t *const measured_cycles,
                                    std::size_t tid, std::size_t num_threads) {
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
template <typename T>
void cpu_cas_bandwidth_seq_kernel(T *data, std::size_t size,
                                  uint32_t const /* loop_length */,
                                  uint64_t const /* target_cycles */,
                                  uint64_t *const memory_accesses,
                                  uint64_t *const measured_cycles,
                                  std::size_t tid, std::size_t num_threads) {
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
    T expected = static_cast<T>(i);
    T new_val = static_cast<T>(i + 1);
    __atomic_compare_exchange_n(&data[i], &expected, new_val, false,
                                __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
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
template <typename T>
void cpu_read_bandwidth_lcg_kernel(T *data, std::size_t size,
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
  T dummy = {0};
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
  if (sum == 0) {
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
template <typename T>
void cpu_write_bandwidth_lcg_kernel(T *data, std::size_t size,
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
template <typename T>
void cpu_cas_bandwidth_lcg_kernel(T *data, std::size_t size,
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
      T expected = static_cast<T>(index);
      T new_val = static_cast<T>(x);
      __atomic_compare_exchange_n(&data[index], &expected, new_val, false,
                                  __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }

    mem_accesses += loop_length;
  } while ((get_clock(stop), stop) - start < target_cycles_i);

  sum = stop - start;

  // Write result
  *measured_cycles = sum;
  *memory_accesses = mem_accesses;
}

// ============== Instantiate templates ==============

#define MAKE_BENCHMARK(FUNCTION_NAME, SUFFIX, DATA_TYPE)                  \
  extern "C" void FUNCTION_NAME##_##SUFFIX(                               \
      uint32_t *data, std::size_t size, uint32_t const loop_length,       \
      uint64_t const target_cycles, uint64_t *const memory_accesses,      \
      uint64_t *const measured_cycles, std::size_t tid,                   \
      std::size_t num_threads) {                                          \
    DATA_TYPE *typed_data = reinterpret_cast<DATA_TYPE *>(data);          \
    FUNCTION_NAME##_kernel<DATA_TYPE>(typed_data, size, loop_length,      \
                                      target_cycles, memory_accesses,     \
                                      measured_cycles, tid, num_threads); \
  }

MAKE_BENCHMARK(cpu_read_bandwidth_seq, 4B, uint32_t)
MAKE_BENCHMARK(cpu_read_bandwidth_seq, 8B, uint64_t)
#if __SIZEOF_INT128__ == 16
MAKE_BENCHMARK(cpu_read_bandwidth_seq, 16B, unsigned __int128)
#endif

MAKE_BENCHMARK(cpu_write_bandwidth_seq, 4B, uint32_t)
MAKE_BENCHMARK(cpu_write_bandwidth_seq, 8B, uint64_t)
#if __SIZEOF_INT128__ == 16
MAKE_BENCHMARK(cpu_write_bandwidth_seq, 16B, unsigned __int128)
#endif

MAKE_BENCHMARK(cpu_cas_bandwidth_seq, 4B, uint32_t)
MAKE_BENCHMARK(cpu_cas_bandwidth_seq, 8B, uint64_t)
#if __SIZEOF_INT128__ == 16
MAKE_BENCHMARK(cpu_cas_bandwidth_seq, 16B, unsigned __int128)
#endif

MAKE_BENCHMARK(cpu_read_bandwidth_lcg, 4B, uint32_t)
MAKE_BENCHMARK(cpu_read_bandwidth_lcg, 8B, uint64_t)
#if __SIZEOF_INT128__ == 16
MAKE_BENCHMARK(cpu_read_bandwidth_lcg, 16B, unsigned __int128)
#endif

MAKE_BENCHMARK(cpu_write_bandwidth_lcg, 4B, uint32_t)
MAKE_BENCHMARK(cpu_write_bandwidth_lcg, 8B, uint64_t)
#if __SIZEOF_INT128__ == 16
MAKE_BENCHMARK(cpu_write_bandwidth_lcg, 16B, unsigned __int128)
#endif

MAKE_BENCHMARK(cpu_cas_bandwidth_lcg, 4B, uint32_t)
MAKE_BENCHMARK(cpu_cas_bandwidth_lcg, 8B, uint64_t)
#if __SIZEOF_INT128__ == 16
MAKE_BENCHMARK(cpu_cas_bandwidth_lcg, 16B, unsigned __int128)
#endif
