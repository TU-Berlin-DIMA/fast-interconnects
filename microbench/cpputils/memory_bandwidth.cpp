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

#include <cpu_clock.h>
#include <cpu_memory_instructions.h>

#include <cstdint>

#if defined(__powerpc64__)
#include <ppc_intrinsics.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#endif

// Disable strided prefetch and set maximum prefetch depth
#define PPC_TUNE_DSCR 7ULL

// X mod Y, assuming that Y is a power of 2
#define FAST_MODULO(X, Y) ((X) & ((Y)-1U))

// Alignment size for memory accesses
#define ALIGN_BYTES 128UL

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

  std::size_t constexpr align_mask = ~(ALIGN_BYTES / sizeof(T) - 1UL);
  std::size_t const chunk_size =
      ((size + num_threads - 1) / num_threads) & align_mask;
  std::size_t const begin = tid * chunk_size;
  std::size_t const end =
      ((tid + 1) * chunk_size - 1 > size) ? size : (tid + 1) * chunk_size - 1;
  uint64_t sum = 0;
  clock_type start = 0;
  clock_type stop = 0;

  get_clock(start);

  T dummy = {0};
  for (std::size_t i = begin; i < end; ++i) {
    dummy += load(&data[i]);
  }

  get_clock(stop);
  sum = stop - start;

  // Write result
  *measured_cycles = sum;
  *memory_accesses = end - begin;

  // Prevent compiler optimization
  if (scalar(dummy) == 0) {
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

  std::size_t constexpr align_mask = ~(ALIGN_BYTES / sizeof(T) - 1UL);
  std::size_t const chunk_size =
      ((size + num_threads - 1) / num_threads) & align_mask;
  std::size_t const begin = tid * chunk_size;
  std::size_t const end =
      ((tid + 1) * chunk_size - 1 > size) ? size : (tid + 1) * chunk_size - 1;
  uint64_t sum = 0;
  clock_type start = 0;
  clock_type stop = 0;

  get_clock(start);

  for (std::size_t i = begin; i < end; ++i) {
    store(&data[i], assign<T>(i));
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

  std::size_t constexpr align_mask = ~(ALIGN_BYTES / sizeof(T) - 1UL);
  std::size_t const chunk_size =
      ((size + num_threads - 1) / num_threads) & align_mask;
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
      dummy += load(&data[index]);
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
      store(&data[index], assign<T>(x));
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
#if defined(__SSE2__)
MAKE_BENCHMARK(cpu_read_bandwidth_seq, 16B, __m128i)
#elif __SIZEOF_INT128__ == 16
MAKE_BENCHMARK(cpu_read_bandwidth_seq, 16B, unsigned __int128)
#endif

MAKE_BENCHMARK(cpu_write_bandwidth_seq, 4B, uint32_t)
MAKE_BENCHMARK(cpu_write_bandwidth_seq, 8B, uint64_t)
#if defined(__SSE2__)
MAKE_BENCHMARK(cpu_write_bandwidth_seq, 16B, __m128i)
#elif __SIZEOF_INT128__ == 16
MAKE_BENCHMARK(cpu_write_bandwidth_seq, 16B, unsigned __int128)
#endif

MAKE_BENCHMARK(cpu_cas_bandwidth_seq, 4B, uint32_t)
MAKE_BENCHMARK(cpu_cas_bandwidth_seq, 8B, uint64_t)
#if __SIZEOF_INT128__ == 16
MAKE_BENCHMARK(cpu_cas_bandwidth_seq, 16B, unsigned __int128)
#endif

MAKE_BENCHMARK(cpu_read_bandwidth_lcg, 4B, uint32_t)
MAKE_BENCHMARK(cpu_read_bandwidth_lcg, 8B, uint64_t)
#if defined(__SSE2__)
MAKE_BENCHMARK(cpu_read_bandwidth_lcg, 16B, __m128i)
#elif __SIZEOF_INT128__ == 16
MAKE_BENCHMARK(cpu_read_bandwidth_lcg, 16B, unsigned __int128)
#endif

MAKE_BENCHMARK(cpu_write_bandwidth_lcg, 4B, uint32_t)
MAKE_BENCHMARK(cpu_write_bandwidth_lcg, 8B, uint64_t)
#if defined(__SSE2__)
MAKE_BENCHMARK(cpu_write_bandwidth_lcg, 16B, __m128i)
#elif __SIZEOF_INT128__ == 16
MAKE_BENCHMARK(cpu_write_bandwidth_lcg, 16B, unsigned __int128)
#endif

MAKE_BENCHMARK(cpu_cas_bandwidth_lcg, 4B, uint32_t)
MAKE_BENCHMARK(cpu_cas_bandwidth_lcg, 8B, uint64_t)
#if __SIZEOF_INT128__ == 16
MAKE_BENCHMARK(cpu_cas_bandwidth_lcg, 16B, unsigned __int128)
#endif
