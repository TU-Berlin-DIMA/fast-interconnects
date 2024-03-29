// Copyright 2020-2022 Clemens Lutz
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

#include <cstdint>

// #define MIN_WARMUP_CYCLES 1000000U
#define MIN_WARMUP_CYCLES 0U

// X mod Y, assuming that Y is a power of 2
#define FAST_MODULO(X, Y) ((X) & ((Y)-1U))

// Initialize the data buffer with strides
//
// `start_offset` optimizes initialization when only the data size is changed,
// but the stride stays the same. In that case, updating the last #stride
// indices is sufficient.
//
// No assumptions are made about `stride` and `data`. They could be a
// non-power-of-two number.
extern "C" __global__ void initialize_strides(uint64_t *const __restrict__ data,
                                              size_t const size,
                                              size_t const start_offset,
                                              size_t const stride) {
  for (uint64_t i = start_offset + blockDim.x * blockIdx.x + threadIdx.x;
       i < size; i += gridDim.x * blockDim.x) {
    data[i] = (i + stride) % size;
  }
}

// Stride over the data and measure the access latency
//
// Pre-conditions:
//  - Launch kernel must have exactly one thread.
//  - `iterations` must be a power of two, and greater or equal to
//    tlb_data_points.
extern "C" __global__ void tlb_stride_single_thread(
    uint64_t const *const __restrict__ data, size_t const size,
    uint32_t const iterations, size_t const tlb_data_points,
    uint32_t *const __restrict__ cycles, uint64_t *const __restrict__ index) {
  extern __shared__ uint64_t shared_mem[];

  uint64_t pos = 0;
  uint64_t dependency = 0;  // Prevent compiler from optimizing away the loop

  // Warm-up the cache
  for (uint64_t c = clock64(); clock64() - c < MIN_WARMUP_CYCLES;) {
    for (uint32_t i = 0; i < iterations; ++i) {
      asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(pos) : "l"(&data[pos]));
      dependency += pos;
    }
  }

  // Note: Don't reset `pos` after warm-up. This avoids cache hits at
  // beginning of loop.

  // Initialize shared memory
  uint64_t *const __restrict__ s_index =
      reinterpret_cast<uint64_t *>(shared_mem);
  uint32_t *const __restrict__ s_cycles =
      reinterpret_cast<uint32_t *>(&s_index[tlb_data_points]);

  for (uint32_t i = threadIdx.x; i < tlb_data_points; i += blockDim.x) {
    s_index[i] = 0xFFFFFFFFFFFFFFFFULL;
  }

  __syncthreads();

  // Do measurement
  clock_type start = 0;
  clock_type stop = 0;

  for (uint32_t i = 0; i < iterations; ++i) {
    get_clock(start);

    // Effectively: pos = data[pos];
    //
    // Bypass the L1 cache, because the L1 is a virtually indexed cache
    // that doesn't require address translation.
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(pos) : "l"(&data[pos]));

    // Execute the write dependency on `pos` before stopping the clock to
    // wait until the read instruction is completed. The write doesn't seem
    // to influence the timing when compared to (sum / iterations).
    s_index[FAST_MODULO(i, tlb_data_points)] = pos;
    get_clock(stop);

    s_cycles[FAST_MODULO(i, tlb_data_points)] =
        static_cast<uint32_t>(stop - start);
  }

  // Write result
  for (uint32_t i = 0; i < tlb_data_points; ++i) {
    cycles[i] = s_cycles[i];
    index[i] = s_index[i];
  }

  // Prevent compiler optimization
  // Note that the branch must never be taken
  if (iterations == 0xFFFFFFFFU) {
    cycles[0] = dependency;
  }
}
