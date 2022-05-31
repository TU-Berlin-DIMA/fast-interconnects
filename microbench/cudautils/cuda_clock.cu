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

#define CLOCK_TUNE_ITERS 1000U

// Measures the overhead of `get_clock()`.
//
// `get_clock()` is implemented as a native special register on each
// architecture. On Volta and later, `get_clock()` is implemented as
// `clock64()`. On Pascal and earlier, `get_clock()` is implemented as
// `clock()`. This avoids the high overhead of emulating `clock64()` when it
// isn't native.
//
// Emulating `clock64()` has a overhead of 59 cycles on Pascal (GTX 1080). For
// comparison, the native `clock64()` on Volta (Tesla V100) takes only 12
// cycles.
//
// Note that `clock()` is 32-bits and will overflow after about 2 seconds.
extern "C" __global__ void cycle_counter_overhead(
    uint32_t *const __restrict__ overhead,
    uint32_t *const __restrict__ fake_dependency) {
  clock_type start = 0;
  clock_type stop = 0;
  clock_type clock_dependency;

  get_clock(clock_dependency);
  get_clock(start);
  for (uint32_t i = 0; i < CLOCK_TUNE_ITERS; ++i) {
    clock_type clock_dst;
    get_clock(clock_dst);
    clock_dependency ^= clock_dst;
  }
  get_clock(stop);
  *overhead = (stop - start) / CLOCK_TUNE_ITERS;
  *fake_dependency = clock_dependency;
}
