/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2020 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

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
