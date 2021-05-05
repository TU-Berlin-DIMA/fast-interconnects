/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

#include <chrono>
#include <cstdint>

#include <timer.hpp>

#if defined(__powerpc64__)
#include <ppc_intrinsics.h>
#endif

// Disable prefeching
#define PPC_TUNE_DSCR 1ULL

extern "C" uint64_t cpu_stride(uint32_t *data, uint32_t iterations) {
#if defined(__powerpc64__)
  __mtspr(PPC_DSCR, PPC_TUNE_DSCR);
#endif

  uint32_t pos = 0;

  // Warm-up
  for (uint32_t i = 0; i < iterations; ++i) {
    pos = data[pos];

    // Prevent compiler optimization
    __asm__ __volatile__("" : "=r"(pos) : "0"(pos) : "memory");
  }

  // Reset position
  if (pos != 0) {
    pos = 0;
  }

  // Measure
  Timer::Timer latency_timer;
  latency_timer.start();

  for (uint32_t i = 0; i < iterations; ++i) {
    pos = data[pos];

    // Prevent compiler optimization
    __asm__ __volatile__("" : "=r"(pos) : "0"(pos) : "memory");
  }

  uint64_t nanos = latency_timer.stop<std::chrono::nanoseconds>();
  nanos = nanos / ((uint64_t)iterations);

  return nanos;
}
