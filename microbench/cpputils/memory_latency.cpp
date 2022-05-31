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
