/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2018, Lutz, Clemens <lutzcle@cml.li>
 */

#include <cstdint>
#include <chrono>

#include <timer.hpp>

extern "C"
uint64_t cpu_stride(uint32_t *data, uint32_t iterations) {
    uint32_t pos = 0;

    // Warm-up
    for (uint32_t i = 0; i < iterations; ++i) {
        pos = data[pos];

        // Prevent compiler optimization
        __asm__ __volatile__ ("" : "=r" (pos) : "0" (pos) : "memory");
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
        __asm__ __volatile__ ("" : "=r" (pos) : "0" (pos) : "memory");
    }

    uint64_t nanos = latency_timer.stop<std::chrono::nanoseconds>();
    nanos = nanos / ((uint64_t) iterations);

    return nanos;
}

