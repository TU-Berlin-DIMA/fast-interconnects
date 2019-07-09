/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2018, Lutz, Clemens <lutzcle@cml.li>
 */

#include <cstdint>
#include <iostream>

#ifdef __x86_64__
#define CLOCK_COUNTER __builtin_ia32_rdtsc()
#define FREQUENCY 1
#endif
#ifdef __powerpc64__
#define CLOCK_COUNTER __builtin_ppc_get_timebase()
#endif


#include <common.h>
#include <timer.hpp>

// Note: GCC intrinsics are not recognized by nvcc. Work-around is to put them in a separate .cpp file and then link them into the executable

// For GCC built-ins on PPC64, see:
// https://gcc.gnu.org/onlinedocs/gcc/Basic-PowerPC-Built-in-Functions-Available-on-all-Configurations.html

extern "C" void cpu_loop(volatile CacheLine *data, uint32_t iterations, volatile Signal *signal, volatile uint64_t *result)
{
    uint64_t sum = 0;
    Timer::Timer sync_time;

    // Wait for signal
    while (*signal == WAIT);

    // Run
    for (uint32_t i = 0; i < iterations; ++i) {

        sync_time.start();

        while (data->value == CPU);
        data->value = CPU;

        sum += sync_time.stop<std::chrono::nanoseconds>();

    }

    // Write average result
    *result = (sum / iterations);
}

