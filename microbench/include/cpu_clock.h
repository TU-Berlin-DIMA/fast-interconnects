/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

#ifndef CPU_CLOCK_H
#define CPU_CLOCK_H

#if defined(__x86_64__)
#include <x86intrin.h>

typedef unsigned long long clock_type;

inline void get_clock(clock_type& dst) { dst = __rdtsc(); }
#elif defined(__powerpc64__)
typedef unsigned long long clock_type;

inline void get_clock(clock_type& dst) { dst = __builtin_ppc_get_timebase(); }
#else
#warning Cycle counter for this CPU architecture is not defined.
#endif

#endif /* CPU_CLOCK_H */
