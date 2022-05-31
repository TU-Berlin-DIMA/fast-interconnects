// Copyright 2021-2022 Clemens Lutz
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
