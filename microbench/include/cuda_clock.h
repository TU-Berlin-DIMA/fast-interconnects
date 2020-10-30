/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2020 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

#if __CUDA_ARCH__ >= 700
typedef unsigned long long clock_type;
__device__ inline void get_clock(clock_type& dst) { dst = clock64(); }
#else
typedef unsigned int clock_type;
__device__ inline void get_clock(clock_type& dst) {
  asm volatile("mov.u32 %0, %%clock;" : "=r"(dst));
}
#endif
