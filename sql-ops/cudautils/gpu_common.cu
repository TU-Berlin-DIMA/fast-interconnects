/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2020-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

#include <gpu_common.h>

#include <cassert>

// Returns the log2 of the next-lower power of two
__device__ int log2_floor_power_of_two(int x) { return 32 - __clz(x) - 1; }

__device__ int log2_floor_power_of_two(long long x) {
  return 64 - __clzll(x) - 1;
}

__device__ int log2_floor_power_of_two(unsigned int x) {
  return log2_floor_power_of_two(static_cast<int>(x));
}

__device__ int log2_floor_power_of_two(unsigned long x) {
  if (sizeof(unsigned long) == sizeof(unsigned long long)) {
    return log2_floor_power_of_two(static_cast<unsigned long long>(x));
  } else if (sizeof(unsigned long) == sizeof(unsigned int)) {
    return log2_floor_power_of_two(static_cast<unsigned int>(x));
  } else {
    assert(false);
  }
}

__device__ int log2_floor_power_of_two(unsigned long long x) {
  return log2_floor_power_of_two(static_cast<long long>(x));
}

// Returns the log2 of the next-higher power of two
__device__ int log2_ceil_power_of_two(int x) { return 32 - __clz(x - 1); }

__device__ int log2_ceil_power_of_two(long long x) {
  return 64 - __clzll(x - 1LL);
}

__device__ int log2_ceil_power_of_two(unsigned int x) {
  return log2_ceil_power_of_two(static_cast<int>(x));
}

__device__ int log2_ceil_power_of_two(unsigned long x) {
  return log2_ceil_power_of_two(static_cast<unsigned long long>(x));
}

__device__ int log2_ceil_power_of_two(unsigned long long x) {
  return log2_ceil_power_of_two(static_cast<long long>(x));
}
