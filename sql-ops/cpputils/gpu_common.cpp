/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

#define CUDA_MODIFIER

#include <gpu_common.h>

// Returns the log2 of the next-lower power of two
int log2_floor_power_of_two(unsigned int x) {
  return 32 - __builtin_clz(x) - 1;
}
int log2_floor_power_of_two(unsigned long x) {
  return sizeof(unsigned long) * 8 - __builtin_clzl(x) - 1;
}
int log2_floor_power_of_two(unsigned long long x) {
  return 64 - __builtin_clzll(x) - 1;
}
int log2_floor_power_of_two(int x) {
  return log2_floor_power_of_two(static_cast<unsigned int>(x));
}
int log2_floor_power_of_two(long long x) {
  return log2_floor_power_of_two(static_cast<unsigned long long>(x));
}

// Returns the log2 of the next-higher power of two
int log2_ceil_power_of_two(unsigned int x) {
  return 32 - __builtin_clz(x - 1U);
}
int log2_ceil_power_of_two(unsigned long x) {
  return sizeof(unsigned long) * 8 - __builtin_clzl(x - 1UL);
}
int log2_ceil_power_of_two(unsigned long long x) {
  return 64 - __builtin_clzll(x - 1ULL);
}
int log2_ceil_power_of_two(int x) {
  return log2_ceil_power_of_two(static_cast<unsigned int>(x));
}
int log2_ceil_power_of_two(long long x) {
  return log2_ceil_power_of_two(static_cast<unsigned long long>(x));
}
