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
