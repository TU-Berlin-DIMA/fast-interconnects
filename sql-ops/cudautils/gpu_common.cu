// Copyright 2020-2022 Clemens Lutz
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
