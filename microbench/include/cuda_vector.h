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

#ifndef CUDA_VECTOR_H
#define CUDA_VECTOR_H

#include <cstdint>

/*
 * Create the specified scalar or CUDA vector type from a 64-bit integer
 *
 * For vectors, broadcast the initialization value to all vector dimensions.
 */
template <typename T>
inline __device__ T make_type(uint64_t value);

template <>
inline __device__ uint32_t make_type<uint32_t>(uint64_t value) {
  return value;
}

template <>
inline __device__ uint64_t make_type<uint64_t>(uint64_t value) {
  return value;
}

template <>
inline __device__ ulonglong2 make_type<ulonglong2>(uint64_t value) {
  return make_ulonglong2(value, value);
}

/*
 * Implement the addition assignment operator (`+=`) for CUDA vector types
 */
inline __device__ ulonglong2& operator+=(ulonglong2& lhs, ulonglong2& rhs) {
  lhs = make_ulonglong2(lhs.x + rhs.x, lhs.y + rhs.y);
  return lhs;
}

#endif /* CUDA_VECTOR_H */
