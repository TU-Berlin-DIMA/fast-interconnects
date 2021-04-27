/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

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
