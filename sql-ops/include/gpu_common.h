/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2020 Clemens Lutz, German Research Center for Artificial
 * Intelligence
 * Author: Clemens Lutz, DFKI GmbH <clemens.lutz@dfki.de>
 */

#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#include <ptx_memory.h>

#ifndef CUDA_MODIFIER
#define CUDA_MODIFIER __device__
#endif

// A key-value tuple.
//
// Note that the struct's layout must be kept in sync with its counterpart in
// Rust.
template <typename K, typename V>
struct Tuple {
  K key;
  V value;

  __device__ __forceinline__ void load(Tuple<int, int> const &src) {
    int2 tmp = *reinterpret_cast<int2 const *>(&src);
    this->key = tmp.x;
    this->value = tmp.y;
  }

  __device__ __forceinline__ void load(Tuple<long long, long long> const &src) {
    longlong2 tmp = *reinterpret_cast<longlong2 const *>(&src);
    this->key = tmp.x;
    this->value = tmp.y;
  }

  __device__ __forceinline__ void store(Tuple<int, int> &dst) {
    int2 tmp = make_int2(this->key, this->value);
    *reinterpret_cast<int2 *>(&dst) = tmp;
  }

  __device__ __forceinline__ void store(Tuple<long long, long long> &dst) {
    longlong2 tmp = make_longlong2(this->key, this->value);
    *reinterpret_cast<longlong2 *>(&dst) = tmp;
  }
};

// A hash table entry.
//
// Note that the struct's layout must be kept in sync with its counterpart in
// Rust.
template <typename K, typename P>
struct HtEntry {
  K key;
  P payload;
};

template <typename K>
CUDA_MODIFIER constexpr K null_key();

template <>
CUDA_MODIFIER constexpr int null_key<int>() {
  return 0xFFFFFFFF;
}

template <>
CUDA_MODIFIER constexpr long long null_key<long long>() {
  return 0xFFFFFFFFFFFFFFFFll;
}

#endif /* GPU_COMMON_H */
