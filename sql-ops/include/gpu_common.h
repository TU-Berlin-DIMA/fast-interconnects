/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2020-2021 Clemens Lutz, German Research Center for Artificial
 * Intelligence
 * Author: Clemens Lutz, DFKI GmbH <clemens.lutz@dfki.de>
 */

#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#include <ptx_memory.h>

#ifndef CUDA_MODIFIER
#define CUDA_MODIFIER __device__
#endif

// Returns the log2 of the next-lower power of two
__device__ int log2_floor_power_of_two(int x);

// Returns the log2 of the next-higher power of two
__device__ int log2_ceil_power_of_two(int x);

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

  __device__ __forceinline__ void load_streaming(Tuple<int, int> const &src) {
    int2 tmp = ptx_load_cache_streaming(reinterpret_cast<int2 const *>(&src));
    this->key = tmp.x;
    this->value = tmp.y;
  }

  __device__ __forceinline__ void load_streaming(
      Tuple<long long, long long> const &src) {
    longlong2 tmp =
        ptx_load_cache_streaming(reinterpret_cast<longlong2 const *>(&src));
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

  __device__ __forceinline__ void store_streaming(Tuple<int, int> &dst) {
    int2 tmp = make_int2(this->key, this->value);
    ptx_store_cache_streaming(reinterpret_cast<int2 *>(&dst), tmp);
  }

  __device__ __forceinline__ void store_streaming(
      Tuple<long long, long long> &dst) {
    longlong2 tmp = make_longlong2(this->key, this->value);
    ptx_store_cache_streaming(reinterpret_cast<longlong2 *>(&dst), tmp);
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

// Multiply-shift hash function
//
// Requirement: hash factor is an odd 64-bit integer
// See Richter et al., Seven-Dimensional Analysis of Hashing Methods
template <typename T>
CUDA_MODIFIER T mult_shift_hash(T value) {}

template <>
CUDA_MODIFIER __forceinline__ int mult_shift_hash(int value) {
  constexpr unsigned int HASH_FACTOR = 1234567891u;
  return static_cast<int>(static_cast<unsigned int>(value) * HASH_FACTOR);
}

template <>
CUDA_MODIFIER __forceinline__ long long mult_shift_hash(long long value) {
  constexpr unsigned long long HASH_FACTOR = 123456789123456789llu;
  return static_cast<long long>(static_cast<unsigned long long>(value) *
                                HASH_FACTOR);
}

// Murmur3 hash function
//
// Copied from
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp Source
// is in public domain Only finalizer for 32-bit and 64-bit results is used, as
// in Richter et al. See Richter et al., Seven-Dimensional Analysis of Hashing
// Methods
template <typename T>
CUDA_MODIFIER T murmur3_hash(T value) {}

template <>
CUDA_MODIFIER __forceinline__ int murmur3_hash(int value) {
  unsigned int h = static_cast<unsigned int>(value);

  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  return static_cast<int>(h);
}

template <>
CUDA_MODIFIER __forceinline__ long long murmur3_hash(long long value) {
  unsigned long long k = static_cast<unsigned long long>(value);

  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdull;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ull;
  k ^= k >> 33;

  return static_cast<long long>(k);
}

// Alias defining the default hash function
template <typename T>
// constexpr auto hash = &mult_shift_hash<T>;
constexpr auto hash = &murmur3_hash<T>;

#endif /* GPU_COMMON_H */
