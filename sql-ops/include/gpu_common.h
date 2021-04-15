/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2020-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#ifdef __CUDACC__
#include <ptx_memory.h>
#endif /* __CUDACC__ */

#ifndef __CUDACC__
#define __forceinline__ __attribute__((always_inline)) inline
#endif /* not __CUDACC__ */

#ifndef CUDA_MODIFIER
#define CUDA_MODIFIER __device__
#endif

// #ifdef __CUDACC__
// Returns the log2 of the next-lower power of two
CUDA_MODIFIER int log2_floor_power_of_two(int x);
CUDA_MODIFIER int log2_floor_power_of_two(long long x);
CUDA_MODIFIER int log2_floor_power_of_two(unsigned int x);
CUDA_MODIFIER int log2_floor_power_of_two(unsigned long x);
CUDA_MODIFIER int log2_floor_power_of_two(unsigned long long x);

// Returns the log2 of the next-higher power of two
CUDA_MODIFIER int log2_ceil_power_of_two(int x);
CUDA_MODIFIER int log2_ceil_power_of_two(long long x);
CUDA_MODIFIER int log2_ceil_power_of_two(unsigned int x);
CUDA_MODIFIER int log2_ceil_power_of_two(unsigned long x);
CUDA_MODIFIER int log2_ceil_power_of_two(unsigned long long x);
// #endif /* __CUDACC__ */

// A key-value tuple.
//
// Note that the struct's layout must be kept in sync with its counterpart in
// Rust.
template <typename K, typename V>
struct Tuple {
  K key;
  V value;

#ifdef __CUDACC__
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
#endif /* __CUDACC__ */
};

// A hash table entry.
//
// Note that the struct's layout must be kept in sync with its counterpart in
// Rust.
template <typename K, typename V>
using HtEntry = Tuple<K, V>;

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
// Takes a value and returns the hash of the value, modulo the number of
// buckets. Takes the number of buckets as log2(buckets).
//
// Requirement: hash factor is an odd integer
// Original source: Dietzfelbinger et al. "A Reliable Randomized Algorithm for
// the Closest-Pair Problem"
//
// Knuth specifies using the golden ratio in "The Art of Computer Programming"
// Volume 3, Section 6.4, Figure 37 pp
//
// Compute golden ratio with:
// floor(2^32 * (-1 + sqrt(5)) / 2) = 2654435769u
// floor(2^64 * (-1 + sqrt(5)) / 2) = 11400714819323198485
// e.g., using Wolfram Alpha
//
// See also Richter et al., Seven-Dimensional Analysis of Hashing Methods
template <typename T>
CUDA_MODIFIER T mult_shift_hash(T value, unsigned int log2_buckets);

template <>
CUDA_MODIFIER __forceinline__ int mult_shift_hash(int value,
                                                  unsigned int log2_buckets) {
  constexpr unsigned int HASH_FACTOR = 2654435769u;
  constexpr unsigned int INT_BITS = 32u;

  unsigned int product = static_cast<unsigned int>(value) * HASH_FACTOR;
  unsigned int shifted = product >> (INT_BITS - log2_buckets);
  return static_cast<int>(shifted);
}

template <>
CUDA_MODIFIER __forceinline__ long long mult_shift_hash(
    long long value, unsigned int log2_buckets) {
  constexpr unsigned long long HASH_FACTOR = 11400714819323198485llu;
  constexpr unsigned int INT_BITS = 64u;

  unsigned long long product =
      static_cast<unsigned long long>(value) * HASH_FACTOR;
  unsigned long long shifted =
      product >> static_cast<unsigned long long>(INT_BITS - log2_buckets);
  return static_cast<long long>(shifted);
}

// Murmur3 hash function
//
// Takes a value and returns the hash of the value, modulo the number of
// buckets. Takes the number of buckets as log2(buckets).
//
// Copied from
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
//
// Source is in public domain Only finalizer for 32-bit and 64-bit results is
// used, as in Richter et al. See Richter et al., Seven-Dimensional Analysis of
// Hashing Methods
template <typename T>
CUDA_MODIFIER T murmur3_hash(T value, unsigned int log2_buckets);

template <>
CUDA_MODIFIER __forceinline__ int murmur3_hash(int value,
                                               unsigned int log2_buckets) {
  unsigned int h = static_cast<unsigned int>(value);

  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  unsigned int buckets_mask = (1u << log2_buckets) - 1u;
  return static_cast<int>(h & buckets_mask);
}

template <>
CUDA_MODIFIER __forceinline__ long long murmur3_hash(
    long long value, unsigned int log2_buckets) {
  unsigned long long k = static_cast<unsigned long long>(value);

  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdull;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ull;
  k ^= k >> 33;

  unsigned long long buckets_mask = (1llu << log2_buckets) - 1llu;
  return static_cast<long long>(k & buckets_mask);
}

// Alias defining the default hash function
template <typename T>
// constexpr auto hash = &mult_shift_hash<T>;
constexpr auto hash = &murmur3_hash<T>;

#endif /* GPU_COMMON_H */
