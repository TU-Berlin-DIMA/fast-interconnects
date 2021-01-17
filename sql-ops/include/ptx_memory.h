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

#ifndef PTX_MEMORY_H
#define PTX_MEMORY_H

/*
 * Cache at global level
 *
 * Loads from L2 cache, by-passing L1 cache.
 */
template <typename T>
__device__ __forceinline__ T ptx_load_cache_global(const T *addr);

template <>
__device__ __forceinline__ unsigned int ptx_load_cache_global<>(
    const unsigned int *addr) {
  unsigned int return_value;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(return_value) : "l"(addr));
  return return_value;
}

template <>
__device__ __forceinline__ int ptx_load_cache_global<>(const int *addr) {
  int return_value;
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(return_value) : "l"(addr));
  return return_value;
}

template <>
__device__ __forceinline__ long long ptx_load_cache_global<>(
    const long long *addr) {
  long long return_value;
  asm volatile("ld.global.cg.s64 %0, [%1];" : "=l"(return_value) : "l"(addr));
  return return_value;
}

template <>
__device__ __forceinline__ longlong2
ptx_load_cache_global<>(const longlong2 *addr) {
  longlong2 return_value;
  asm volatile("ld.global.cg.v2.s64 {%0, %1}, [%2];"
               : "=l"(return_value.x), "=l"(return_value.y)
               : "l"(addr));
  return return_value;
}

template <>
__device__ __forceinline__ unsigned long long ptx_load_cache_global<>(
    const unsigned long long *addr) {
  unsigned long long return_value;
  asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(return_value) : "l"(addr));
  return return_value;
}

/*
 * Cache streaming, likely accessed once
 *
 * Load with first-eviction policy in L1 and L2 to limit cache pollution.
 */
template <typename T>
__device__ __forceinline__ T ptx_load_cache_streaming(const T *addr);

template <>
__device__ __forceinline__ int ptx_load_cache_streaming<>(const int *addr) {
  int return_value;
  asm volatile("ld.global.cs.s32 %0, [%1];" : "=r"(return_value) : "l"(addr));
  return return_value;
}

template <>
__device__ __forceinline__ int2 ptx_load_cache_streaming<>(const int2 *addr) {
  int2 return_value;
  asm volatile("ld.global.cs.v2.s32 {%0, %1}, [%2];"
               : "=r"(return_value.x), "=r"(return_value.y)
               : "l"(addr));
  return return_value;
}

template <>
__device__ __forceinline__ long long ptx_load_cache_streaming<>(
    const long long *addr) {
  long long return_value;
  asm volatile("ld.global.cs.s64 %0, [%1];" : "=l"(return_value) : "l"(addr));
  return return_value;
}

template <>
__device__ __forceinline__ longlong2
ptx_load_cache_streaming<>(const longlong2 *addr) {
  longlong2 return_value;
  asm volatile("ld.global.cs.v2.s64 {%0, %1}, [%2];"
               : "=l"(return_value.x), "=l"(return_value.y)
               : "l"(addr));
  return return_value;
}

/*
 * Don't cache and fetch again
 *
 * Invalidates and discardes matching L2 cacheline and loads again from memory.
 */
template <typename T>
__device__ __forceinline__ T ptx_load_nocache(const T *addr);

template <>
__device__ __forceinline__ int ptx_load_nocache<>(const int *addr) {
  int return_value;
  asm volatile("ld.global.cv.s32 %0, [%1];" : "=r"(return_value) : "l"(addr));
  return return_value;
}

template <>
__device__ __forceinline__ long long ptx_load_nocache<>(const long long *addr) {
  long long return_value;
  asm volatile("ld.global.cv.s64 %0, [%1];" : "=l"(return_value) : "l"(addr));
  return return_value;
}

/*
 * Cache at global level
 *
 * Stores only in L2 cache, by-passing L1 cache.
 */
template <typename T>
__device__ __forceinline__ void ptx_store_cache_global(T *addr, T value);

template <>
__device__ __forceinline__ void ptx_store_cache_global<>(int *addr, int value) {
  asm volatile("st.global.cg.s32 [%0], %1;" : : "l"(addr), "r"(value));
}

template <>
__device__ __forceinline__ void ptx_store_cache_global<>(unsigned int *addr,
                                                         unsigned int value) {
  asm volatile("st.global.cg.u32 [%0], %1;" : : "l"(addr), "r"(value));
}

template <>
__device__ __forceinline__ void ptx_store_cache_global<>(long long *addr,
                                                         long long value) {
  asm volatile("st.global.cg.s64 [%0], %1;" : : "l"(addr), "l"(value));
}

template <>
__device__ __forceinline__ void ptx_store_cache_global<>(
    unsigned long long *addr, unsigned long long value) {
  asm volatile("st.global.cg.u64 [%0], %1;" : : "l"(addr), "l"(value));
}

template <>
__device__ __forceinline__ void ptx_store_cache_global<>(longlong2 *addr,
                                                         longlong2 value) {
  asm volatile("st.global.cg.v2.s64 [%0], {%1, %2};"
               :
               : "l"(addr), "l"(value.x), "l"(value.y));
}

/*
 * Cache streaming, likely accessed once
 *
 * Stores by allocating cacheline with first-eviction policy in L1 and L2 to
 * limit cache pollution.
 */
template <typename T>
__device__ __forceinline__ void ptx_store_cache_streaming(T *addr, T value);

template <>
__device__ __forceinline__ void ptx_store_cache_streaming<>(int *addr,
                                                            int value) {
  asm volatile("st.global.cs.s32 [%0], %1;" : : "l"(addr), "r"(value));
}

template <>
__device__ __forceinline__ void ptx_store_cache_streaming<>(int2 *addr,
                                                            int2 value) {
  asm volatile("st.global.cs.v2.s32 [%0], {%1, %2};"
               :
               : "l"(addr), "r"(value.x), "r"(value.y));
}

template <>
__device__ __forceinline__ void ptx_store_cache_streaming<>(long long *addr,
                                                            long long value) {
  asm volatile("st.global.cs.s64 [%0], %1;" : : "l"(addr), "l"(value));
}

template <>
__device__ __forceinline__ void ptx_store_cache_streaming<>(longlong2 *addr,
                                                            longlong2 value) {
  asm volatile("st.global.cs.v2.s64 [%0], {%1, %2};"
               :
               : "l"(addr), "l"(value.x), "l"(value.y));
}

/*
 * Don't cache and write-through
 *
 * Writes through the L2 cache to memory.
 */
template <typename T>
__device__ __forceinline__ void ptx_store_nocache(T *addr, T value);

template <>
__device__ __forceinline__ void ptx_store_nocache<>(int *addr, int value) {
  asm volatile("st.global.wt.s32 [%0], %1;" : : "l"(addr), "r"(value));
}

template <>
__device__ __forceinline__ void ptx_store_nocache<>(long long *addr,
                                                    long long value) {
  asm volatile("st.global.wt.s64 [%0], %1;" : : "l"(addr), "l"(value));
}

/*
 * Prefetch into L1 data cache
 */
__device__ __forceinline__ void ptx_prefetch_l1(void const *const ptr) {
  asm("prefetch.global.L1 [%0];" : : "l"(ptr));
}

/*
 * Prefetch into L2 data cache
 */
__device__ __forceinline__ void ptx_prefetch_l2(void const *const ptr) {
  asm("prefetch.global.L2 [%0];" : : "l"(ptr));
}

#endif /* PTX_MEMORY_H */
