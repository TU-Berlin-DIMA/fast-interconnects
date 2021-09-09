/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

#ifndef CPU_MEMORY_INSTRUCTIONS_H
#define CPU_MEMORY_INSTRUCTIONS_H

#include <cstddef>

#if defined(__x86_64__)
#include <immintrin.h>
#endif /* defined(__x86_64__) */

/*
 * Returns the type as a scalar value
 */
template <typename T>
inline unsigned long long scalar(T const value) {
  return static_cast<unsigned long long>(value);
}

#if defined(__SSE2__)
template <>
inline unsigned long long scalar(__m128i const value) {
  // Copy the lower 64-bit integer
  return _mm_cvtsi128_si64(value);
}
#endif /* defined(__SSE2__) */

/*
 * Assigns a scalar value
 */
template <typename T>
inline T assign(std::size_t const value) {
  return static_cast<T>(value);
}

#if defined(__SSE2__)
template <>
inline __m128i assign<__m128i>(std::size_t const value) {
  return _mm_set_epi64x(value, value);
}
#endif /* defined(__SSE2__) */

/*
 * Load the value at the given address
 */
template <typename T>
inline T load(T const *const addr) {
  return *addr;
}

#if defined(__powerpc64__)
/*
 * Using an atomic load forces the compiler to generate a load instruction with
 * the specified width (i.e., 32 bit, 64 bit or 128 bit).
 *
 * Effectively: dummy += data[i];
 *
 * On a POWER9 CPU in little-endian mode, GCC 8.3 generates two 64-bit loads
 * (ld) instead of one 128-bit load (lq). The reason is that these may be
 * faster. Quote from the POWER ISA manual v3.0B, S. 3.3.4, p. 58:
 *
 * "The lq and stq instructions exist primarily to permit software to access
 * quadwords in storage "atomically". Because GPRs are 64 bits long, the
 * Fixed-Point Facility on many designs is optimized for storage accesses of at
 * most eight bytes. On such designs, the quadword atomicity required for lq
 * and stq makes these instructions complex to implement, with the result that
 * the instructions may perform less well on these designs than the
 * corresponding two Load Double-word or Store Doubleword instructions."
 *
 * See GCC source code:
 * https://github.com/gcc-mirror/gcc/blob/d03ca8a6148f55e119b8220a9c65147173b32065/gcc/config/rs6000/rs6000.c#L4019
 */
template <>
inline unsigned __int128 load(unsigned __int128 const *const addr) {
  return __atomic_load_n(addr, __ATOMIC_RELAXED);
}

#elif defined(__SSE4_1__)
/*
 * Load a 16-byte aligned address without polluting caches
 *
 * https://software.intel.com/sites/landingpage/IntrinsicsGuide/#!=undefined&expand=6894,4257,4257,6884&cats=Load&techs=SSE4_1
 */
template <>
inline __m128i load(__m128i const *const addr) {
  return _mm_stream_load_si128(const_cast<__m128i *>(addr));
}
#endif /* defined(__powerpc64__) */

/*
 * Store the value to the given address
 */
template <typename T>
inline void store(T *const addr, T const value) {
  *addr = value;
}

#if defined(__powerpc64__)
/*
 * Using an atomic store forces the compiler to generate a store instruction
 * with the specified width (i.e., 32 bit, 64 bit or 128 bit).
 *
 * Effectively: data[i] = i;
 *
 * On a POWER9 CPU in little-endian mode, GCC 8.3 generates two 64-bit stores
 * (std) instead of one 128-bit store (stq). The reason is that these may be
 * faster. Quote from the POWER ISA manual v3.0B, S. 3.3.4, p.  58:
 *
 * "The lq and stq instructions exist primarily to permit software to access
 * quadwords in storage "atomically". Because GPRs are 64 bits long, the
 * Fixed-Point Facility on many designs is optimized for storage accesses of at
 * most eight bytes. On such designs, the quadword atomicity required for lq
 * and stq makes these instructions complex to implement, with the result that
 * the instructions may perform less well on these designs than the
 * corresponding two Load Double-word or Store Doubleword instructions."
 *
 * See GCC source code:
 * https://github.com/gcc-mirror/gcc/blob/d03ca8a6148f55e119b8220a9c65147173b32065/gcc/config/rs6000/rs6000.c#L4019
 */
template <>
inline void store(unsigned __int128 *const addr,
                  unsigned __int128 const value) {
  __atomic_store_n(addr, value, __ATOMIC_RELAXED);
}

#elif defined(__SSE2__)
/*
 * Store a 16-byte aligned address without polluting caches
 *
 * https://software.intel.com/sites/landingpage/IntrinsicsGuide/#!=undefined&expand=4257,4257,6894,6894&cats=Store&techs=SSE2
 *
 */
template <>
inline void store(__m128i *const addr, __m128i const value) {
  _mm_stream_si128(addr, value);
}
#endif /* defined(__powerpc64__) */

#endif /* CPU_MEMORY_INSTRUCTIONS_H */
