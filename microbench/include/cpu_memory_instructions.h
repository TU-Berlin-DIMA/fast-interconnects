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
#endif /* defined(__powerpc64__) */

#endif /* CPU_MEMORY_INSTRUCTIONS_H */
