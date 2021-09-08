/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2020 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

// NOTE: Keep this file in sync with sql-ops/include/ppc_intrinsics.h

#ifndef PPC_INTRINSICS_H
#define PPC_INTRINSICS_H

#define PPC_DSCR 3

// Set cacheline to zero
#define __dcbz(base) __asm__ volatile("dcbz 0,%0" ::"r"(base) : "memory")

// Cacheline transient store hint
#define __dcbtstt(base) __asm__ volatile("dcbtstt 0,%0" ::"r"(base) : "memory")

// Flush cacheline to memory
// Programs which manage coherence in software must use this dcbf command
#define __dcbf(base) __asm__ volatile("dcbf 0,%0,0" ::"r"(base) : "memory")

// Flush local, i.e., flush cacheline to L3 cache
// Serves as a transient store hint
#define __dcbfl(base) __asm__ volatile("dcbf 0,%0,1" ::"r"(base) : "memory")

// Flush local primary, i.e., flush only L1 cacheline of the executing CPU core
#define __dcbflp(base) __asm__ volatile("dcbf 0,%0,3" ::"r"(base) : "memory")

// Move to special-purpose register, e.g., data stream control register == 3
#define __mtspr(spr, value) \
  __asm__ volatile("mtspr %0,%1" : : "n"(spr), "r"(value))

#endif /* PPC_INTRINSICS_H */
