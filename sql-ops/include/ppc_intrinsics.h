/*
 * Copyright 2020 Clemens Lutz, German Research Center for Artificial
 * Intelligence
 *
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
