/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2020-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

#include <cstdint>

#if defined(__powerpc64__)
#include <ppc_intrinsics.h>
#endif

// Disable strided prefetch and set maximum prefetch depth
#define PPC_TUNE_DSCR 7ULL

extern "C" void tpch_q6_branching(uint64_t length, int32_t *l_shipdate,
                                  int32_t *l_discount, int32_t *l_quantity,
                                  int32_t *l_extendedprice, int64_t *revenue) {
#if defined(__powerpc64__)
  __mtspr(PPC_DSCR, PPC_TUNE_DSCR);
#endif

  *revenue = 0;
  for (uint64_t i = 0; i < length; ++i) {
    if (l_shipdate[i] >= 366 + 365 + 1 && l_shipdate[i] < 366 + 365 + 365 + 1 &&
        l_discount[i] >= 5 && l_discount[i] <= 7 && l_quantity[i] < 24) {
      *revenue += l_extendedprice[i] * l_discount[i];
    }
  }
}

extern "C" void tpch_q6_predication(uint64_t length, int32_t *l_shipdate,
                                    int32_t *l_discount, int32_t *l_quantity,
                                    int32_t *l_extendedprice,
                                    int64_t *revenue) {
#if defined(__powerpc64__)
  __mtspr(PPC_DSCR, PPC_TUNE_DSCR);
#endif

  *revenue = 0;
  for (uint64_t i = 0; i < length; ++i) {
    int condition = (l_shipdate[i] >= 366 + 365 + 1) &
                    (l_shipdate[i] < 366 + 365 + 365 + 1) &
                    (l_discount[i] >= 5) & (l_discount[i] <= 7) &
                    (l_quantity[i] < 24);
    condition = ((!condition) << 31) >> 31;
    *revenue += condition & (l_extendedprice[i] * l_discount[i]);
  }
}
