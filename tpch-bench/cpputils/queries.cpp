/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2020, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

#include <cstdint>

extern "C" void tpch_q6_branching(uint64_t length, int32_t *l_shipdate,
                                  int32_t *l_discount, int32_t *l_quantity,
                                  int32_t *l_extendedprice, int64_t *revenue) {
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
  *revenue = 0;
  for (uint64_t i = 0; i < length; ++i) {
    if ((l_shipdate[i] >= 366 + 365 + 1) &
        (l_shipdate[i] < 366 + 365 + 365 + 1) & (l_discount[i] >= 5) &
        (l_discount[i] <= 7) & (l_quantity[i] < 24)) {
      *revenue += l_extendedprice[i] * l_discount[i];
    }
  }
}
