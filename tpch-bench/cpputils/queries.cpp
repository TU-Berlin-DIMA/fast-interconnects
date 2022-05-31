// Copyright 2020-2022 Clemens Lutz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
