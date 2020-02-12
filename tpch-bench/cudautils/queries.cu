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

extern "C" __global__ void tpch_q6_branching(
    uint64_t length, int32_t *l_shipdate, int32_t *l_discount,
    int32_t *l_quantity, int32_t *l_extendedprice, uint64_t *revenue,
    uint64_t *negative_revenue) {
  const uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_threads = blockDim.x * gridDim.x;

  __shared__ unsigned long long block_revenue;
  __shared__ unsigned long long block_negative_revenue;

  if (threadIdx.x == 0) {
    block_revenue = 0;
    block_negative_revenue = 0;
  }
  __syncthreads();

  // Parallel query computation
  long long private_revenue = 0;
  for (uint64_t i = global_idx; i < length; i += global_threads) {
    if (l_shipdate[i] >= 366 + 365 + 1 && l_shipdate[i] < 366 + 365 + 365 + 1 &&
        l_discount[i] >= 5 && l_discount[i] <= 7 && l_quantity[i] < 24) {
      private_revenue += l_extendedprice[i] * l_discount[i];
    }
  }

  // Reduce result, with work-around because CUDA doesn't support atomicAdd for
  // long long int (i.e., signed 64-bit integers)
  if (private_revenue >= 0) {
    atomicAdd(&block_revenue, private_revenue);
  } else {
    atomicAdd(&block_negative_revenue, -private_revenue);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    if (block_revenue > 0) {
      atomicAdd(reinterpret_cast<unsigned long long *>(revenue), block_revenue);
    }
    if (block_negative_revenue > 0) {
      atomicAdd(reinterpret_cast<unsigned long long *>(negative_revenue),
                block_negative_revenue);
    }
  }
}

extern "C" __global__ void tpch_q6_predication(
    uint64_t length, int32_t *l_shipdate, int32_t *l_discount,
    int32_t *l_quantity, int32_t *l_extendedprice, uint64_t *revenue,
    uint64_t *negative_revenue) {
  const uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_threads = blockDim.x * gridDim.x;

  __shared__ unsigned long long block_revenue;
  __shared__ unsigned long long block_negative_revenue;

  if (threadIdx.x == 0) {
    block_revenue = 0;
    block_negative_revenue = 0;
  }
  __syncthreads();

  // Parallel query computation
  long long private_revenue = 0;
  for (uint64_t i = global_idx; i < length; i += global_threads) {
    int condition = (l_shipdate[i] >= 366 + 365 + 1) &
                    (l_shipdate[i] < 366 + 365 + 365 + 1) &
                    (l_discount[i] >= 5) & (l_discount[i] <= 7) &
                    (l_quantity[i] < 24);
    condition = ((!condition) << 31) >> 31;
    private_revenue += condition & (l_extendedprice[i] * l_discount[i]);
  }

  // Reduce result, with work-around because CUDA doesn't support atomicAdd for
  // long long int (i.e., signed 64-bit integers)
  if (private_revenue >= 0) {
    atomicAdd(&block_revenue, private_revenue);
  } else {
    atomicAdd(&block_negative_revenue, -private_revenue);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    if (block_revenue > 0) {
      atomicAdd(reinterpret_cast<unsigned long long *>(revenue), block_revenue);
    }
    if (block_negative_revenue > 0) {
      atomicAdd(reinterpret_cast<unsigned long long *>(negative_revenue),
                block_negative_revenue);
    }
  }
}
