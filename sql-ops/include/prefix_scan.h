/*
 * Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019-2022 Clemens Lutz
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PREFIX_SCAN_H
#define PREFIX_SCAN_H

#include <constants.h>

#include "prefix_scan_state.h"
#include "ptx_memory.h"

#ifndef SIZE_T
#define SIZE_T unsigned int
#endif

#ifndef LOG2_NUM_BANKS
#define LOG2_NUM_BANKS 5
#endif

#include <cstdint>

// Grid synchronization is only supported on Pascal and higher, and will not
// compile on Maxwell or lower.
#if __CUDA_ARCH__ >= 600
#include <cooperative_groups.h>
#endif

// Block-wise exclusive prefix sum
template <typename T>
__device__ void block_exclusive_prefix_sum(T *const data, SIZE_T size,
                                           SIZE_T padding, T *const temp) {
  SIZE_T thread_items =
      (threadIdx.x < size) ? (size + blockDim.x - 1) / blockDim.x : 0;

  // Below is the basic structure of using a shfl instruction
  // for a scan.
  // Record "value" as a variable - we accumulate it along the way
  T thread_total = 0;
  for (SIZE_T i = 0; i < thread_items; ++i) {
    thread_total += data[threadIdx.x * thread_items + i];
  }

  temp[threadIdx.x + (threadIdx.x >> LOG2_NUM_BANKS)] = thread_total;

  // Up sweep
  SIZE_T offset = 1;
#pragma unroll
  for (SIZE_T d = blockDim.x >> 1; d > 0U; d >>= 1) {
    __syncthreads();

    if (threadIdx.x < d) {
      SIZE_T ai = offset * (2U * threadIdx.x + 1U) - 1U;
      SIZE_T bi = offset * (2U * threadIdx.x + 2U) - 1U;
      ai += ai >> LOG2_NUM_BANKS;
      bi += bi >> LOG2_NUM_BANKS;

      temp[bi] += temp[ai];
    }

    offset *= 2;
  }

  if (threadIdx.x == 0) {
    temp[blockDim.x - 1 + ((blockDim.x - 1) >> LOG2_NUM_BANKS)] = 0;
  }

  // Down sweep
#pragma unroll
  for (SIZE_T d = 1U; d < blockDim.x; d *= 2U) {
    offset >>= 1;

    __syncthreads();

    if (threadIdx.x < d) {
      SIZE_T ai = offset * (2U * threadIdx.x + 1U) - 1U;
      SIZE_T bi = offset * (2U * threadIdx.x + 2U) - 1U;
      ai += ai >> LOG2_NUM_BANKS;
      bi += bi >> LOG2_NUM_BANKS;

      T t = temp[ai];

      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

  __syncthreads();

  T thread_sum = temp[threadIdx.x + (threadIdx.x >> LOG2_NUM_BANKS)];
  for (SIZE_T i = 0; i < thread_items; ++i) {
    T value = data[threadIdx.x * thread_items + i];
    data[threadIdx.x * thread_items + i] =
        thread_sum + (threadIdx.x + 1) * padding;
    thread_sum += value;
  }
}

// Block-wise exclusive prefix sum using warp-level shuffle
template <typename T>
__noinline__  // Reduce GPU register usage
    __device__ void
    block_shuffle_exclusive_prefix_sum(T *const data, SIZE_T size,
                                       SIZE_T padding, T *const sums) {
  unsigned int lane_id = threadIdx.x % warpSize;
  // determine a warp_id within a block
  unsigned int warp_id = threadIdx.x / warpSize;
  SIZE_T thread_items =
      (threadIdx.x < size) ? (size + blockDim.x - 1) / blockDim.x : 0;

  // Below is the basic structure of using a shfl instruction
  // for a scan.
  // Record "value" as a variable - we accumulate it along the way
  T thread_total = 0;
  for (SIZE_T i = 0; i < thread_items; ++i) {
    thread_total += data[threadIdx.x * thread_items + i];
  }

  // Now accumulate in log steps up the chain
  // compute sums, with another thread's value who is
  // distance delta away (i).  Note
  // those threads where the thread 'i' away would have
  // been out of bounds of the warp are unaffected.  This
  // creates the scan sum.
  T warp_sum = thread_total;
  unsigned int mask = 0xffffffffU;
#pragma unroll
  for (unsigned int i = 1; i <= warpSize; i *= 2) {
    T n = __shfl_up_sync(mask, warp_sum, i, warpSize);

    if (lane_id >= i) warp_sum += n;
  }

  // value now holds the scan value for the individual thread
  // next sum the largest values for each warp

  // write the sum of the warp to smem
  if (lane_id == warpSize - 1) {
    sums[warp_id] = warp_sum;
  }

  // convert inclusive prefix scan into exclusive prefix scan inside warp
  warp_sum = __shfl_up_sync(mask, warp_sum, 1, warpSize);
  if (lane_id == 0) warp_sum = 0;

  __syncthreads();

  //
  // scan sum the warp sums
  // the same shfl scan operation, but performed on warp sums
  //
  if (warp_id == 0 && lane_id < (blockDim.x / warpSize)) {
    T warp_sum = sums[lane_id];

    unsigned int mask = (1 << (blockDim.x / warpSize)) - 1;
#pragma unroll
    for (unsigned int i = 1; i <= (blockDim.x / warpSize); i *= 2) {
      T n = __shfl_up_sync(mask, warp_sum, i, (blockDim.x / warpSize));

      if (lane_id >= i) warp_sum += n;
    }

    sums[lane_id] = warp_sum;
  }

  __syncthreads();

  // perform a uniform add across warps in the block
  // read neighbouring warp's sum and add it to threads value
  T blockSum = 0;

  if (warp_id > 0) {
    blockSum = sums[warp_id - 1];
  }

  // Now write out our result
  T thread_sum = blockSum + warp_sum;
  for (SIZE_T i = 0; i < thread_items; ++i) {
    T value = data[threadIdx.x * thread_items + i];
    data[threadIdx.x * thread_items + i] =
        thread_sum + (threadIdx.x + 1) * padding;
    thread_sum += value;
  }
}

// Initializes the ScanState of the GPU-wide exclusive prefix sum
template <typename T>
__device__ void device_exclusive_prefix_sum_initialize(
    ScanState<T> *const state) {
  unsigned int global_id = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int global_size = gridDim.x * blockDim.x;
  unsigned int num_warps = global_size / warpSize;
  unsigned int state_len = num_warps + warpSize;

  for (unsigned int i = global_id; i < state_len; i += global_size) {
    state[i].status = (i < warpSize) ? static_cast<T>(SCAN_STATUS_PREFIX_AVAIL)
                                     : static_cast<T>(SCAN_STATUS_INVALID);
    state[i].aggregate = {0};
    state[i].prefix = {0};
    state[i].__padding = {0};
  }
}

// GPU-wide exclusive prefix sum using the Merrill-Garland decoupled lookback
// algorithm
//
// FIXME: Replace "multiple items per thread" strategy with iteration strategy.
//        In each iteration, process one item per thread. Then reduce
//        state_index by one, because last position must retain
//        SCAN_STATUS_PREFIX_AVAIL value. Then reinitialize warp-local state,
//        and continue processing the next batch.  This change wil transform
//        the current two-pass strategy into a single-pass strategy.
template <typename T>
__device__ void device_exclusive_prefix_sum(T *const data, SIZE_T size,
                                            SIZE_T padding,
                                            ScanState<T> *const state) {
  unsigned int lane_id = threadIdx.x % warpSize;
  // determine a warp_id within a block
  unsigned int global_id = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int warp_id = global_id / warpSize;
  unsigned int state_index = warp_id + warpSize;
  SIZE_T thread_items =
      (blockDim.x * blockIdx.x + threadIdx.x < size)
          ? (size + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x)
          : 0;

  T thread_total = 0;
  for (SIZE_T i = 0; i < thread_items; ++i) {
    thread_total += data[global_id * thread_items + i];
  }

  // Now accumulate in log steps up the chain
  // compute sums, with another thread's value who is
  // distance delta away (i).  Note
  // those threads where the thread 'i' away would have
  // been out of bounds of the warp are unaffected.  This
  // creates the scan sum.
  T warp_sum = thread_total;
  unsigned int mask = 0xffffffffU;
#pragma unroll
  for (unsigned int i = 1; i <= warpSize; i *= 2) {
    T n = __shfl_up_sync(mask, warp_sum, i, warpSize);

    if (lane_id >= i) warp_sum += n;
  }

  // value now holds the scan value for the individual thread

  // write the sum of the warp to dmem
  if (lane_id == warpSize - 1) {
    ptx_store_cache_global(&(state[state_index].aggregate), warp_sum);
    __threadfence();
    ptx_store_cache_global(&(state[state_index].status),
                           static_cast<T>(SCAN_STATUS_AGGREGATE_AVAIL));
  }

  // initialized to: [32 x PREFIX_AVAIL = 0]
  // if any has invalid, wait in loop
  // if any has prefix_avail, exit loop
  // else all must have aggergate avail, thus continue to next predecessor
  unsigned int predecessor = state_index - lane_id - 1U;
  T start_value = 0;
  unsigned int prefix_ballot = 0U;
  T status = SCAN_STATUS_INVALID;
  do {
    status = ptx_load_cache_global(&(state[predecessor].status));
    __threadfence();

    if (__any_sync(mask, status == SCAN_STATUS_INVALID)) {
      continue;
    }

    T aggregate = ptx_load_cache_global(&(state[predecessor].aggregate));
    T prefix = ptx_load_cache_global(&(state[predecessor].prefix));

    if ((prefix_ballot =
             __ballot_sync(mask, status == SCAN_STATUS_PREFIX_AVAIL)) > 0) {
      // Only one lane should add the prefix. Note that the order of lanes
      // is reversed due to the predecessor initialization above, i.e., lane_0
      // has predecessor + 1 of lane_1.
      unsigned int first_lane_with_prefix = __ffs(prefix_ballot) - 1;
      if (lane_id < first_lane_with_prefix) {
        start_value += aggregate;
      } else if (lane_id == first_lane_with_prefix) {
        start_value += prefix;
      }

      break;
    } else {
      start_value += aggregate;
      predecessor -= warpSize;
    }
  } while (true);

  // reduce the aggregates to a total sum
  // use XOR mode to perform butterfly reduction
#pragma unroll
  for (unsigned int i = warpSize / 2; i >= 1; i /= 2) {
    start_value += __shfl_xor_sync(mask, start_value, i, warpSize);
  }

  // write the final inclusive prefix of the warp to dmem
  if (lane_id == warpSize - 1) {
    ptx_store_cache_global(&(state[state_index].prefix),
                           warp_sum + start_value);
    __threadfence();
    ptx_store_cache_global(&(state[state_index].status),
                           static_cast<T>(SCAN_STATUS_PREFIX_AVAIL));
  }

  // convert inclusive prefix scan into exclusive prefix scan inside warp
  warp_sum = __shfl_up_sync(mask, warp_sum, 1, warpSize);
  if (lane_id == 0) warp_sum = 0;

  // write out result
  T thread_sum = warp_sum + start_value;
  for (SIZE_T i = 0; i < thread_items; ++i) {
    T value = data[global_id * thread_items + i];
    data[global_id * thread_items + i] = thread_sum + (global_id + 1) * padding;
    thread_sum += value;
  }
}

// Compute shared memory size as number of elements
template <typename T>
__device__ uint32_t block_exclusive_prefix_sum_size() {
  return blockDim.x + (blockDim.x >> LOG2_NUM_BANKS);
}

// Export `block_exlusive_prefix_sum` to host (for unit testing)
extern "C" __global__ void host_block_exclusive_prefix_sum_uint64(
    uint64_t *const data, SIZE_T size, SIZE_T padding) {
  extern __shared__ uint64_t shared_mem_prefix_sum[];
  SIZE_T block_size = size / gridDim.x;
  uint64_t *block_data = &data[block_size * blockIdx.x];
  block_exclusive_prefix_sum(block_data, block_size, padding,
                             shared_mem_prefix_sum);
}

// Export `device_exlusive_prefix_sum_initialize` to host (for unit testing)
extern "C" __global__ void host_device_exclusive_prefix_sum_initialize_uint64(
    ScanState<unsigned long long> *const state) {
  device_exclusive_prefix_sum_initialize(state);

  // Device-wide synchronization to avoid race-condition between the
  // initialization and the prefix sum.
#if __CUDA_ARCH__ >= 600
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  grid.sync();
#endif
}

// Export `device_exlusive_prefix_sum` to host (for unit testing)
extern "C" __global__ void host_device_exclusive_prefix_sum_uint64(
    unsigned long long *const data, SIZE_T size, SIZE_T padding,
    ScanState<unsigned long long> *const state) {
  device_exclusive_prefix_sum(data, size, padding, state);
}

#endif /* PREFIX_SCAN_H */
