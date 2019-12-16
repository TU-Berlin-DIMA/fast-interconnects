/*
 * Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019 Clemens Lutz, German Research Center for Artificial
 * Intelligence
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

#include <cstdint>

// Block-wise exclusive prefix sum
template <typename T>
__device__ void block_exclusive_prefix_sum(T *const data, size_t size,
                                           size_t padding, T *const sums) {
  size_t lane_id = threadIdx.x % warpSize;
  // determine a warp_id within a block
  size_t warp_id = threadIdx.x / warpSize;
  size_t thread_items =
      (threadIdx.x < size) ? (size + blockDim.x - 1) / blockDim.x : 0;

  // Below is the basic structure of using a shfl instruction
  // for a scan.
  // Record "value" as a variable - we accumulate it along the way
  T thread_total = 0;
  for (size_t i = 0; i < thread_items; ++i) {
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
  for (size_t i = 1; i <= warpSize; i *= 2) {
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
    size_t warp_sum = sums[lane_id];

    unsigned int mask = (1 << (blockDim.x / warpSize)) - 1;
    for (size_t i = 1; i <= (blockDim.x / warpSize); i *= 2) {
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
  for (size_t i = 0; i < thread_items; ++i) {
    T value = data[threadIdx.x * thread_items + i];
    data[threadIdx.x * thread_items + i] =
        thread_sum + (threadIdx.x + 1) * padding;
    thread_sum += value;
  }
}

// Export `block_exlusive_prefix_sum` to host (for unit testing)
extern "C" __global__ void host_block_exclusive_prefix_sum_uint64(
    uint64_t *const data, size_t size, size_t padding) {
  extern __shared__ uint64_t shared_mem[];
  size_t block_size = size / gridDim.x;
  size_t *block_data = &data[block_size * blockIdx.x];
  block_exclusive_prefix_sum(block_data, block_size, padding, shared_mem);
}

#endif /* PREFIX_SCAN_H */
