/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019-2020 Clemens Lutz, German Research Center for Artificial
 * Intelligence
 * Author: Clemens Lutz, DFKI GmbH <clemens.lutz@dfki.de>
 */

#include <prefix_scan.h>
#include <ptx_memory.h>

#ifndef TUPLES_PER_THREAD
#define TUPLES_PER_THREAD 5U
#endif

#define __UINT_MAX__ static_cast<unsigned int>(__INT_MAX__ * 2U + 1U)

#include <cassert>
#include <cstdint>

using namespace std;

// Returns the log2 of the next-lower power of two
__device__ int log2_floor_power_of_two(int x) { return 32 - __clz(x) - 1; }

// Returns the log2 of the next-higher power of two
__device__ int log2_ceil_power_of_two(int x) { return 32 - __clz(x - 1); }

// Arguments to the partitioning function.
//
// Note that the struct's layout must be kept in sync with its counterpart in
// Rust.
struct RadixPartitionArgs {
  // Inputs
  const void *const __restrict__ join_attr_data;
  const void *const __restrict__ payload_attr_data;
  size_t const data_length;
  uint32_t const padding_length;
  uint32_t const radix_bits;

  // State
  uint32_t *const __restrict__ tmp_partition_offsets;
  char *const __restrict__ device_memory_buffers;
  uint64_t const device_memory_buffer_bytes;

  // Outputs
  uint64_t *const __restrict__ partition_offsets;
  void *const __restrict__ partitioned_relation;
};

// A key-value tuple.
//
// Note that the struct's layout must be kept in sync with its counterpart in
// Rust.
template <typename K, typename V>
struct Tuple {
  K key;
  V value;
};

__device__ __forceinline__ uint32_t write_combine_slot(
    uint32_t tuples_per_buffer, uint32_t p_index, uint32_t slot) {
  return tuples_per_buffer * p_index + slot;
}

// Computes the partition ID of a given key.
template <typename T, typename B>
__device__ uint32_t key_to_partition(T key, uint64_t mask, B bits) {
  return static_cast<uint32_t>((static_cast<uint64_t>(key) & mask) >> bits);
}

// Chunked radix partitioning.
//
// See the Rust module for details.
template <typename K, typename V>
__device__ void gpu_chunked_radix_partition(RadixPartitionArgs &args) {
  extern __shared__ uint32_t shared_mem[];

  const uint32_t fanout = 1U << args.radix_bits;
  // const K mask = static_cast<K>(fanout - 1);
  const uint64_t mask = static_cast<uint64_t>(fanout - 1U);

  // Calculate the data_length per block
  size_t data_length = (args.data_length + gridDim.x - 1U) / gridDim.x;
  size_t data_offset = data_length * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = args.data_length - data_offset;
  }

  auto join_attr_data =
      static_cast<const K *>(args.join_attr_data) + data_offset;
  auto payload_attr_data =
      static_cast<const V *>(args.payload_attr_data) + data_offset;
  auto partitioned_relation =
      static_cast<Tuple<K, V> *>(args.partitioned_relation) + data_offset;

  uint32_t *const tmp_partition_offsets = shared_mem;
  uint32_t *const prefix_tmp = &shared_mem[fanout];

  // Ensure counters are all zeroed.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] = 0;
  }

  __syncthreads();

  // 1. Compute local histograms per partition for thread block.
  for (size_t i = threadIdx.x; i < data_length; i += blockDim.x) {
    auto key = join_attr_data[i];
    auto p_index = key_to_partition(key, mask, 0);
    atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
  }

  __syncthreads();

  // 2. Compute offsets with exclusive prefix sum for thread block.
  block_exclusive_prefix_sum(tmp_partition_offsets, fanout, args.padding_length,
                             prefix_tmp);

  __syncthreads();

  // Add data offset onto partitions offsets and write out to global memory.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    uint64_t offset = tmp_partition_offsets[i] + data_offset;
    args.partition_offsets[blockIdx.x * fanout + i] = offset;
  }

  __syncthreads();

  // 3. Partition
  for (size_t i = threadIdx.x; i < data_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    auto p_index = key_to_partition(tuple.key, mask, 0);
    auto offset =
        atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
    partitioned_relation[offset] = tuple;
  }
}

template <typename K, typename V>
__device__ void gpu_chunked_laswwc_radix_partition(RadixPartitionArgs &args) {
  extern __shared__ uint32_t shared_mem[];

  const uint32_t fanout = 1U << args.radix_bits;
  const uint64_t mask = static_cast<uint64_t>(fanout - 1);

  // Calculate the data_length per block
  size_t data_length = (args.data_length + gridDim.x - 1U) / gridDim.x;
  size_t data_offset = data_length * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = args.data_length - data_offset;
  }

  auto join_attr_data =
      static_cast<const K *>(args.join_attr_data) + data_offset;
  auto payload_attr_data =
      static_cast<const V *>(args.payload_attr_data) + data_offset;
  auto partitioned_relation =
      static_cast<Tuple<K, V> *>(args.partitioned_relation) + data_offset;

  auto prefix_tmp_size = block_exclusive_prefix_sum_size<uint32_t>();

  K *const cached_keys = (K *)shared_mem;
  V *const cached_vals = (V *)&cached_keys[blockDim.x * TUPLES_PER_THREAD];
  uint32_t *const tmp_partition_offsets =
      (uint32_t *)&cached_vals[blockDim.x * TUPLES_PER_THREAD];
  uint32_t *const prefix_tmp =
      &tmp_partition_offsets[fanout];  // max(prefix_tmp_size, fanout)

  // Ensure counters are all zeroed.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] = 0;
  }

  __syncthreads();

  // 1. Compute local histograms per partition for thread block.
  for (size_t i = threadIdx.x; i < data_length; i += blockDim.x) {
    auto key = join_attr_data[i];
    auto p_index = key_to_partition(key, mask, 0);
    atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1ULL);
  }

  __syncthreads();

  // 2. Compute offsets with exclusive prefix sum for thread block.
  block_exclusive_prefix_sum(tmp_partition_offsets, fanout, args.padding_length,
                             prefix_tmp);

  __syncthreads();

  // Add data offset onto partitions offsets and write out to global memory.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    uint64_t offset = tmp_partition_offsets[i] + data_offset;
    args.partition_offsets[blockIdx.x * fanout + i] = offset;
  }

  __syncthreads();

  // 3. Partition

  // Reuse space from prefix sum for storing cache offsets
  uint32_t *const cache_offsets = prefix_tmp;
  size_t loop_length = (data_length / (blockDim.x * TUPLES_PER_THREAD)) *
                       (blockDim.x * TUPLES_PER_THREAD);

  for (size_t i = threadIdx.x; i < loop_length;
       i += blockDim.x * TUPLES_PER_THREAD) {
    // Load tuples
    Tuple<K, V> tuple[TUPLES_PER_THREAD];
#pragma unroll
    for (int k = 0; k < TUPLES_PER_THREAD; ++k) {
      tuple[k].key = join_attr_data[i + k * blockDim.x];
      tuple[k].value = payload_attr_data[i + k * blockDim.x];
    }

    // Ensure counters are all zeroed.
    for (uint32_t h = threadIdx.x; h < fanout; h += blockDim.x) {
      cache_offsets[h] = 0;
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < TUPLES_PER_THREAD; ++k) {
      // Hash keys to partition IDs
      auto p_index = key_to_partition(tuple[k].key, mask, 0);

      // Build histogram of cached tuples
      atomicAdd((unsigned int *)&cache_offsets[p_index], 1U);
    }

    // Set linear allocator to zero
    uint32_t *const allocator = (uint32_t *)cached_keys;
    if (threadIdx.x == 0) {
      *allocator = 0;
    }

    __syncthreads();

    // Allocate space per partition for tuple reordering
    for (uint32_t h = threadIdx.x; h < fanout; h += blockDim.x) {
      auto count = cache_offsets[h];
      cache_offsets[h] = atomicAdd((unsigned int *)allocator, count);
    }

    __syncthreads();

    // Allocate space per tuple for tuple reordering and then do reordering
#pragma unroll
    for (int k = 0; k < TUPLES_PER_THREAD; ++k) {
      auto p_index = key_to_partition(tuple[k].key, mask, 0);
      auto pos = atomicAdd((unsigned int *)&cache_offsets[p_index], 1U);
      cached_keys[pos] = tuple[k].key;
      cached_vals[pos] = tuple[k].value;
    }

    __syncthreads();

    // Write tuples to global memory
#pragma unroll
    for (uint32_t k = threadIdx.x; k < blockDim.x * TUPLES_PER_THREAD;
         k += blockDim.x) {
      Tuple<K, V> tuple;
      tuple.key = cached_keys[k];
      tuple.value = cached_vals[k];
      auto p_index = key_to_partition(tuple.key, mask, 0);

      auto offset = cache_offsets[p_index] - (k + 1);
      offset += tmp_partition_offsets[p_index];

      partitioned_relation[offset] = tuple;
    }

    __syncthreads();

    // Update partition offsets for next loop iteration
#pragma unroll
    for (uint32_t k = threadIdx.x; k < blockDim.x * TUPLES_PER_THREAD;
         k += blockDim.x) {
      auto key = cached_keys[k];
      auto p_index = key_to_partition(key, mask, 0);
      atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1);
    }
  }

  // Handle case when data_length % (TUPLES_PER_THREAD * blockDim) != 0
  for (size_t i = loop_length + threadIdx.x; i < data_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    auto p_index = key_to_partition(tuple.key, mask, 0);
    auto offset =
        atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
    partitioned_relation[offset] = tuple;
  }
}

template <typename K, typename V, bool non_temporal>
__device__ void gpu_chunked_sswwc_radix_partition(RadixPartitionArgs &args,
                                                  uint32_t shared_mem_bytes) {
  extern __shared__ uint32_t shared_mem[];

  const uint32_t fanout = 1U << args.radix_bits;
  const uint64_t mask = static_cast<uint64_t>(fanout - 1);
  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;
  constexpr uint32_t warp_mask = 0xFFFFFFFFu;

  // Calculate the data_length per block
  size_t data_length = (args.data_length + gridDim.x - 1U) / gridDim.x;
  size_t data_offset = data_length * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = args.data_length - data_offset;
  }

  auto join_attr_data =
      static_cast<const K *>(args.join_attr_data) + data_offset;
  auto payload_attr_data =
      static_cast<const V *>(args.payload_attr_data) + data_offset;
  auto partitioned_relation =
      static_cast<Tuple<K, V> *>(args.partitioned_relation) + data_offset;

  auto prefix_tmp_size = block_exclusive_prefix_sum_size<uint32_t>();

  const uint32_t sswwc_buffer_bytes =
      shared_mem_bytes - 2U * fanout * sizeof(uint32_t);
  const uint32_t tuples_per_buffer =
      sswwc_buffer_bytes / sizeof(Tuple<K, V>) / fanout;

  uint32_t *const tmp_partition_offsets = (uint32_t *)shared_mem;
  uint32_t *const prefix_tmp = &tmp_partition_offsets[fanout];
  uint32_t *const slots = prefix_tmp;  // alias to reuse space
  Tuple<K, V> *const buffers = reinterpret_cast<Tuple<K, V> *>(&slots[fanout]);

  // Ensure counters are all zeroed.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] = 0;
  }

  __syncthreads();

  // 1. Compute local histograms per partition for thread block.
  for (size_t i = threadIdx.x; i < data_length; i += blockDim.x) {
    auto key = join_attr_data[i];
    auto p_index = key_to_partition(key, mask, 0);
    atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1ULL);
  }

  __syncthreads();

  // 2. Compute offsets with exclusive prefix sum for thread block.
  block_exclusive_prefix_sum(tmp_partition_offsets, fanout, args.padding_length,
                             prefix_tmp);

  __syncthreads();

  // Add data offset onto partitions offsets and write out to global memory.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    uint64_t offset = tmp_partition_offsets[i] + data_offset;
    args.partition_offsets[blockIdx.x * fanout + i] = offset;
  }

  // Zero slots
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    slots[i] = 0;
  }

  __syncthreads();

  // 3. Partition

  // All threads in warp must participate in each loop iteration
  size_t loop_length = (data_length / warpSize) * warpSize;
  for (size_t i = threadIdx.x; i < loop_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    if (non_temporal) {
      tuple.key = ptx_load_cache_streaming(&join_attr_data[i]);
      tuple.value = ptx_load_cache_streaming(&payload_attr_data[i]);
    } else {
      tuple.key = join_attr_data[i];
      tuple.value = payload_attr_data[i];
    }

    uint32_t p_index = key_to_partition(tuple.key, mask, 0);
    uint32_t pos = 0;
    bool done = false;
    do {
      // Fetch a position if don't have a valid position yet
      if (not done) {
        pos = atomicAdd((unsigned int *)&slots[p_index], 1U);

        if (pos < tuples_per_buffer) {
          buffers[write_combine_slot(tuples_per_buffer, p_index, pos)] = tuple;
          done = true;
        }
      }

      // Must flush the buffer
      // Cases:
      //   1. one or more threads in a warp has a full buffer
      //     a) in same partition -> handled by atomicAdd and retry
      //     b) in different partitions -> handled by ballot and loop
      //   2. one or more threads in block but other warp has a full buffer in
      //        same partition -> handled by atomicAdd and retry
      uint32_t ballot = 0;
      int is_candidate = (pos == tuples_per_buffer);
      while (ballot = __ballot_sync(warp_mask, is_candidate)) {
        int leader_id = __ffs(ballot) - 1;
        uint32_t current_index = __shfl_sync(warp_mask, p_index, leader_id);
        uint32_t dst = tmp_partition_offsets[current_index];

        // Memcpy from cached buffer to memory
        for (uint32_t i = lane_id; i < tuples_per_buffer; i += warpSize) {
          if (non_temporal) {
            ptx_store_cache_streaming(
                &partitioned_relation[dst + i].key,
                buffers[write_combine_slot(tuples_per_buffer, current_index, i)]
                    .key);
            ptx_store_cache_streaming(
                &partitioned_relation[dst + i].value,
                buffers[write_combine_slot(tuples_per_buffer, current_index, i)]
                    .value);
          } else {
            partitioned_relation[dst + i] = buffers[write_combine_slot(
                tuples_per_buffer, current_index, i)];
          }
        }

        if (lane_id == leader_id) {
          // Update offsets; this update must be seen by all threads in our
          // block before setting slot index to zero
          tmp_partition_offsets[current_index] += tuples_per_buffer;
          __threadfence_block();

          // Normal write is not visible to other threads
          //   Use atomic function instead of:
          //   slots[current_index] = 0;
          atomicExch(&slots[current_index], 0);

          // Not a leader candidate anymore, because partition is flushed
          is_candidate = 0;
        }
      }
    } while (__any_sync(warp_mask, not done));
  }

  // Wait until all warps are done
  __syncthreads();

  // Flush buffers
  for (uint32_t i = threadIdx.x; i < fanout * tuples_per_buffer;
       i += blockDim.x) {
    uint32_t p_index = i / tuples_per_buffer;
    uint32_t slot = i % tuples_per_buffer;

    if (slot < slots[p_index]) {
      uint32_t dst =
          atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
      partitioned_relation[dst] = buffers[i];
    }
  }

  // __syncthreads() not necessary due to atomicAdd() space reservation in flush

  // Handle case when data_length % warpSize != 0
  for (size_t i = loop_length + threadIdx.x; i < data_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    auto p_index = key_to_partition(tuple.key, mask, 0);
    auto offset =
        atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
    partitioned_relation[offset] = tuple;
  }
}

/// SSWWC v2 tries to avoid blocking warps by acquiring only one buffer lock at
/// a time. If a warp acquires more than one lock, all locks but one are
/// released.
template <typename K, typename V>
__device__ void gpu_chunked_sswwc_radix_partition_v2(
    RadixPartitionArgs &args, uint32_t shared_mem_bytes) {
  extern __shared__ uint32_t shared_mem[];

  const uint32_t fanout = 1U << args.radix_bits;
  const uint64_t mask = static_cast<uint64_t>(fanout - 1);
  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;
  constexpr uint32_t warp_mask = 0xFFFFFFFFu;

  // Calculate the data_length per block
  size_t data_length = (args.data_length + gridDim.x - 1U) / gridDim.x;
  size_t data_offset = data_length * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = args.data_length - data_offset;
  }

  auto join_attr_data =
      static_cast<const K *>(args.join_attr_data) + data_offset;
  auto payload_attr_data =
      static_cast<const V *>(args.payload_attr_data) + data_offset;
  auto partitioned_relation =
      static_cast<Tuple<K, V> *>(args.partitioned_relation) + data_offset;

  auto prefix_tmp_size = block_exclusive_prefix_sum_size<uint32_t>();

  const uint32_t sswwc_buffer_bytes =
      shared_mem_bytes - 2U * fanout * sizeof(uint32_t);
  const uint32_t tuples_per_buffer =
      sswwc_buffer_bytes / sizeof(Tuple<K, V>) / fanout;

  uint32_t *const tmp_partition_offsets = (uint32_t *)shared_mem;
  uint32_t *const prefix_tmp = &tmp_partition_offsets[fanout];
  uint32_t *const slots = prefix_tmp;  // alias to reuse space
  Tuple<K, V> *const buffers = reinterpret_cast<Tuple<K, V> *>(&slots[fanout]);

  // Ensure counters are all zeroed.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] = 0;
  }

  __syncthreads();

  // 1. Compute local histograms per partition for thread block.
  for (size_t i = threadIdx.x; i < data_length; i += blockDim.x) {
    auto key = join_attr_data[i];
    auto p_index = key_to_partition(key, mask, 0);
    atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1ULL);
  }

  __syncthreads();

  // 2. Compute offsets with exclusive prefix sum for thread block.
  block_exclusive_prefix_sum(tmp_partition_offsets, fanout, args.padding_length,
                             prefix_tmp);

  __syncthreads();

  // Add data offset onto partitions offsets and write out to global memory.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    uint64_t offset = tmp_partition_offsets[i] + data_offset;
    args.partition_offsets[blockIdx.x * fanout + i] = offset;
  }

  // Zero slots
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    slots[i] = 0;
  }

  __syncthreads();

  // 3. Partition

  // All threads in warp must participate in each loop iteration
  size_t loop_length = (data_length / warpSize) * warpSize;
  for (size_t i = threadIdx.x; i < loop_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    uint32_t p_index = key_to_partition(tuple.key, mask, 0);
    uint32_t pos = 0;
    bool done = false;
    do {
      // Fetch a position if don't have a valid position yet
      if (not done) {
        pos = atomicAdd((unsigned int *)&slots[p_index], 1U);

        if (pos < tuples_per_buffer) {
          buffers[write_combine_slot(tuples_per_buffer, p_index, pos)] = tuple;
          done = true;
        }
      }

      // Must flush the buffer
      // Cases:
      //   1. one or more threads in a warp has a full buffer
      //     a) in same partition -> handled by atomicAdd and retry
      //     b) in different partitions -> handled by ballot and loop
      //   2. one or more threads in block but other warp has a full buffer in
      //        same partition -> handled by atomicAdd and retry
      uint32_t ballot = 0;
      int is_candidate = (pos == tuples_per_buffer);
      if (ballot = __ballot_sync(warp_mask, is_candidate)) {
        int leader_id = __ffs(ballot) - 1;

        // Release the lock if not the leader and try again in next round.
        if (is_candidate && leader_id != lane_id) {
          atomicExch(&slots[p_index], tuples_per_buffer);
        }

        uint32_t current_index = __shfl_sync(warp_mask, p_index, leader_id);
        uint32_t dst = tmp_partition_offsets[current_index];

        // Memcpy from cached buffer to memory
        for (uint32_t i = lane_id; i < tuples_per_buffer; i += warpSize) {
          partitioned_relation[dst + i] =
              buffers[write_combine_slot(tuples_per_buffer, current_index, i)];
        }

        if (lane_id == leader_id) {
          // Update offsets; this update must be seen by all threads in our
          // block before setting slot index to zero
          tmp_partition_offsets[current_index] += tuples_per_buffer;
          __threadfence_block();

          // Normal write is not visible to other threads
          //   Use atomic function instead of:
          //   slots[current_index] = 0;
          atomicExch(&slots[current_index], 0);
        }
      }
    } while (__any_sync(warp_mask, not done));
  }

  // Wait until all warps are done
  __syncthreads();

  // Flush buffers
  for (uint32_t i = threadIdx.x; i < fanout * tuples_per_buffer;
       i += blockDim.x) {
    uint32_t p_index = i / tuples_per_buffer;
    uint32_t slot = i % tuples_per_buffer;

    if (slot < slots[p_index]) {
      uint32_t dst =
          atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
      partitioned_relation[dst] = buffers[i];
    }
  }

  // __syncthreads() not necessary due to atomicAdd() space reservation in flush

  // Handle case when data_length % warpSize != 0
  for (size_t i = loop_length + threadIdx.x; i < data_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    auto p_index = key_to_partition(tuple.key, mask, 0);
    auto offset =
        atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
    partitioned_relation[offset] = tuple;
  }
}

template <typename K, typename V>
__device__ void gpu_chunked_hsswwc_radix_partition(RadixPartitionArgs &args,
                                                   uint32_t shared_mem_bytes) {
  extern __shared__ uint32_t shared_mem[];

  const uint32_t fanout = 1U << args.radix_bits;
  const uint64_t mask = static_cast<uint64_t>(fanout - 1);
  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;
  constexpr uint32_t warp_mask = 0xFFFFFFFFu;

  // Calculate the data_length per block
  size_t data_length = (args.data_length + gridDim.x - 1U) / gridDim.x;
  size_t data_offset = data_length * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = args.data_length - data_offset;
  }

  auto join_attr_data =
      static_cast<const K *>(args.join_attr_data) + data_offset;
  auto payload_attr_data =
      static_cast<const V *>(args.payload_attr_data) + data_offset;
  auto partitioned_relation =
      static_cast<Tuple<K, V> *>(args.partitioned_relation) + data_offset;
  auto dmem_buffers = reinterpret_cast<Tuple<K, V> *>(
      args.device_memory_buffers +
      args.device_memory_buffer_bytes * blockIdx.x);

  auto prefix_tmp_size = block_exclusive_prefix_sum_size<uint32_t>();

  const uint32_t sswwc_buffer_bytes =
      shared_mem_bytes - fanout * sizeof(uint32_t);
  const uint32_t tuples_per_buffer =
      1U << log2_floor_power_of_two(sswwc_buffer_bytes / sizeof(Tuple<K, V>) /
                                    fanout);
  const uint32_t tuples_per_dmem_buffer =
      1U << log2_floor_power_of_two(args.device_memory_buffer_bytes /
                                    sizeof(Tuple<K, V>) / fanout);
  const uint32_t slots_per_dmem_buffer =
      tuples_per_dmem_buffer / tuples_per_buffer;

  uint32_t *const tmp_partition_offsets = (uint32_t *)shared_mem;
  uint32_t *const prefix_tmp = &tmp_partition_offsets[fanout];
  unsigned long long *const slots = reinterpret_cast<unsigned long long *>(
      prefix_tmp);  // alias to reuse space
  Tuple<K, V> *const buffers = reinterpret_cast<Tuple<K, V> *>(&slots[fanout]);

  // Ensure counters are all zeroed.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] = 0;
  }

  __syncthreads();

  // 1. Compute local histograms per partition for thread block.
  for (size_t i = threadIdx.x; i < data_length; i += blockDim.x) {
    auto key = join_attr_data[i];
    auto p_index = key_to_partition(key, mask, 0);
    atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1ULL);
  }

  __syncthreads();

  // 2. Compute offsets with exclusive prefix sum for thread block.
  block_exclusive_prefix_sum(tmp_partition_offsets, fanout, args.padding_length,
                             prefix_tmp);

  __syncthreads();

  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    // Add padding to device memory slots.
    tmp_partition_offsets[i] += (i + 1) * args.padding_length;

    // Add data offset onto partitions offsets
    // and writing them out to global memory.
    uint64_t offset = tmp_partition_offsets[i] + data_offset;
    args.partition_offsets[blockIdx.x * fanout + i] = offset;
  }

  // Zero shared memory slots.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    slots[i] = 0;
  }

  __syncthreads();

  // 3. Partition

  // Slot bits must occupy upper bits, because we need as much number range as
  // possible for the ticket lock. The lock will eventually overflow, but we
  // can defer it with a larger number range.
  const uint32_t smem_slot_bits =
      64 - log2_ceil_power_of_two(slots_per_dmem_buffer);
  const uint64_t smem_slot_mask = (1ULL << smem_slot_bits) - 1ULL;

  // All threads in warp must participate in each loop iteration
  size_t loop_length = (data_length / warpSize) * warpSize;
  for (size_t i = threadIdx.x; i < loop_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    uint32_t p_index = key_to_partition(tuple.key, mask, 0);
    uint32_t pos = 0;
    bool done = false;
    do {
      // Fetch a position if don't have a valid position yet
      unsigned long long slot = 0;
      if (not done) {
        slot = atomicAdd(&slots[p_index], 1ULL);
        /* slot = atomicInc((unsigned int *)&slots[p_index], fill-in-the-max);
         */

        pos = static_cast<unsigned int>(slot & smem_slot_mask);

        if (pos < tuples_per_buffer) {
          buffers[write_combine_slot(tuples_per_buffer, p_index, pos)] = tuple;
          done = true;
        }
      }

      // Must flush the buffer
      // Cases:
      //   1. one or more threads in a warp has a full buffer
      //     a) in same partition -> handled by atomicAdd and retry
      //     b) in different partitions -> handled by ballot and loop
      //   2. one or more threads in block but other warp has a full buffer in
      //        same partition -> handled by atomicAdd and retry
      uint32_t ballot = 0;
      int is_candidate = (pos == tuples_per_buffer);
      while (ballot = __ballot_sync(warp_mask, is_candidate)) {
        int leader_id = __ffs(ballot) - 1;
        uint32_t dmem_slot = static_cast<unsigned int>(slot >> smem_slot_bits);
        dmem_slot = __shfl_sync(warp_mask, dmem_slot, leader_id);
        uint32_t current_index = __shfl_sync(warp_mask, p_index, leader_id);

        // Flush smem buffers to dmem buffers.
        for (uint32_t i = lane_id; i < tuples_per_buffer; i += warpSize) {
          dmem_buffers[write_combine_slot(
              tuples_per_dmem_buffer, current_index,
              write_combine_slot(tuples_per_buffer, dmem_slot, i))] =
              buffers[write_combine_slot(tuples_per_buffer, current_index, i)];
        }

        // Flush dmem buffer to memory.
        dmem_slot += 1U;
        if (dmem_slot == slots_per_dmem_buffer) {
          dmem_slot = 0;
          uint32_t dst = tmp_partition_offsets[current_index];

          // Wait for warp to finish writing to smem buffer
          __syncwarp();

          for (uint32_t i = lane_id; i < tuples_per_dmem_buffer;
               i += warpSize) {
            partitioned_relation[dst + i] = dmem_buffers[write_combine_slot(
                tuples_per_dmem_buffer, current_index, i)];
          }

          if (lane_id == leader_id) {
            // Update offsets. This update must be seen by all threads in our
            // block before setting slot index to zero.
            tmp_partition_offsets[current_index] += tuples_per_dmem_buffer;

            // Flush the new offset before releasing the partition lock.
            __threadfence_block();
          }
        }

        if (lane_id == leader_id) {
          // Add set smem_slot to zero.
          slot = (static_cast<unsigned long long>(dmem_slot) << smem_slot_bits);

          // Normal write is not visible to other threads
          //   Use atomic function instead of:
          //   slots[p_indexj] = slot;
          atomicExch(&slots[current_index], slot);

          // Not a leader candidate anymore, because partition is flushed
          is_candidate = 0;
        }
      }
    } while (__any_sync(warp_mask, not done));
  }

  // Wait until all warps are done
  __syncthreads();

  // Flush dmem buffers to memory.
  uint32_t log2_tuples_per_buffer = log2_floor_power_of_two(tuples_per_buffer);
  uint32_t log2_tuples_per_dmem_buffer =
      log2_floor_power_of_two(tuples_per_dmem_buffer);
  for (uint32_t i = threadIdx.x; i < (fanout << log2_tuples_per_dmem_buffer);
       i += blockDim.x) {
    uint32_t p_index = i >> log2_tuples_per_dmem_buffer;
    uint32_t slot = i & (tuples_per_dmem_buffer - 1U);

    if (slot < (static_cast<unsigned int>(slots[p_index] >> smem_slot_bits)
                << log2_tuples_per_buffer)) {
      uint32_t dst =
          atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
      partitioned_relation[dst] = dmem_buffers[i];
    }
  }

  // Flush smem buffers directly to memory, because we don't keep track of dmem
  // buffer fill state on tuple-wise granularity.
  for (uint32_t i = threadIdx.x; i < (fanout << log2_tuples_per_buffer);
       i += blockDim.x) {
    uint32_t p_index = i >> log2_tuples_per_buffer;
    uint32_t slot = i & (tuples_per_buffer - 1U);

    if (slot < (slots[p_index] & smem_slot_mask)) {
      uint32_t dst =
          atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
      partitioned_relation[dst] = buffers[i];
    }
  }

  // Handle case when data_length % warpSize != 0
  for (size_t i = loop_length + threadIdx.x; i < data_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    auto p_index = key_to_partition(tuple.key, mask, 0);
    auto offset =
        atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
    partitioned_relation[offset] = tuple;
  }
}

/// HSSWWC v2 tries to avoid blocking warps by acquiring only one buffer lock at
/// a time. If a warp acquires more than one lock, all locks but one are
/// released.
template <typename K, typename V>
__device__ void gpu_chunked_hsswwc_radix_partition_v2(
    RadixPartitionArgs &args, uint32_t shared_mem_bytes) {
  extern __shared__ uint32_t shared_mem[];

  const uint32_t fanout = 1U << args.radix_bits;
  const uint64_t mask = static_cast<uint64_t>(fanout - 1);
  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;
  constexpr uint32_t warp_mask = 0xFFFFFFFFu;

  // Calculate the data_length per block
  size_t data_length = (args.data_length + gridDim.x - 1U) / gridDim.x;
  size_t data_offset = data_length * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = args.data_length - data_offset;
  }

  auto join_attr_data =
      static_cast<const K *>(args.join_attr_data) + data_offset;
  auto payload_attr_data =
      static_cast<const V *>(args.payload_attr_data) + data_offset;
  auto partitioned_relation =
      static_cast<Tuple<K, V> *>(args.partitioned_relation) + data_offset;
  auto dmem_buffers = reinterpret_cast<Tuple<K, V> *>(
      args.device_memory_buffers +
      args.device_memory_buffer_bytes * blockIdx.x);

  auto prefix_tmp_size = block_exclusive_prefix_sum_size<uint32_t>();

  const uint32_t sswwc_buffer_bytes =
      shared_mem_bytes - fanout * sizeof(uint32_t);
  const uint32_t tuples_per_buffer =
      1U << log2_floor_power_of_two(sswwc_buffer_bytes / sizeof(Tuple<K, V>) /
                                    fanout);
  const uint32_t tuples_per_dmem_buffer =
      1U << log2_floor_power_of_two(args.device_memory_buffer_bytes /
                                    sizeof(Tuple<K, V>) / fanout);
  const uint32_t slots_per_dmem_buffer =
      tuples_per_dmem_buffer / tuples_per_buffer;

  uint32_t *const tmp_partition_offsets = (uint32_t *)shared_mem;
  uint32_t *const prefix_tmp = &tmp_partition_offsets[fanout];
  unsigned long long *const slots = reinterpret_cast<unsigned long long *>(
      prefix_tmp);  // alias to reuse space
  Tuple<K, V> *const buffers = reinterpret_cast<Tuple<K, V> *>(&slots[fanout]);

  // Ensure counters are all zeroed.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] = 0;
  }

  __syncthreads();

  // 1. Compute local histograms per partition for thread block.
  for (size_t i = threadIdx.x; i < data_length; i += blockDim.x) {
    auto key = join_attr_data[i];
    auto p_index = key_to_partition(key, mask, 0);
    atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1ULL);
  }

  __syncthreads();

  // 2. Compute offsets with exclusive prefix sum for thread block.
  block_exclusive_prefix_sum(tmp_partition_offsets, fanout, args.padding_length,
                             prefix_tmp);

  __syncthreads();

  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    // Add padding to device memory slots.
    tmp_partition_offsets[i] += (i + 1) * args.padding_length;

    // Add data offset onto partitions offsets
    // and writing them out to global memory.
    uint64_t offset = tmp_partition_offsets[i] + data_offset;
    args.partition_offsets[blockIdx.x * fanout + i] = offset;
  }

  // Zero shared memory slots.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    slots[i] = 0;
  }

  __syncthreads();

  // 3. Partition

  // Slot bits must occupy upper bits, because we need as much number range as
  // possible for the ticket lock. The lock will eventually overflow, but we
  // can defer it with a larger number range.
  const uint32_t smem_slot_bits =
      64 - log2_ceil_power_of_two(slots_per_dmem_buffer);
  const uint64_t smem_slot_mask = (1ULL << smem_slot_bits) - 1ULL;

  // All threads in warp must participate in each loop iteration
  size_t loop_length = (data_length / warpSize) * warpSize;
  for (size_t i = threadIdx.x; i < loop_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    uint32_t p_index = key_to_partition(tuple.key, mask, 0);
    uint32_t pos = 0;
    bool done = false;
    do {
      // Fetch a position if don't have a valid position yet
      unsigned long long slot = 0;
      if (not done) {
        slot = atomicAdd(&slots[p_index], 1ULL);
        /* slot = atomicInc((unsigned int *)&slots[p_index], fill-in-the-max);
         */

        pos = static_cast<unsigned int>(slot & smem_slot_mask);

        if (pos < tuples_per_buffer) {
          buffers[write_combine_slot(tuples_per_buffer, p_index, pos)] = tuple;
          done = true;
        }
      }

      // Must flush the buffer
      // Cases:
      //   1. one or more threads in a warp has a full buffer
      //     a) in same partition -> handled by atomicAdd and retry
      //     b) in different partitions -> handled by ballot and loop
      //   2. one or more threads in block but other warp has a full buffer in
      //        same partition -> handled by atomicAdd and retry
      uint32_t ballot = 0;
      int is_candidate = (pos == tuples_per_buffer);
      if (ballot = __ballot_sync(warp_mask, is_candidate)) {
        int leader_id = __ffs(ballot) - 1;

        // Release the lock if not the leader and try again in next round.
        // Releasing happens by resetting the smem slot to tuples_per_buffer,
        // but we also have to set the dmem_slot field in the upper bits
        // appropriately. The short-cut is to reuse our slot, because it already
        // contains the corrent smem and dmem values.
        if (is_candidate && lane_id != leader_id) {
          atomicExch(&slots[p_index], slot);
        }

        uint32_t dmem_slot = static_cast<unsigned int>(slot >> smem_slot_bits);
        dmem_slot = __shfl_sync(warp_mask, dmem_slot, leader_id);
        uint32_t current_index = __shfl_sync(warp_mask, p_index, leader_id);

        // Flush smem buffers to dmem buffers.
        for (uint32_t i = lane_id; i < tuples_per_buffer; i += warpSize) {
          dmem_buffers[write_combine_slot(
              tuples_per_dmem_buffer, current_index,
              write_combine_slot(tuples_per_buffer, dmem_slot, i))] =
              buffers[write_combine_slot(tuples_per_buffer, current_index, i)];
        }

        // Flush dmem buffer to memory.
        dmem_slot += 1U;
        if (dmem_slot == slots_per_dmem_buffer) {
          dmem_slot = 0;
          uint32_t dst = tmp_partition_offsets[current_index];

          // Wait for warp to finish writing to smem buffer
          __syncwarp();

          for (uint32_t i = lane_id; i < tuples_per_dmem_buffer;
               i += warpSize) {
            partitioned_relation[dst + i] = dmem_buffers[write_combine_slot(
                tuples_per_dmem_buffer, current_index, i)];
          }

          if (lane_id == leader_id) {
            // Update offsets. This update must be seen by all threads in our
            // block before setting slot index to zero.
            tmp_partition_offsets[current_index] += tuples_per_dmem_buffer;

            // Flush the new offset before releasing the partition lock.
            __threadfence_block();
          }
        }

        if (lane_id == leader_id) {
          // Add set smem_slot to zero.
          slot = (static_cast<unsigned long long>(dmem_slot) << smem_slot_bits);

          // Normal write is not visible to other threads
          //   Use atomic function instead of:
          //   slots[p_indexj] = slot;
          atomicExch(&slots[current_index], slot);
        }
      }
    } while (__any_sync(warp_mask, not done));
  }

  // Wait until all warps are done
  __syncthreads();

  // Flush dmem buffers to memory.
  uint32_t log2_tuples_per_buffer = log2_floor_power_of_two(tuples_per_buffer);
  uint32_t log2_tuples_per_dmem_buffer =
      log2_floor_power_of_two(tuples_per_dmem_buffer);
  for (uint32_t i = threadIdx.x; i < (fanout << log2_tuples_per_dmem_buffer);
       i += blockDim.x) {
    uint32_t p_index = i >> log2_tuples_per_dmem_buffer;
    uint32_t slot = i & (tuples_per_dmem_buffer - 1U);

    if (slot < (static_cast<unsigned int>(slots[p_index] >> smem_slot_bits)
                << log2_tuples_per_buffer)) {
      uint32_t dst =
          atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
      partitioned_relation[dst] = dmem_buffers[i];
    }
  }

  // Flush smem buffers directly to memory, because we don't keep track of dmem
  // buffer fill state on tuple-wise granularity.
  for (uint32_t i = threadIdx.x; i < (fanout << log2_tuples_per_buffer);
       i += blockDim.x) {
    uint32_t p_index = i >> log2_tuples_per_buffer;
    uint32_t slot = i & (tuples_per_buffer - 1U);

    if (slot < (slots[p_index] & smem_slot_mask)) {
      uint32_t dst =
          atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
      partitioned_relation[dst] = buffers[i];
    }
  }

  // Handle case when data_length % warpSize != 0
  for (size_t i = loop_length + threadIdx.x; i < data_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    auto p_index = key_to_partition(tuple.key, mask, 0);
    auto offset =
        atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
    partitioned_relation[offset] = tuple;
  }
}

// HSSWWC v3 performs the buffer flush from dmem to memory asynchronously.
// This change enables other warps to make progress during the dmem flush, which
// is important because the dmem buffer is large (several MBs) and the flush can
// take a long time.
template <typename K, typename V>
__device__ void gpu_chunked_hsswwc_radix_partition_v3(
    RadixPartitionArgs &args, uint32_t shared_mem_bytes) {
  extern __shared__ uint32_t shared_mem[];

  const uint32_t fanout = 1U << args.radix_bits;
  const uint64_t mask = static_cast<uint64_t>(fanout - 1);
  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;
  constexpr uint32_t warp_mask = 0xFFFFFFFFu;

  // Calculate the data_length per block
  size_t data_length = (args.data_length + gridDim.x - 1U) / gridDim.x;
  size_t data_offset = data_length * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = args.data_length - data_offset;
  }

  auto join_attr_data =
      static_cast<const K *>(args.join_attr_data) + data_offset;
  auto payload_attr_data =
      static_cast<const V *>(args.payload_attr_data) + data_offset;
  auto partitioned_relation =
      static_cast<Tuple<K, V> *>(args.partitioned_relation) + data_offset;
  auto dmem_buffers = reinterpret_cast<Tuple<K, V> *>(
      args.device_memory_buffers +
      args.device_memory_buffer_bytes * blockIdx.x);

  auto prefix_tmp_size = block_exclusive_prefix_sum_size<uint32_t>();

  const uint32_t sswwc_buffer_bytes =
      shared_mem_bytes - fanout * sizeof(uint32_t);
  const uint32_t tuples_per_buffer =
      1U << log2_floor_power_of_two(sswwc_buffer_bytes / sizeof(Tuple<K, V>) /
                                    fanout);
  const uint32_t tuples_per_dmem_buffer =
      1U << log2_floor_power_of_two(args.device_memory_buffer_bytes /
                                    sizeof(Tuple<K, V>) / fanout);
  const uint32_t slots_per_dmem_buffer =
      tuples_per_dmem_buffer / tuples_per_buffer;

  uint32_t *const tmp_partition_offsets = (uint32_t *)shared_mem;
  uint32_t *const prefix_tmp = &tmp_partition_offsets[fanout];
  unsigned long long *const slots = reinterpret_cast<unsigned long long *>(
      prefix_tmp);  // alias to reuse space
  unsigned int *const dmem_locks =
      reinterpret_cast<unsigned int *>(&slots[fanout]);
  Tuple<K, V> *const buffers =
      reinterpret_cast<Tuple<K, V> *>(&dmem_locks[fanout]);

  // Ensure counters are all zeroed.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] = 0;
  }

  __syncthreads();

  // 1. Compute local histograms per partition for thread block.
  for (size_t i = threadIdx.x; i < data_length; i += blockDim.x) {
    auto key = join_attr_data[i];
    auto p_index = key_to_partition(key, mask, 0);
    atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1ULL);
  }

  __syncthreads();

  // 2. Compute offsets with exclusive prefix sum for thread block.
  block_exclusive_prefix_sum(tmp_partition_offsets, fanout, args.padding_length,
                             prefix_tmp);

  __syncthreads();

  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    // Add padding to device memory slots.
    tmp_partition_offsets[i] += (i + 1) * args.padding_length;

    // Add data offset onto partitions offsets
    // and writing them out to global memory.
    uint64_t offset = tmp_partition_offsets[i] + data_offset;
    args.partition_offsets[blockIdx.x * fanout + i] = offset;
  }

  // Zero shared memory slots and initialize dmem locks.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    slots[i] = 0;
    dmem_locks[i] = 0;
  }

  __syncthreads();

  // 3. Partition

  // Slot bits must occupy upper bits, because we need as much number range as
  // possible for the ticket lock. The lock will eventually overflow, but we
  // can defer it with a larger number range.
  const uint32_t smem_slot_bits =
      64 - log2_ceil_power_of_two(slots_per_dmem_buffer);
  const uint64_t smem_slot_mask = (1ULL << smem_slot_bits) - 1ULL;

  // All threads in warp must participate in each loop iteration
  size_t loop_length = (data_length / warpSize) * warpSize;
  for (size_t i = threadIdx.x; i < loop_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    uint32_t p_index = key_to_partition(tuple.key, mask, 0);
    uint32_t pos = 0;
    bool done = false;
    do {
      // Fetch a position if don't have a valid position yet
      unsigned long long slot = 0;
      if (not done) {
        slot = atomicAdd(&slots[p_index], 1ULL);
        /* slot = atomicInc((unsigned int *)&slots[p_index], fill-in-the-max);
         */

        pos = static_cast<unsigned int>(slot & smem_slot_mask);

        if (pos < tuples_per_buffer) {
          buffers[write_combine_slot(tuples_per_buffer, p_index, pos)] = tuple;
          done = true;
        }
      }

      // Must flush the buffer
      // Cases:
      //   1. one or more threads in a warp has a full buffer
      //     a) in same partition -> handled by atomicAdd and retry
      //     b) in different partitions -> handled by ballot and loop
      //   2. one or more threads in block but other warp has a full buffer in
      //        same partition -> handled by atomicAdd and retry
      uint32_t ballot = 0;
      int is_candidate = (pos == tuples_per_buffer);
      if (ballot = __ballot_sync(warp_mask, is_candidate)) {
        int leader_id = __ffs(ballot) - 1;

        // Release the lock if not the leader and try again in next round.
        // Releasing happens by resetting the smem slot to tuples_per_buffer,
        // but we also have to set the dmem_slot field in the upper bits
        // appropriately. The short-cut is to reuse our slot, because it already
        // contains the corrent smem and dmem values.
        if (is_candidate && lane_id != leader_id) {
          atomicExch(&slots[p_index], slot);
        }

        uint32_t dmem_slot = static_cast<unsigned int>(slot >> smem_slot_bits);
        dmem_slot = __shfl_sync(warp_mask, dmem_slot, leader_id);
        uint32_t current_index = __shfl_sync(warp_mask, p_index, leader_id);

        // Acquire lock on dmem buffer before flushing smem. This performs
        // lock-coupling, because we are still holding the smem lock, which has
        // the advantage that we avoid other threads from acquiring dmem lock.
        //
        // The lock is necessary because the dmem flush is asynchronous to the
        // smem flush. Thus, while we are flushing the dmem buffer to memory,
        // another warp could fill up the smem buffer and try to flush to dmem.
        if (leader_id == lane_id) {
          while (atomicCAS(&dmem_locks[current_index], 0U, 1U) != 0U)
            ;
        }
        // Warp must wait until leader acquires the lock.
        __syncwarp();

        // Flush smem buffers to dmem buffers.
        for (uint32_t i = lane_id; i < tuples_per_buffer; i += warpSize) {
          dmem_buffers[write_combine_slot(
              tuples_per_dmem_buffer, current_index,
              write_combine_slot(tuples_per_buffer, dmem_slot, i))] =
              buffers[write_combine_slot(tuples_per_buffer, current_index, i)];
        }

        // Update the dmem slot.
        dmem_slot += 1U;

        // Wait until warp is finished flushing the smem buffer, then release
        // the smem lock. Threadfence ensures that flushed data are observed as
        // occuring before the smem lock is released.
        __threadfence_block();
        __syncwarp();
        if (lane_id == leader_id) {
          // Add set smem_slot to zero.
          slot = (static_cast<unsigned long long>(dmem_slot) << smem_slot_bits);

          // Normal write is not visible to other threads
          //   Use atomic function instead of:
          //   slots[p_indexj] = slot;
          atomicExch(&slots[current_index], slot);
        }

        // Flush dmem buffer to memory if necessary, otherwise skip to releasing
        // the dmem lock.
        if (dmem_slot == slots_per_dmem_buffer) {
          dmem_slot = 0;
          uint32_t dst = tmp_partition_offsets[current_index];

          for (uint32_t i = lane_id; i < tuples_per_dmem_buffer;
               i += warpSize) {
            partitioned_relation[dst + i] = dmem_buffers[write_combine_slot(
                tuples_per_dmem_buffer, current_index, i)];
          }

          // Ensure that flushed data are observed as occuring before the dmem
          // lock is released.
          __threadfence_block();

          // Ensure that the warp has finished flushing before proceeding to
          // updating tmp_partition_offsets and unlocking the dmem lock.
          __syncwarp();

          if (lane_id == leader_id) {
            // Update offsets. This update must be seen by all threads in our
            // block before setting slot index to zero.
            tmp_partition_offsets[current_index] += tuples_per_dmem_buffer;
            __threadfence_block();
          }
        }

        // Release the dmem lock.
        if (leader_id == lane_id) {
          // Note: AtomicExch used because non-atomics not visible to other
          // threads.
          atomicExch(&dmem_locks[current_index], 0U);
        }
      }
    } while (__any_sync(warp_mask, not done));
  }

  // Wait until all warps are done
  __syncthreads();

  // Flush dmem buffers to memory.
  uint32_t log2_tuples_per_buffer = log2_floor_power_of_two(tuples_per_buffer);
  uint32_t log2_tuples_per_dmem_buffer =
      log2_floor_power_of_two(tuples_per_dmem_buffer);
  for (uint32_t i = threadIdx.x; i < (fanout << log2_tuples_per_dmem_buffer);
       i += blockDim.x) {
    uint32_t p_index = i >> log2_tuples_per_dmem_buffer;
    uint32_t slot = i & (tuples_per_dmem_buffer - 1U);

    if (slot < (static_cast<unsigned int>(slots[p_index] >> smem_slot_bits)
                << log2_tuples_per_buffer)) {
      uint32_t dst =
          atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
      partitioned_relation[dst] = dmem_buffers[i];
    }
  }

  // Flush smem buffers directly to memory, because we don't keep track of dmem
  // buffer fill state on tuple-wise granularity.
  for (uint32_t i = threadIdx.x; i < (fanout << log2_tuples_per_buffer);
       i += blockDim.x) {
    uint32_t p_index = i >> log2_tuples_per_buffer;
    uint32_t slot = i & (tuples_per_buffer - 1U);

    if (slot < (slots[p_index] & smem_slot_mask)) {
      uint32_t dst =
          atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
      partitioned_relation[dst] = buffers[i];
    }
  }

  // Handle case when data_length % warpSize != 0
  for (size_t i = loop_length + threadIdx.x; i < data_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    auto p_index = key_to_partition(tuple.key, mask, 0);
    auto offset =
        atomicAdd((unsigned int *)&tmp_partition_offsets[p_index], 1U);
    partitioned_relation[offset] = tuple;
  }
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_chunked_radix_partition_int32_int32(RadixPartitionArgs *args) {
  gpu_chunked_radix_partition<int32_t, int32_t>(*args);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_chunked_radix_partition_int64_int64(RadixPartitionArgs *args) {
  gpu_chunked_radix_partition<int64_t, int64_t>(*args);
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_laswwc_radix_partition_int32_int32(
        RadixPartitionArgs *args) {
  gpu_chunked_laswwc_radix_partition<int32_t, int32_t>(*args);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_laswwc_radix_partition_int64_int64(
        RadixPartitionArgs *args) {
  gpu_chunked_laswwc_radix_partition<int64_t, int64_t>(*args);
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_sswwc_radix_partition_int32_int32(
        RadixPartitionArgs *args, uint32_t shared_mem_bytes) {
  gpu_chunked_sswwc_radix_partition<int, int, false>(*args, shared_mem_bytes);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_sswwc_radix_partition_int64_int64(
        RadixPartitionArgs *args, uint32_t shared_mem_bytes) {
  gpu_chunked_sswwc_radix_partition<long long, long long, false>(
      *args, shared_mem_bytes);
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_sswwc_non_temporal_radix_partition_int32_int32(
        RadixPartitionArgs *args, uint32_t shared_mem_bytes) {
  gpu_chunked_sswwc_radix_partition<int, int, true>(*args, shared_mem_bytes);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_sswwc_non_temporal_radix_partition_int64_int64(
        RadixPartitionArgs *args, uint32_t shared_mem_bytes) {
  gpu_chunked_sswwc_radix_partition<long long, long long, true>(
      *args, shared_mem_bytes);
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_sswwc_radix_partition_v2_int32_int32(
        RadixPartitionArgs *args, uint32_t shared_mem_bytes) {
  gpu_chunked_sswwc_radix_partition_v2<int, int>(*args, shared_mem_bytes);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_sswwc_radix_partition_v2_int64_int64(
        RadixPartitionArgs *args, uint32_t shared_mem_bytes) {
  gpu_chunked_sswwc_radix_partition_v2<long long, long long>(*args,
                                                             shared_mem_bytes);
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_hsswwc_radix_partition_int32_int32(
        RadixPartitionArgs *args, uint32_t shared_mem_bytes) {
  gpu_chunked_hsswwc_radix_partition<int, int>(*args, shared_mem_bytes);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_hsswwc_radix_partition_int64_int64(
        RadixPartitionArgs *args, uint32_t shared_mem_bytes) {
  gpu_chunked_hsswwc_radix_partition<long long, long long>(*args,
                                                           shared_mem_bytes);
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_hsswwc_radix_partition_v2_int32_int32(
        RadixPartitionArgs *args, uint32_t shared_mem_bytes) {
  gpu_chunked_hsswwc_radix_partition_v2<int, int>(*args, shared_mem_bytes);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_hsswwc_radix_partition_v2_int64_int64(
        RadixPartitionArgs *args, uint32_t shared_mem_bytes) {
  gpu_chunked_hsswwc_radix_partition_v2<long long, long long>(*args,
                                                              shared_mem_bytes);
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_hsswwc_radix_partition_v3_int32_int32(
        RadixPartitionArgs *args, uint32_t shared_mem_bytes) {
  gpu_chunked_hsswwc_radix_partition_v3<int, int>(*args, shared_mem_bytes);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_hsswwc_radix_partition_v3_int64_int64(
        RadixPartitionArgs *args, uint32_t shared_mem_bytes) {
  gpu_chunked_hsswwc_radix_partition_v3<long long, long long>(*args,
                                                              shared_mem_bytes);
}
