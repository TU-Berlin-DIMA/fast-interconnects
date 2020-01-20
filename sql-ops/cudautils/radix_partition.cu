/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019 Clemens Lutz, German Research Center for Artificial
 * Intelligence
 * Author: Clemens Lutz, DFKI GmbH <clemens.lutz@dfki.de>
 */

#include <prefix_scan.h>

#ifndef TUPLES_PER_THREAD
#define TUPLES_PER_THREAD 4U
#endif

#include <cassert>
#include <cstdint>

using namespace std;

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
__device__ void gpu_chunked_sswwc_radix_partition(RadixPartitionArgs &args) {
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

  // FIXME Handle case: data_length % TUPLES_PER_THREAD != 0

  // Reuse space from prefix sum for storing cache offsets
  uint32_t *const cache_offsets = prefix_tmp;
  data_length = (data_length / (blockDim.x * TUPLES_PER_THREAD)) * (blockDim.x * TUPLES_PER_THREAD);
  for (size_t i = threadIdx.x; i < data_length;
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
    void gpu_chunked_sswwc_radix_partition_int32_int32(
        RadixPartitionArgs *args) {
  gpu_chunked_sswwc_radix_partition<int32_t, int32_t>(*args);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_sswwc_radix_partition_int64_int64(
        RadixPartitionArgs *args) {
  gpu_chunked_sswwc_radix_partition<int64_t, int64_t>(*args);
}
