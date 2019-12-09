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

#include <cassert>
#include <cstdint>
#include <cstring>

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
  size_t const padding_length;
  uint32_t const radix_bits;

  // State
  uint64_t *const __restrict__ tmp_partition_offsets;
  void *const __restrict__ write_combine_buffer;

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
__device__ T key_to_partition(T key, T mask, B bits) {
  return (key & mask) >> bits;
}

// Chunked radix partitioning.
//
// See the Rust module for details.
// FIXME: support for running multiple grid size > 1
template <typename K, typename V>
__device__ void gpu_chunked_radix_partition(RadixPartitionArgs &args) {
  extern __shared__ uint64_t shared_mem[];

  auto join_attr_data = static_cast<const K *>(args.join_attr_data);
  auto payload_attr_data = static_cast<const V *>(args.payload_attr_data);
  auto partitioned_relation =
      static_cast<Tuple<K, V> *>(args.partitioned_relation);

  const size_t fanout = 1ULL << args.radix_bits;
  const K mask = static_cast<K>(fanout - 1);

  uint64_t *const prefix_tmp = shared_mem;
  uint64_t *const tmp_partition_offsets = &shared_mem[blockDim.x / warpSize];

  // Ensure counters are all zeroed
  for (size_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] = 0;
  }

  __syncthreads();

  // 1. Compute local histograms per partition for thread block
  for (size_t i = threadIdx.x; i < args.data_length; i += blockDim.x) {
    auto key = join_attr_data[i];
    auto p_index = key_to_partition(key, mask, 0);
    atomicAdd((unsigned long long *)&tmp_partition_offsets[p_index], 1ULL);
  }

  __syncthreads();

  // 2. Compute offsets with exclusive prefix sum for thread block
  block_exclusive_prefix_sum(tmp_partition_offsets, fanout, args.padding_length,
                             prefix_tmp);

  __syncthreads();

  for (size_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    args.partition_offsets[i] = tmp_partition_offsets[i];
  }

  // 3. Partition
  for (size_t i = threadIdx.x; i < args.data_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    auto p_index = key_to_partition(tuple.key, mask, 0);
    auto offset =
        atomicAdd((unsigned long long *)&tmp_partition_offsets[p_index], 1ULL);
    __threadfence_block();
    partitioned_relation[offset] = tuple;
  }
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" __global__ void gpu_chunked_radix_partition_int32_int32(
    RadixPartitionArgs *args) {
  gpu_chunked_radix_partition<int32_t, int32_t>(*args);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __global__ void gpu_chunked_radix_partition_int64_int64(
    RadixPartitionArgs *args) {
  gpu_chunked_radix_partition<int64_t, int64_t>(*args);
}
