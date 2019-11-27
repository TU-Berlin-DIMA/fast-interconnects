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

// Defines the cache-line size; usually this should be passed via the build
// script.
#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 128
#endif

// Defines the software write-combine buffer size; usually this should be passed
// via the build script.
#ifndef SWWC_BUFFER_SIZE
#define SWWC_BUFFER_SIZE CACHE_LINE_SIZE
#endif

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

// A set of buffers used for software write-combinining.
//
// Supports two views of its data. The purpose is to align each buffer to the
// cache-line size. This requires periodic overwriting of `meta.slot` when a
// buffer becomes full. After emptying the buffer, the slot's value must be
// restored.
template <typename T, uint32_t size>
union WriteCombineBuffer {
  struct {
    T data[size / sizeof(T)];
  } tuples;

  struct {
    T data[(size - sizeof(uint64_t)) / sizeof(T)];
    char _padding[(size - sizeof(uint64_t)) -
                  (((size - sizeof(uint64_t)) / sizeof(T)) * sizeof(T))];
    uint64_t slot;  // Padding makes `slot` 8-byte aligned if sizeof(T) % 8 != 0
  } meta;

  // Computes the number of tuples contained in a buffer.
  static constexpr size_t tuples_per_buffer() { return size / sizeof(T); }
} __attribute__((packed));

// Computes the partition ID of a given key.
template <typename T, typename B>
__device__ T key_to_partition(T key, T mask, B bits) {
  return (key & mask) >> bits;
}

// Chunked radix partitioning.
//
// See the Rust module for details.
template <typename K, typename V>
__device__ void gpu_chunked_radix_partition(RadixPartitionArgs &args) {
  auto join_attr_data = static_cast<const K *>(args.join_attr_data);
  auto payload_attr_data = static_cast<const V *>(args.payload_attr_data);
  auto partitioned_relation =
      static_cast<Tuple<K, V> *>(args.partitioned_relation);

  const size_t fanout = 1UL << args.radix_bits;
  const K mask = static_cast<K>(fanout - 1);

  // Ensure counters are all zeroed
  for (size_t i = 0; i < fanout; ++i) {
    args.partition_offsets[i] = 0;
  }

  // 1. Compute local histograms per partition
  for (size_t i = 0; i < args.data_length; ++i) {
    auto key = join_attr_data[i];
    auto p_index = key_to_partition(key, mask, 0);
    args.partition_offsets[p_index] += 1;
  }

  // 2. Compute offsets with exclusive prefix sum
  for (size_t i = 0, sum = 0, offset = 0; i < fanout; ++i, offset = sum) {
    sum += args.partition_offsets[i];
    offset += (i + 1) * args.padding_length;
    args.partition_offsets[i] = offset;
    args.tmp_partition_offsets[i] = offset;
  }

  // 3. Partition
  for (size_t i = 0; i < args.data_length; ++i) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    auto p_index = key_to_partition(tuple.key, mask, 0);
    auto &offset = args.tmp_partition_offsets[p_index];
    partitioned_relation[offset] = tuple;
    offset += 1;
  }
}

// Block-wise radix partitioning.
//
// See the Rust module for details.
// FIXME: support for running multiple grid size > 1
template <typename K, typename V>
__device__ void gpu_block_radix_partition(RadixPartitionArgs &args) {
  extern __shared__ size_t shared_mem[];

  auto join_attr_data = static_cast<const K *>(args.join_attr_data);
  auto payload_attr_data = static_cast<const V *>(args.payload_attr_data);
  auto partitioned_relation =
      static_cast<Tuple<K, V> *>(args.partitioned_relation);

  const size_t fanout = 1ULL << args.radix_bits;
  const K mask = static_cast<K>(fanout - 1);

  size_t *const prefix_tmp = shared_mem;
  size_t *const tmp_partition_offsets = &shared_mem[blockDim.x / warpSize];

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

// Exports the the size of all SWWC buffers.
extern "C" size_t gpu_swwc_buffer_bytes() { return SWWC_BUFFER_SIZE; }

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

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" __global__ void gpu_block_radix_partition_int32_int32(
    RadixPartitionArgs *args) {
  gpu_block_radix_partition<int32_t, int32_t>(*args);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __global__ void gpu_block_radix_partition_int64_int64(
    RadixPartitionArgs *args) {
  gpu_block_radix_partition<int64_t, int64_t>(*args);
}
