// Copyright 2019-2022 Clemens Lutz
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

#define CUDA_MODIFIER __device__

#include <gpu_common.h>
#include <gpu_radix_partition.h>
#include <prefix_scan.h>
#include <ptx_memory.h>

// Grid synchronization is only supported on Pascal and higher, and will not
// compile on Maxwell or lower.
#if __CUDA_ARCH__ >= 600
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif

#include <cassert>
#include <cstdint>

using namespace std;

__device__ __forceinline__ uint32_t write_combine_slot(
    uint32_t tuples_per_buffer, uint32_t p_index, uint32_t slot) {
  return tuples_per_buffer * p_index + slot;
}

// Chunked prefix sum computation
template <typename K>
__device__ void gpu_chunked_prefix_sum(PrefixSumArgs &args) {
  extern __shared__ uint32_t shared_mem[];

  const uint32_t fanout = 1U << args.radix_bits;
  const uint64_t mask = static_cast<uint64_t>(fanout - 1U) << args.ignore_bits;
  constexpr size_t input_align_mask =
      ~(static_cast<size_t>(ALIGN_BYTES / sizeof(K)) - 1ULL);

  // Calculate the data_length per block
  size_t data_length =
      ((args.data_length + gridDim.x - 1U) / gridDim.x) & input_align_mask;
  size_t data_offset = data_length * blockIdx.x;
  size_t partitioned_data_offset =
      (data_length + args.padding_length * fanout) * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = args.data_length - data_offset;
  }

  auto partition_attr =
      reinterpret_cast<const K *>(args.partition_attr) + data_offset;

  unsigned int *const tmp_partition_offsets =
      reinterpret_cast<unsigned int *>(shared_mem);
  unsigned int *const prefix_tmp = &tmp_partition_offsets[fanout];

  // Ensure counters are all zeroed.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] = 0;
  }

  __syncthreads();

  // 1. Compute local histograms per partition for thread block.
  for (size_t i = threadIdx.x; i < data_length; i += blockDim.x) {
    auto key = partition_attr[i];
    auto p_index = key_to_partition(key, mask, args.ignore_bits);
    atomicAdd(&tmp_partition_offsets[p_index], 1U);
  }

  __syncthreads();

  // 2. Compute offsets with exclusive prefix sum for thread block.
  block_exclusive_prefix_sum(tmp_partition_offsets, fanout, 0, prefix_tmp);

  __syncthreads();

  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    // Add block offset and padding to device memory slots.
    uint64_t offset = static_cast<uint64_t>(tmp_partition_offsets[i]) +
                      partitioned_data_offset + (i + 1) * args.padding_length;

    // Add data offset onto partitions offsets and write out the final offsets
    // to device memory.
    args.partition_offsets[blockIdx.x * fanout + i] = offset;
  }
}

// Contiguous prefix sum computation
//
// Note that this function must be launched with cudaLaunchCooperativeKernel.
// Cooperative grid synchronization is only supported on Pascal and later GPUs.
// For testing on pre-Pascal GPUs, the function can be launched with
// grid_size = 1.
template <typename K>
__device__ void gpu_contiguous_prefix_sum(PrefixSumArgs &args) {
  extern __shared__ uint32_t shared_mem[];

#if __CUDA_ARCH__ >= 600
  cg::grid_group grid = cg::this_grid();
#endif
  const uint32_t fanout = 1U << args.radix_bits;
  const uint64_t mask = static_cast<uint64_t>(fanout - 1U) << args.ignore_bits;
  constexpr size_t input_align_mask =
      ~(static_cast<size_t>(ALIGN_BYTES / sizeof(K)) - 1ULL);

  // Calculate the data_length per block
  size_t data_length =
      ((args.data_length + gridDim.x - 1U) / gridDim.x) & input_align_mask;
  size_t data_offset = data_length * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = args.data_length - data_offset;
  }

  auto partition_attr =
      reinterpret_cast<const K *>(args.partition_attr) + data_offset;

  unsigned long long *const tmp_partition_offsets =
      reinterpret_cast<unsigned long long *>(shared_mem);

  // Ensure counters are all zeroed.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] = 0;
  }

  __syncthreads();

  // Compute local histograms per partition for thread block.
  for (size_t i = threadIdx.x; i < data_length; i += blockDim.x) {
    auto key = partition_attr[i];
    auto p_index = key_to_partition(key, mask, args.ignore_bits);
    atomicAdd(&tmp_partition_offsets[p_index], 1U);
  }

  __syncthreads();

  // Copy local histograms to device memory.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    args.tmp_partition_offsets[gridDim.x * i + blockIdx.x] =
        tmp_partition_offsets[i];
  }

  // Initialize prefix sum state before synchronizing the grid.
  device_exclusive_prefix_sum_initialize(args.prefix_scan_state);

#if __CUDA_ARCH__ >= 600
  grid.sync();
#else
  __syncthreads();
#endif

  // Compute offsets with exclusive prefix sum for thread block.
  device_exclusive_prefix_sum(args.tmp_partition_offsets, fanout * gridDim.x, 0,
                              args.prefix_scan_state);

#if __CUDA_ARCH__ >= 600
  grid.sync();
#else
  __syncthreads();
#endif

  // Temporarily buffer the prefix sums into shared memory, as we need to
  // transpose them in memory. Add padding between the contiguous partitions,
  // but not between the chunks within a partition.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] =
        args.tmp_partition_offsets[gridDim.x * i + blockIdx.x] +
        (i + 1) * args.padding_length;
  }

#if __CUDA_ARCH__ >= 600
  grid.sync();
#else
  __syncthreads();
#endif

  // Write the transposed partition offsets to global memory.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    args.tmp_partition_offsets[fanout * blockIdx.x + i] =
        tmp_partition_offsets[i];
  }

  // Write partitions offsets to global memory. As there exists only one chunk
  // only the first block must perform the write.
  if (blockIdx.x == 0) {
    for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
      args.partition_offsets[i] = tmp_partition_offsets[i];
    }
  }
}

// Contiguous prefix sum and copy with payload
//
// Note that this function must be launched with cudaLaunchCooperativeKernel.
// Cooperative grid synchronization is only supported on Pascal and later GPUs.
// For testing on pre-Pascal GPUs, the function can be launched with
// grid_size = 1.
template <typename K, typename V>
__device__ void gpu_contiguous_prefix_sum_and_copy_with_payload(
    PrefixSumAndCopyWithPayloadArgs &args) {
  extern __shared__ uint32_t shared_mem[];

#if __CUDA_ARCH__ >= 600
  cg::grid_group grid = cg::this_grid();
#endif
  const uint32_t fanout = 1U << args.radix_bits;
  const uint64_t mask = static_cast<uint64_t>(fanout - 1U) << args.ignore_bits;
  constexpr size_t input_align_mask =
      ~(static_cast<size_t>(ALIGN_BYTES / sizeof(K)) - 1ULL);

  // Calculate the data_length per block
  size_t data_length =
      ((args.data_length + gridDim.x - 1U) / gridDim.x) & input_align_mask;
  size_t data_offset = data_length * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = args.data_length - data_offset;
  }

  auto src_partition_attr =
      reinterpret_cast<const K *>(args.src_partition_attr) + data_offset;
  auto src_payload_attr =
      reinterpret_cast<const V *>(args.src_payload_attr) + data_offset;
  auto dst_partition_attr =
      reinterpret_cast<K *>(args.dst_partition_attr) + data_offset;
  auto dst_payload_attr =
      reinterpret_cast<V *>(args.dst_payload_attr) + data_offset;

  unsigned long long *const tmp_partition_offsets =
      reinterpret_cast<unsigned long long *>(shared_mem);

  // Ensure counters are all zeroed.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] = 0;
  }

  __syncthreads();

  // Compute local histograms per partition for thread block.
  for (size_t i = threadIdx.x; i < data_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = src_partition_attr[i];
    tuple.value = src_payload_attr[i];

    dst_partition_attr[i] = tuple.key;
    dst_payload_attr[i] = tuple.value;

    auto p_index = key_to_partition(tuple.key, mask, args.ignore_bits);
    atomicAdd(&tmp_partition_offsets[p_index], 1U);
  }

  __syncthreads();

  // Copy local histograms to device memory.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    args.tmp_partition_offsets[gridDim.x * i + blockIdx.x] =
        tmp_partition_offsets[i];
  }

  // Initialize prefix sum state before synchronizing the grid.
  device_exclusive_prefix_sum_initialize(args.prefix_scan_state);

#if __CUDA_ARCH__ >= 600
  grid.sync();
#else
  __syncthreads();
#endif

  // Compute offsets with exclusive prefix sum for thread block.
  device_exclusive_prefix_sum(args.tmp_partition_offsets, fanout * gridDim.x, 0,
                              args.prefix_scan_state);

#if __CUDA_ARCH__ >= 600
  grid.sync();
#else
  __syncthreads();
#endif

  // Temporarily buffer the prefix sums into shared memory, as we need to
  // transpose them in memory. Add padding between the contiguous partitions,
  // but not between the chunks within a partition.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] =
        args.tmp_partition_offsets[gridDim.x * i + blockIdx.x] +
        (i + 1) * args.padding_length;
  }

#if __CUDA_ARCH__ >= 600
  grid.sync();
#else
  __syncthreads();
#endif

  // Write the transposed partition offsets to global memory.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    args.tmp_partition_offsets[fanout * blockIdx.x + i] =
        tmp_partition_offsets[i];
  }

  // Write partitions offsets to global memory. As there exists only one chunk
  // only the first block must perform the write.
  if (blockIdx.x == 0) {
    for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
      args.partition_offsets[i] = tmp_partition_offsets[i];
    }
  }
}

// Contiguous prefix sum and transform
//
// Note that this function must be launched with cudaLaunchCooperativeKernel.
// Cooperative grid synchronization is only supported on Pascal and later GPUs.
// For testing on pre-Pascal GPUs, the function can be launched with
// grid_size = 1.
template <typename K, typename V>
__device__ void gpu_contiguous_prefix_sum_and_transform(
    PrefixSumAndTransformArgs &args) {
  extern __shared__ uint32_t shared_mem[];

#if __CUDA_ARCH__ >= 600
  cg::grid_group grid = cg::this_grid();
#endif
  const uint32_t src_fanout = 1U << args.src_radix_bits;
  const uint32_t fanout = 1U << args.radix_bits;
  const uint64_t mask = static_cast<uint64_t>(fanout - 1U) << args.ignore_bits;
  constexpr size_t input_align_mask =
      ~(static_cast<size_t>(ALIGN_BYTES / sizeof(K)) - 1ULL);

  unsigned long long *const tmp_partition_offsets =
      reinterpret_cast<unsigned long long *>(shared_mem);
  unsigned long long *const src_chunks_offset = &tmp_partition_offsets[fanout];
  unsigned long long *const src_chunks_len =
      &src_chunks_offset[args.src_chunks];

  // Load chunk offsets
  size_t src_data_len = 0U;
  for (uint32_t i = threadIdx.x; i < args.src_chunks; i += blockDim.x) {
    auto chunk_begin = args.src_offsets[i * src_fanout + args.partition_id];
    auto chunk_end =
        (i + 1 == args.src_chunks && args.partition_id + 1 == src_fanout)
            ? args.data_length
            : args.src_offsets[i * src_fanout + args.partition_id + 1U] -
                  args.padding_length;
    auto len = chunk_end - chunk_begin;

    src_chunks_offset[i] = chunk_begin;
    src_chunks_len[i] = len;
    src_data_len += len;
  }

  // Aggregate length of all chunks
  if (threadIdx.x == 0) {
    tmp_partition_offsets[0] = 0;
  }
  __syncthreads();
  atomicAdd(&tmp_partition_offsets[0], src_data_len);
  __syncthreads();
  src_data_len = tmp_partition_offsets[0];
  // Ensure all threads have read src_data_len before initializing
  // tmp_partition_offsets
  __syncthreads();

  // Calculate the data_length per block
  size_t data_length =
      ((src_data_len + gridDim.x - 1U) / gridDim.x) & input_align_mask;
  size_t data_offset = data_length * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = src_data_len - data_offset;
  }

  auto src_relation = reinterpret_cast<const Tuple<K, V> *>(args.src_relation);
  auto dst_partition_attr = reinterpret_cast<K *>(args.dst_partition_attr);
  auto dst_payload_attr = reinterpret_cast<V *>(args.dst_payload_attr);

  // Ensure counters are all zeroed.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] = 0;
  }

  __syncthreads();

  // Compute local histograms per partition for thread block.
  uint32_t chunk = 0xFFFFFFFFU;
  size_t chunk_end = 0;
  for (size_t i = data_offset + threadIdx.x, pos = 0;
       i < data_offset + data_length; i += blockDim.x, pos += blockDim.x) {
    while (i >= chunk_end) {
      chunk += 1;
      pos = src_chunks_offset[chunk] + (i - chunk_end);
      chunk_end += src_chunks_len[chunk];
    }

    // size_t pos = chunk_offset + (i - chunk_begin);
    Tuple<K, V> tuple;
    tuple.load(src_relation[pos]);

    dst_partition_attr[i] = tuple.key;
    dst_payload_attr[i] = tuple.value;

    auto p_index = key_to_partition(tuple.key, mask, args.ignore_bits);
    atomicAdd(&tmp_partition_offsets[p_index], 1U);
  }

  __syncthreads();

  // Copy local histograms to device memory.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    args.tmp_partition_offsets[gridDim.x * i + blockIdx.x] =
        tmp_partition_offsets[i];
  }

  // Initialize prefix sum state before synchronizing the grid.
  device_exclusive_prefix_sum_initialize(args.prefix_scan_state);

#if __CUDA_ARCH__ >= 600
  grid.sync();
#else
  __syncthreads();
#endif

  // Compute offsets with exclusive prefix sum for thread block.
  device_exclusive_prefix_sum(args.tmp_partition_offsets, fanout * gridDim.x, 0,
                              args.prefix_scan_state);

#if __CUDA_ARCH__ >= 600
  grid.sync();
#else
  __syncthreads();
#endif

  // Temporarily buffer the prefix sums into shared memory, as we need to
  // transpose them in memory. Add padding between the contiguous partitions,
  // but not between the chunks within a partition.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] =
        args.tmp_partition_offsets[gridDim.x * i + blockIdx.x] +
        (i + 1) * args.padding_length;
  }

#if __CUDA_ARCH__ >= 600
  grid.sync();
#else
  __syncthreads();
#endif

  // Write the transposed partition offsets to global memory.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    args.tmp_partition_offsets[fanout * blockIdx.x + i] =
        tmp_partition_offsets[i];
  }

  // Write partitions offsets to global memory. As there exists only one chunk
  // only the first block must perform the write.
  if (blockIdx.x == 0) {
    for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
      args.partition_offsets[i] = tmp_partition_offsets[i];
    }
  }
}

// Non-cached radix partitioning.
//
// See the Rust module for details.
template <typename K, typename V>
__device__ void gpu_chunked_radix_partition(RadixPartitionArgs &args) {
  extern __shared__ uint32_t shared_mem[];

  const uint32_t fanout = 1U << args.radix_bits;
  const uint64_t mask = static_cast<uint64_t>(fanout - 1U) << args.ignore_bits;
  constexpr size_t input_align_mask =
      ~(static_cast<size_t>(ALIGN_BYTES / sizeof(K)) - 1ULL);

  // Calculate the data_length per block
  size_t data_length =
      ((args.data_length + gridDim.x - 1U) / gridDim.x) & input_align_mask;
  size_t data_offset = data_length * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = args.data_length - data_offset;
  }
  unsigned long long partitioned_relation_offset =
      args.partition_offsets[blockIdx.x * fanout];

  auto join_attr_data =
      reinterpret_cast<const K *>(args.join_attr_data) + data_offset;
  auto payload_attr_data =
      reinterpret_cast<const V *>(args.payload_attr_data) + data_offset;
  // Handle relations larger than 32 GiB that cause integer overflow.  Subtract
  // chunk offset so that relative value is within range of unsigned int.
  auto partitioned_relation =
      reinterpret_cast<Tuple<K, V> *>(args.partitioned_relation) +
      partitioned_relation_offset;

  unsigned int *const tmp_partition_offsets =
      reinterpret_cast<unsigned int *>(shared_mem);

  // Load partition offsets from device memory into shared memory.
  // Sub-optimal memory access pattern doesn't matter, because we are reading
  // at maxiumum 3 MB data.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] = static_cast<unsigned int>(
        args.partition_offsets[blockIdx.x * fanout + i] -
        partitioned_relation_offset);
  }

  __syncthreads();

  // Partition data
  for (size_t i = threadIdx.x; i < data_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    auto p_index = key_to_partition(tuple.key, mask, args.ignore_bits);
    auto offset = atomicAdd(&tmp_partition_offsets[p_index], 1U);
    tuple.store(partitioned_relation[offset]);
  }
}

template <typename K, typename V>
__device__ void gpu_chunked_laswwc_radix_partition(RadixPartitionArgs &args,
                                                   uint32_t shared_mem_bytes) {
  extern __shared__ uint32_t shared_mem[];

  const uint32_t fanout = 1U << args.radix_bits;
  const uint64_t mask = static_cast<uint64_t>(fanout - 1) << args.ignore_bits;
  constexpr size_t input_align_mask =
      ~(static_cast<size_t>(ALIGN_BYTES / sizeof(K)) - 1ULL);

  // Calculate the data_length per block
  size_t data_length =
      ((args.data_length + gridDim.x - 1U) / gridDim.x) & input_align_mask;
  size_t data_offset = data_length * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = args.data_length - data_offset;
  }
  unsigned long long partitioned_relation_offset =
      args.partition_offsets[blockIdx.x * fanout];

  auto join_attr_data =
      reinterpret_cast<const K *>(args.join_attr_data) + data_offset;
  auto payload_attr_data =
      reinterpret_cast<const V *>(args.payload_attr_data) + data_offset;
  // Handle relations larger than 32 GiB that cause integer overflow.  Subtract
  // chunk offset so that relative value is within range of unsigned int.
  auto partitioned_relation =
      reinterpret_cast<Tuple<K, V> *>(args.partitioned_relation) +
      partitioned_relation_offset;

  assert(((size_t)join_attr_data) % (ALIGN_BYTES / sizeof(K)) == 0U &&
         "Key column should be aligned to ALIGN_BYTES for best performance");
  assert(
      ((size_t)payload_attr_data) % (ALIGN_BYTES / sizeof(K)) == 0U &&
      "Payload column should be aligned to ALIGN_BYTES for best performance");

  const uint32_t laswwc_bytes =
      blockDim.x * LASWWC_TUPLES_PER_THREAD * (sizeof(K) + sizeof(V)) +
      2U * fanout * sizeof(uint32_t);

  assert(laswwc_bytes <= shared_mem_bytes &&
         "LA-SWWC buffer must fit into shared memory");

  K *const cached_keys = (K *)shared_mem;
  V *const cached_vals =
      (V *)&cached_keys[blockDim.x * LASWWC_TUPLES_PER_THREAD];
  unsigned int *const tmp_partition_offsets = reinterpret_cast<unsigned int *>(
      &cached_vals[blockDim.x * LASWWC_TUPLES_PER_THREAD]);
  unsigned int *const cache_offsets =
      reinterpret_cast<unsigned int *>(&tmp_partition_offsets[fanout]);

  // Load partition offsets from device memory into shared memory.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] = static_cast<unsigned int>(
        args.partition_offsets[blockIdx.x * fanout + i] -
        partitioned_relation_offset);
  }

  __syncthreads();

  // Partition data
  size_t loop_length = (data_length / (blockDim.x * LASWWC_TUPLES_PER_THREAD)) *
                       (blockDim.x * LASWWC_TUPLES_PER_THREAD);

  for (size_t i = threadIdx.x; i < loop_length;
       i += blockDim.x * LASWWC_TUPLES_PER_THREAD) {
    // Load tuples
    Tuple<K, V> tuple[LASWWC_TUPLES_PER_THREAD];
#pragma unroll
    for (uint32_t k = 0; k < LASWWC_TUPLES_PER_THREAD; ++k) {
      tuple[k].key = join_attr_data[i + k * blockDim.x];
      tuple[k].value = payload_attr_data[i + k * blockDim.x];
    }

    // Ensure counters are all zeroed.
    for (uint32_t h = threadIdx.x; h < fanout; h += blockDim.x) {
      cache_offsets[h] = 0;
    }

    __syncthreads();

#pragma unroll
    for (uint32_t k = 0; k < LASWWC_TUPLES_PER_THREAD; ++k) {
      // Hash keys to partition IDs
      auto p_index = key_to_partition(tuple[k].key, mask, args.ignore_bits);

      // Build histogram of cached tuples
      atomicAdd(&cache_offsets[p_index], 1U);
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
    for (uint32_t k = 0; k < LASWWC_TUPLES_PER_THREAD; ++k) {
      auto p_index = key_to_partition(tuple[k].key, mask, args.ignore_bits);
      auto pos = atomicAdd(&cache_offsets[p_index], 1U);
      cached_keys[pos] = tuple[k].key;
      cached_vals[pos] = tuple[k].value;
    }

    __syncthreads();

    // Write tuples to global memory
#pragma unroll
    for (uint32_t k = threadIdx.x; k < blockDim.x * LASWWC_TUPLES_PER_THREAD;
         k += blockDim.x) {
      Tuple<K, V> tuple;
      tuple.key = cached_keys[k];
      tuple.value = cached_vals[k];
      auto p_index = key_to_partition(tuple.key, mask, args.ignore_bits);

      unsigned int offset = cache_offsets[p_index] - (k + 1);
      offset += tmp_partition_offsets[p_index];

      tuple.store(partitioned_relation[offset]);
    }

    __syncthreads();

    // Update partition offsets for next loop iteration
#pragma unroll
    for (uint32_t k = threadIdx.x; k < blockDim.x * LASWWC_TUPLES_PER_THREAD;
         k += blockDim.x) {
      auto key = cached_keys[k];
      auto p_index = key_to_partition(key, mask, args.ignore_bits);
      atomicAdd(&tmp_partition_offsets[p_index], 1);
    }
  }

  __syncthreads();

  // Handle case when data_length % (LASWWC_TUPLES_PER_THREAD * blockDim) != 0
  for (size_t i = loop_length + threadIdx.x; i < data_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    auto p_index = key_to_partition(tuple.key, mask, args.ignore_bits);
    auto offset = atomicAdd(&tmp_partition_offsets[p_index], 1U);
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
  const uint64_t mask = static_cast<uint64_t>(fanout - 1) << args.ignore_bits;
  const int lane_id = threadIdx.x % warpSize;
  constexpr uint32_t warp_mask = 0xFFFFFFFFu;
  constexpr size_t input_align_mask =
      ~(static_cast<size_t>(ALIGN_BYTES / sizeof(K)) - 1ULL);
  constexpr uint32_t align_tuples = ALIGN_BYTES / sizeof(Tuple<K, V>);

  assert(align_tuples <= args.padding_length &&
         "Padding must be large enough for alignment");

  // Calculate the data_length per block
  size_t data_length =
      ((args.data_length + gridDim.x - 1U) / gridDim.x) & input_align_mask;
  size_t data_offset = data_length * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = args.data_length - data_offset;
  }

  // Substract the alignment padding to avoid an unsigned integer underflow
  // when computing the aligned array offsets.
  unsigned long long partitioned_relation_offset =
      args.partition_offsets[blockIdx.x * fanout] -
      ALIGN_BYTES / sizeof(Tuple<K, V>);

  auto join_attr_data =
      reinterpret_cast<const K *>(args.join_attr_data) + data_offset;
  auto payload_attr_data =
      reinterpret_cast<const V *>(args.payload_attr_data) + data_offset;
  // Handle relations larger than 32 GiB that cause integer overflow.  Subtract
  // chunk offset so that relative value is within range of unsigned int.
  auto partitioned_relation =
      reinterpret_cast<Tuple<K, V> *>(args.partitioned_relation) +
      partitioned_relation_offset;

  assert(((size_t)join_attr_data) % (ALIGN_BYTES / sizeof(K)) == 0U &&
         "Key column should be aligned to ALIGN_BYTES for best performance");
  assert(
      ((size_t)payload_attr_data) % (ALIGN_BYTES / sizeof(V)) == 0U &&
      "Payload column should be aligned to ALIGN_BYTES for best performance");

  const uint32_t sswwc_buffer_bytes =
      shared_mem_bytes - 4U * fanout * sizeof(uint32_t);
  const uint32_t tuples_per_buffer =
      1U << log2_floor_power_of_two(sswwc_buffer_bytes / sizeof(Tuple<K, V>) /
                                    fanout);
  assert(tuples_per_buffer > 0 &&
         "At least one tuple per partition must fit into SWWC buffer");

  Tuple<K, V> *const buffers = reinterpret_cast<Tuple<K, V> *>(shared_mem);
  unsigned int *const tmp_partition_offsets =
      reinterpret_cast<unsigned int *>(&buffers[fanout * tuples_per_buffer]);
  unsigned int *const base_partition_offsets =
      reinterpret_cast<unsigned int *>(&tmp_partition_offsets[fanout]);
  unsigned int *const slots =
      reinterpret_cast<unsigned int *>(&base_partition_offsets[fanout]);
  unsigned int *const signal_slots = &slots[fanout];

  // Load partition offsets from device memory into shared memory.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    base_partition_offsets[i] = static_cast<unsigned int>(
        args.partition_offsets[blockIdx.x * fanout + i] -
        partitioned_relation_offset);
  }

  // Alignment offset for the initial slots
  const uintptr_t offset_align_mask =
      ~static_cast<uintptr_t>(ALIGN_BYTES - 1UL);

  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    // Align the initial slots to cache lines
    auto offset = base_partition_offsets[i];
    Tuple<K, V> *base_ptr = reinterpret_cast<Tuple<K, V> *>(
        reinterpret_cast<uintptr_t>(&partitioned_relation[offset]) &
        offset_align_mask);
    uint32_t aligned_fill_state = &partitioned_relation[offset] - base_ptr;
    aligned_fill_state = aligned_fill_state % tuples_per_buffer;
    tmp_partition_offsets[i] = offset - aligned_fill_state;

    slots[i] = aligned_fill_state;
    signal_slots[i] = aligned_fill_state;
  }

  // Initialize the buffers with NULL so that we don't write out uninitialized
  // data
  for (uint32_t i = threadIdx.x; i < fanout * tuples_per_buffer;
       i += blockDim.x) {
    buffers[i] = {null_key<K>(), {}};
  }

  __syncthreads();

  // 3. Partition

  // All threads in warp must participate in each loop iteration
  size_t loop_length = (data_length / warpSize) * warpSize;
  for (size_t i = threadIdx.x; i < loop_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    uint32_t p_index = key_to_partition(tuple.key, mask, args.ignore_bits);
    uint32_t pos = 0;
    bool done = false;
    do {
      // Fetch a position if don't have a valid position yet
      if (not done) {
        pos = atomicAdd(&slots[p_index], 1U);

        if (pos < tuples_per_buffer) {
          buffers[write_combine_slot(tuples_per_buffer, p_index, pos)] = tuple;

          // Wait until tuple write is flushed.
          __threadfence_block();

          // Signal that we are done writing the tuple.
          atomicAdd(&signal_slots[p_index], 1U);
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
      if ((ballot = __ballot_sync(warp_mask, is_candidate))) {
        int leader_id = __ffs(ballot) - 1;

        // Release the lock if not the leader and try again in next round.
        if (is_candidate && leader_id != lane_id) {
          atomicExch(&slots[p_index], tuples_per_buffer);
        }

        // Finish unlocking slots before entering busy-wait.
        __syncwarp();

        // Wait until all threads are done writing their tuples into the buffer.
        // Then reset the signal slot to zero.
        if (leader_id == lane_id) {
          while (atomicOr(&signal_slots[p_index], 0U) != tuples_per_buffer)
            ;
          atomicExch(&signal_slots[p_index], 0U);
        }
        __syncwarp();

        uint32_t current_index = __shfl_sync(warp_mask, p_index, leader_id);
        auto dst = tmp_partition_offsets[current_index];

        // Memcpy from cached buffer to memory
        //
        // If partitions small, then there exists a race condition in which
        // the first flush performed by a thread block overwrites the last
        // values written by the neighboring thread block. In this case,
        // threads are not allowed to write before the first valid
        // destination.
        //
        // The race condition typically occurs within a small partition, e.g.,
        // in the second partitioning pass.
        for (uint32_t i = lane_id; i < tuples_per_buffer; i += warpSize) {
          if (dst + i >= base_partition_offsets[current_index]) {
            Tuple<K, V> tuple = buffers[write_combine_slot(tuples_per_buffer,
                                                           current_index, i)];
            tuple.store(partitioned_relation[dst + i]);
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
        }
      }
    } while (__any_sync(warp_mask, not done));
  }

  // Wait until all warps are done
  __syncthreads();

  // Flush buffers; retain order of tuples as buffer is aligned and may contain
  // invalid tuples (i.e., don't use atomicAdd to get a destination slot)
  for (uint32_t i = threadIdx.x; i < fanout * tuples_per_buffer;
       i += blockDim.x) {
    uint32_t p_index = i / tuples_per_buffer;
    uint32_t slot = i % tuples_per_buffer;

    if (slot < slots[p_index]) {
      auto dst = tmp_partition_offsets[p_index] + slot;

      // Consider that buffer flush is aligned; don't invalid tuples in front
      // of the partition
      if (dst >= base_partition_offsets[p_index]) {
        Tuple<K, V> tuple = buffers[i];
        tuple.store(partitioned_relation[dst]);
      }
    }
  }

  __syncthreads();

  // Update offsets for partitioning of unbuffered tuples
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] += slots[i];
  }

  __syncthreads();

  // Handle case when data_length % warpSize != 0
  for (size_t i = loop_length + threadIdx.x; i < data_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    auto p_index = key_to_partition(tuple.key, mask, args.ignore_bits);
    auto offset = atomicAdd(&tmp_partition_offsets[p_index], 1U);
    partitioned_relation[offset] = tuple;
  }
}

/// SSWWC v2G is the same algorithm variant as v2, but stores the SWWC buffers
/// in device memory. It tries to keep them in the L2 cache by using cache
/// bypass instructions when necessary.
template <typename K, typename V>
__device__ void gpu_chunked_sswwc_radix_partition_v2g(
    RadixPartitionArgs &args) {
  const uint32_t fanout = 1U << args.radix_bits;
  const uint64_t mask = static_cast<uint64_t>(fanout - 1) << args.ignore_bits;
  const int lane_id = threadIdx.x % warpSize;
  constexpr uint32_t warp_mask = 0xFFFFFFFFu;
  constexpr size_t input_align_mask =
      ~(static_cast<size_t>(ALIGN_BYTES / sizeof(K)) - 1ULL);
  constexpr uint32_t align_tuples = ALIGN_BYTES / sizeof(Tuple<K, V>);

  assert(align_tuples <= args.padding_length &&
         "Padding must be large enough for alignment");

  // Calculate the data_length per block
  size_t data_length =
      ((args.data_length + gridDim.x - 1U) / gridDim.x) & input_align_mask;
  size_t data_offset = data_length * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = args.data_length - data_offset;
  }
  unsigned long long partitioned_relation_offset =
      args.partition_offsets[blockIdx.x * fanout];

  auto join_attr_data =
      reinterpret_cast<const K *>(args.join_attr_data) + data_offset;
  auto payload_attr_data =
      reinterpret_cast<const V *>(args.payload_attr_data) + data_offset;
  // Handle relations larger than 32 GiB that cause integer overflow.  Subtract
  // chunk offset so that relative value is within range of unsigned int.
  auto partitioned_relation =
      reinterpret_cast<Tuple<K, V> *>(args.partitioned_relation) +
      partitioned_relation_offset;

  assert(((size_t)join_attr_data) % (ALIGN_BYTES / sizeof(K)) == 0U &&
         "Key column should be aligned to ALIGN_BYTES for best performance");
  assert(
      ((size_t)payload_attr_data) % (ALIGN_BYTES / sizeof(V)) == 0U &&
      "Payload column should be aligned to ALIGN_BYTES for best performance");

  const uint32_t tuples_per_buffer = GPU_CACHE_LINE_SIZE / sizeof(Tuple<K, V>);
  assert(tuples_per_buffer > 0 &&
         "At least one tuple per partition must fit into SWWC buffer");

  Tuple<K, V> *const buffers =
      reinterpret_cast<Tuple<K, V> *>(args.l2_cache_buffers) +
      static_cast<size_t>(tuples_per_buffer) * fanout * blockIdx.x;
  unsigned int *const tmp_partition_offsets =
      reinterpret_cast<unsigned int *>(args.tmp_partition_offsets) +
      static_cast<size_t>(fanout) * 3UL * blockIdx.x;
  unsigned int *const slots =
      reinterpret_cast<unsigned int *>(&tmp_partition_offsets[fanout]);
  unsigned int *const signal_slots = &slots[fanout];

  // Load partition offsets from host or device memory into a device memory
  // buffer.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] = static_cast<unsigned int>(
        args.partition_offsets[blockIdx.x * fanout + i] -
        partitioned_relation_offset);
  }

  // Align the initial slots
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    auto offset = tmp_partition_offsets[i];
    uint32_t aligned_fill_state = offset % min(align_tuples, tuples_per_buffer);
    tmp_partition_offsets[i] = offset - aligned_fill_state;

    slots[i] = aligned_fill_state;
    signal_slots[i] = aligned_fill_state;
  }

  // Initialize the buffers with NULL so that we don't write out uninitialized
  // data
  for (uint32_t i = threadIdx.x; i < fanout * tuples_per_buffer;
       i += blockDim.x) {
    buffers[i] = {null_key<K>(), {}};
  }

  __syncthreads();

  // Partition

  // All threads in warp must participate in each loop iteration
  size_t loop_length = (data_length / warpSize) * warpSize;
  for (size_t i = threadIdx.x; i < loop_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = ptx_load_cache_streaming(&join_attr_data[i]);
    tuple.value = ptx_load_cache_streaming(&payload_attr_data[i]);

    uint32_t p_index = key_to_partition(tuple.key, mask, args.ignore_bits);
    uint32_t pos = 0;
    bool done = false;
    do {
      // Fetch a position if don't have a valid position yet
      if (not done) {
        pos = atomicAdd(&slots[p_index], 1U);

        if (pos < tuples_per_buffer) {
          buffers[write_combine_slot(tuples_per_buffer, p_index, pos)] = tuple;

          // Wait until tuple write is flushed.
          __threadfence_block();

          // Signal that we are done writing the tuple.
          atomicAdd(&signal_slots[p_index], 1U);
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
      if ((ballot = __ballot_sync(warp_mask, is_candidate))) {
        int leader_id = __ffs(ballot) - 1;

        // Release the lock if not the leader and try again in next round.
        if (is_candidate && leader_id != lane_id) {
          atomicExch(&slots[p_index], tuples_per_buffer);
        }

        // Finish unlocking slots before entering busy-wait.
        __syncwarp();

        // Wait until all threads are done writing their tuples into the buffer.
        // Then reset the signal slot to zero.
        if (leader_id == lane_id) {
          while (atomicOr(&signal_slots[p_index], 0U) != tuples_per_buffer)
            ;
          atomicExch(&signal_slots[p_index], 0U);
        }
        __syncwarp();

        uint32_t current_index = __shfl_sync(warp_mask, p_index, leader_id);
        auto dst = tmp_partition_offsets[current_index];

        // Memcpy from cached buffer to memory
        for (uint32_t i = lane_id; i < tuples_per_buffer; i += warpSize) {
          Tuple<K, V> tuple =
              buffers[write_combine_slot(tuples_per_buffer, current_index, i)];
          tuple.store_streaming(partitioned_relation[dst + i]);
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

  // Flush buffers; retain order of tuples as buffer is aligned and may contain
  // invalid tuples (i.e., don't use atomicAdd to get a destination slot)
  for (uint32_t i = threadIdx.x; i < fanout * tuples_per_buffer;
       i += blockDim.x) {
    uint32_t p_index = i / tuples_per_buffer;
    uint32_t slot = i % tuples_per_buffer;

    if (slot < slots[p_index]) {
      auto dst = tmp_partition_offsets[p_index] + slot;
      auto base_offset = static_cast<unsigned int>(
          args.partition_offsets[blockIdx.x * fanout + p_index] -
          partitioned_relation_offset);

      // Consider that buffer flush is aligned; don't invalid tuples in front
      // of the partition
      if (dst >= base_offset) {
        Tuple<K, V> tuple = buffers[i];
        tuple.store_streaming(partitioned_relation[dst]);
      }
    }
  }

  __syncthreads();

  // Update offsets for partitioning of unbuffered tuples
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    tmp_partition_offsets[i] += slots[i];
  }

  __syncthreads();

  // Handle case when data_length % warpSize != 0
  for (size_t i = loop_length + threadIdx.x; i < data_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = ptx_load_cache_streaming(&join_attr_data[i]);
    tuple.value = ptx_load_cache_streaming(&payload_attr_data[i]);

    auto p_index = key_to_partition(tuple.key, mask, args.ignore_bits);
    auto offset = atomicAdd(&tmp_partition_offsets[p_index], 1U);
    tuple.store_streaming(partitioned_relation[offset]);
  }
}

// HSSWWC v4 performs the buffer flush from dmem to memory asynchronously with
// double-buffering. Double-buffering ensures that all warps make progress
// during the dmem flush.
template <typename K, typename V>
__device__ void gpu_chunked_hsswwc_radix_partition_v4(
    RadixPartitionArgs &args, uint32_t shared_mem_bytes) {
  extern __shared__ uint32_t shared_mem[];

  const uint32_t fanout = 1U << args.radix_bits;
  const uint64_t mask = static_cast<uint64_t>(fanout - 1) << args.ignore_bits;
  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;
  constexpr uint32_t warp_mask = 0xFFFFFFFFu;
  constexpr size_t input_align_mask =
      ~(static_cast<size_t>(ALIGN_BYTES / sizeof(K)) - 1ULL);
  constexpr uint32_t align_tuples = ALIGN_BYTES / sizeof(Tuple<K, V>);

  // Calculate the data_length per block
  size_t data_length =
      ((args.data_length + gridDim.x - 1U) / gridDim.x) & input_align_mask;
  size_t data_offset = data_length * blockIdx.x;
  if (blockIdx.x + 1U == gridDim.x) {
    data_length = args.data_length - data_offset;
  }

  // Substract the alignment padding to avoid an unsigned integer underflow
  // when computing the aligned array offsets.
  unsigned long long partitioned_relation_offset =
      args.partition_offsets[blockIdx.x * fanout] -
      ALIGN_BYTES / sizeof(Tuple<K, V>);

  auto join_attr_data =
      reinterpret_cast<const K *>(args.join_attr_data) + data_offset;
  auto payload_attr_data =
      reinterpret_cast<const V *>(args.payload_attr_data) + data_offset;
  // Handle relations larger than 32 GiB that cause integer overflow.  Subtract
  // chunk offset so that relative value is within range of unsigned int.
  auto partitioned_relation =
      reinterpret_cast<Tuple<K, V> *>(args.partitioned_relation) +
      partitioned_relation_offset;
  auto dmem_buffers = reinterpret_cast<Tuple<K, V> *>(
      args.device_memory_buffers +
      args.device_memory_buffer_bytes * blockIdx.x);

  assert(((size_t)join_attr_data) % (ALIGN_BYTES / sizeof(K)) == 0U &&
         "Key column should be aligned to ALIGN_BYTES for best performance");
  assert(
      ((size_t)payload_attr_data) % (ALIGN_BYTES / sizeof(V)) == 0U &&
      "Payload column should be aligned to ALIGN_BYTES for best performance");
  assert(((size_t)dmem_buffers) % align_tuples == 0U &&
         "DMem buffers should be aligned to ALIGN_BYTES for best performance");

  const uint32_t sswwc_buffer_bytes = shared_mem_bytes -
                                      1 * fanout * sizeof(uint16_t) -
                                      3 * fanout * sizeof(uint32_t);
  const uint32_t tuples_per_dmem_buffer =
      1U << log2_floor_power_of_two(args.device_memory_buffer_bytes /
                                    sizeof(Tuple<K, V>) / (fanout + num_warps));
  uint32_t tuples_per_buffer =
      1U << log2_floor_power_of_two(sswwc_buffer_bytes / sizeof(Tuple<K, V>) /
                                    fanout);
  tuples_per_buffer = tuples_per_buffer < tuples_per_dmem_buffer
                          ? tuples_per_buffer
                          : tuples_per_dmem_buffer;
  const uint32_t slots_per_dmem_buffer =
      tuples_per_dmem_buffer / tuples_per_buffer;

  assert(tuples_per_buffer > 0 &&
         "At least one tuple per partition must fit into SWWC buffer");
  assert(tuples_per_dmem_buffer % tuples_per_buffer == 0 &&
         "DMem buffer size must be a multiple of SMem buffer size");

  Tuple<K, V> *const buffers = reinterpret_cast<Tuple<K, V> *>(shared_mem);
  unsigned int *const tmp_partition_offsets =
      reinterpret_cast<unsigned int *>(&buffers[fanout * tuples_per_buffer]);
  unsigned int *const slots =
      reinterpret_cast<unsigned int *>(&tmp_partition_offsets[fanout]);
  uint32_t *const signal_slots = reinterpret_cast<uint32_t *>(&slots[fanout]);
  unsigned short int *const dmem_buffer_map =
      reinterpret_cast<unsigned short int *>(&signal_slots[fanout]);
  unsigned short int *const dmem_slots =
      reinterpret_cast<unsigned short int *>(&dmem_buffer_map[fanout]);

  // Alignment offset for the initial slots
  const uintptr_t offset_align_mask =
      ~static_cast<uintptr_t>(ALIGN_BYTES - 1UL);

  // Zero shared memory slots and initialize dmem buffer map.
  for (uint32_t i = threadIdx.x; i < fanout; i += blockDim.x) {
    // Align the initial slots to cache lines
    auto offset = static_cast<unsigned int>(
        args.partition_offsets[blockIdx.x * fanout + i] -
        partitioned_relation_offset);
    Tuple<K, V> *base_ptr = reinterpret_cast<Tuple<K, V> *>(
        reinterpret_cast<uintptr_t>(&partitioned_relation[offset]) &
        offset_align_mask);
    uint32_t aligned_fill_state = &partitioned_relation[offset] - base_ptr;
    aligned_fill_state = aligned_fill_state % tuples_per_dmem_buffer;
    tmp_partition_offsets[i] = offset - aligned_fill_state;

    uint32_t smem_fill_state = aligned_fill_state % tuples_per_buffer;
    uint32_t dmem_fill_state = aligned_fill_state / tuples_per_buffer;

    slots[i] = smem_fill_state;
    dmem_slots[i] = dmem_fill_state;
    signal_slots[i] = smem_fill_state;
    dmem_buffer_map[i] = i;
  }

  // Initialize the buffers with NULL so that we don't write out uninitialized
  // data
  for (uint32_t i = threadIdx.x; i < fanout * tuples_per_buffer;
       i += blockDim.x) {
    buffers[i] = {null_key<K>(), {}};
  }

  __syncthreads();

  // Partition data
  //
  // Assign initial spare dmem buffers
  uint32_t spare_dmem_buffer;
  if (lane_id == 0) {
    spare_dmem_buffer = fanout + warp_id;
  }

  // All threads in warp must participate in each loop iteration
  size_t loop_length = (data_length / warpSize) * warpSize;
  for (size_t i = threadIdx.x; i < loop_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    uint32_t p_index = key_to_partition(tuple.key, mask, args.ignore_bits);
    uint32_t pos = 0;
    bool done = false;
    do {
      // Fetch a position if don't have a valid position yet
      if (not done) {
        pos = atomicAdd(&slots[p_index], 1ULL);
        /* pos = atomicInc((unsigned int *)&slots[p_index], fill-in-the-max);
         */

        if (pos < tuples_per_buffer) {
          buffers[write_combine_slot(tuples_per_buffer, p_index, pos)] = tuple;

          // Wait until tuple write is flushed.
          __threadfence_block();

          // Signal that we are done writing the tuple.
          atomicAdd(&signal_slots[p_index], 1U);
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
      if ((ballot = __ballot_sync(warp_mask, is_candidate))) {
        int leader_id = __ffs(ballot) - 1;

        // Release the lock if not the leader and try again in next round.
        // Releasing happens by resetting the smem slot to tuples_per_buffer,
        // but we also have to set the dmem_slot field in the upper bits
        // appropriately. The short-cut is to reuse our slot, because it already
        // contains the corrent smem and dmem values.
        if (is_candidate && lane_id != leader_id) {
          atomicExch(&slots[p_index], pos);
        }

        // Finish unlocking slots before entering busy-wait.
        __syncwarp();

        // Wait until all threads are done writing their tuples into the buffer.
        // Then reset the signal slot to zero.
        if (leader_id == lane_id) {
          while (atomicOr(&signal_slots[p_index], 0U) != tuples_per_buffer)
            ;
          atomicExch(&signal_slots[p_index], 0U);
        }
        __syncwarp();

        uint32_t current_index = __shfl_sync(warp_mask, p_index, leader_id);
        unsigned short int dmem_slot = dmem_slots[current_index];

        // Look up the dmem_buffer for current_index in the map.
        unsigned short int current_dmem_buffer = dmem_buffer_map[current_index];

        // Flush smem buffers to dmem buffers.
        for (uint32_t i = lane_id; i < tuples_per_buffer; i += warpSize) {
          Tuple<K, V> tmp =
              buffers[write_combine_slot(tuples_per_buffer, current_index, i)];
          tmp.store(dmem_buffers[write_combine_slot(
              tuples_per_dmem_buffer, current_dmem_buffer,
              write_combine_slot(tuples_per_buffer, dmem_slot, i))]);
        }

        // Update the dmem slot.
        dmem_slot += 1U;

        // Wait until warp is finished flushing the smem buffer.
        __syncwarp();

        // Swap the dmem_buffer for an empty one if we need to flush it.
        // The swap must occur before the smem lock is released.
        uint32_t dst;
        bool do_dmem_flush = (dmem_slot == slots_per_dmem_buffer);
        if (do_dmem_flush) {
          dmem_slot = 0;
          if (lane_id == 0) {
            dmem_buffer_map[current_index] = spare_dmem_buffer;
            dst = tmp_partition_offsets[current_index];
            tmp_partition_offsets[current_index] += tuples_per_dmem_buffer;
          }

          // Ensure that smem flush and buffer swap occur before the smem lock
          // is released.
          __syncwarp();
        }

        // Release the smem_lock.
        if (lane_id == leader_id) {
          dmem_slots[current_index] = dmem_slot;

          // Ensure that the dmem_slot is updated before releasing the smem
          // lock.
          __threadfence_block();

          // Normal write is not visible to other threads
          //   Use atomic function instead of:
          //   slots[p_indexj] = slot;
          atomicExch(&slots[current_index], 0U);
        }

        // Flush dmem buffer to memory if necessary.
        if (do_dmem_flush) {
          dst = __shfl_sync(warp_mask, dst, 0);
          for (uint32_t i = lane_id; i < tuples_per_dmem_buffer;
               i += warpSize) {
            Tuple<K, V> tmp;
            tmp.load(dmem_buffers[write_combine_slot(tuples_per_dmem_buffer,
                                                     current_dmem_buffer, i)]);
            tmp.store(partitioned_relation[dst + i]);
          }

          // Memorize the spare buffer for the future.
          if (lane_id == 0) {
            spare_dmem_buffer = current_dmem_buffer;
          }
        }
      }
    } while (__any_sync(warp_mask, not done));
  }

  // Wait until all warps are done
  __syncthreads();

  partitioned_relation =
      reinterpret_cast<Tuple<K, V> *>(args.partitioned_relation) +
      partitioned_relation_offset;

  // Flush buffers
  //
  // If no dmem buffer has been flushed yet, flushing the current dmem buffer
  // might overwrite the results of the previous partition (i.e., p_index - 1)
  // due to the cacheline alignment.
  //
  // Thus, if no dmem buffer has yet been flushed, we now flush only the filled
  // dmem slots, and _do not_ flush any empty dmem slots.
  for (uint32_t p_index = warp_id; p_index < fanout; p_index += num_warps) {
    // Check how many dmem buffers have been flushed by computing the total
    // number of alreadly processed dmem slots.
    auto offset = static_cast<unsigned int>(
        args.partition_offsets[blockIdx.x * fanout + p_index] -
        partitioned_relation_offset);
    Tuple<K, V> *base_ptr = reinterpret_cast<Tuple<K, V> *>(
        reinterpret_cast<uintptr_t>(&partitioned_relation[offset]) &
        offset_align_mask);

    uint32_t aligned_fill_state = &partitioned_relation[offset] - base_ptr;
    aligned_fill_state = aligned_fill_state % tuples_per_dmem_buffer;

    auto dst = tmp_partition_offsets[p_index];
    uint32_t tuples_processed = &partitioned_relation[dst] - base_ptr;
    uint32_t smem_buffers_processed = tuples_processed / tuples_per_buffer;

    // If we're still in the first dmem buffer, then skip the empty "alignment"
    // slots. Else, flush the whole dmem buffer.
    uint32_t dmem_start_state = (smem_buffers_processed < slots_per_dmem_buffer)
                                    ? aligned_fill_state
                                    : 0;

    auto dmem_slot = dmem_slots[p_index];
    uint32_t smem_fill_state = signal_slots[p_index];
    uint32_t dmem_fill_state = dmem_slot * tuples_per_buffer + smem_fill_state;
    auto current_dmem_buffer = dmem_buffer_map[p_index];

    if (lane_id == 0) {
      tmp_partition_offsets[p_index] += dmem_fill_state;
    }

    // Flush the smem buffer into the dmem buffer, so that the empty "alignment"
    // smem slots are flushed to CPU memory in the correct order.
    for (uint32_t i = lane_id; i < smem_fill_state; i += warpSize) {
      Tuple<K, V> tmp =
          buffers[write_combine_slot(tuples_per_buffer, p_index, i)];
      tmp.store(dmem_buffers[write_combine_slot(
          tuples_per_dmem_buffer, current_dmem_buffer,
          write_combine_slot(tuples_per_buffer, dmem_slot, i))]);
    }

    __syncwarp();

    // Flush dmem buffers to memory.
    for (uint32_t i = dmem_start_state + lane_id; i < dmem_fill_state;
         i += warpSize) {
      Tuple<K, V> tmp;
      tmp.load(dmem_buffers[write_combine_slot(tuples_per_dmem_buffer,
                                               current_dmem_buffer, i)]);
      tmp.store(partitioned_relation[dst + i]);
    }
  }

  __syncthreads();

  // Handle case when data_length % warpSize != 0
  for (size_t i = loop_length + threadIdx.x; i < data_length; i += blockDim.x) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    auto p_index = key_to_partition(tuple.key, mask, args.ignore_bits);
    auto offset = atomicAdd(&tmp_partition_offsets[p_index], 1U);
    partitioned_relation[offset] = tuple;
  }
}

// Exports the histogram function for 8-byte keys.
extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_chunked_prefix_sum_int32(PrefixSumArgs args) {
  gpu_chunked_prefix_sum<int>(args);
}

// Exports the histogram function for 16-byte keys.
extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_chunked_prefix_sum_int64(PrefixSumArgs args) {
  gpu_chunked_prefix_sum<long long>(args);
}

// Exports the histogram function for 8-byte keys.
extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_contiguous_prefix_sum_int32(PrefixSumArgs args) {
  gpu_contiguous_prefix_sum<int>(args);
}

// Exports the histogram function for 16-byte keys.
extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_contiguous_prefix_sum_int64(PrefixSumArgs args) {
  gpu_contiguous_prefix_sum<long long>(args);
}

// Exports the histogram function for 8-byte key/value tuples.
extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_contiguous_prefix_sum_and_copy_with_payload_int32_int32(
        PrefixSumAndCopyWithPayloadArgs args) {
  gpu_contiguous_prefix_sum_and_copy_with_payload<int, int>(args);
}

// Exports the histogram function for 16-byte key/value tuples.
extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_contiguous_prefix_sum_and_copy_with_payload_int64_int64(
        PrefixSumAndCopyWithPayloadArgs args) {
  gpu_contiguous_prefix_sum_and_copy_with_payload<long long, long long>(args);
}

// Exports the histogram function for 8-byte key/value tuples.
extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_contiguous_prefix_sum_and_transform_int32_int32(
        PrefixSumAndTransformArgs args) {
  gpu_contiguous_prefix_sum_and_transform<int, int>(args);
}

// Exports the histogram function for 16-byte key/value tuples.
extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_contiguous_prefix_sum_and_transform_int64_int64(
        PrefixSumAndTransformArgs args) {
  gpu_contiguous_prefix_sum_and_transform<long long, long long>(args);
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_chunked_radix_partition_int32_int32(RadixPartitionArgs args) {
  gpu_chunked_radix_partition<int, int>(args);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_chunked_radix_partition_int64_int64(RadixPartitionArgs args) {
  gpu_chunked_radix_partition<long long, long long>(args);
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_laswwc_radix_partition_int32_int32(
        RadixPartitionArgs args, uint32_t shared_mem_bytes) {
  gpu_chunked_laswwc_radix_partition<int, int>(args, shared_mem_bytes);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_laswwc_radix_partition_int64_int64(
        RadixPartitionArgs args, uint32_t shared_mem_bytes) {
  gpu_chunked_laswwc_radix_partition<long long, long long>(args,
                                                           shared_mem_bytes);
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_sswwc_radix_partition_v2_int32_int32(
        RadixPartitionArgs args, uint32_t shared_mem_bytes) {
  gpu_chunked_sswwc_radix_partition_v2<int, int>(args, shared_mem_bytes);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_sswwc_radix_partition_v2_int64_int64(
        RadixPartitionArgs args, uint32_t shared_mem_bytes) {
  gpu_chunked_sswwc_radix_partition_v2<long long, long long>(args,
                                                             shared_mem_bytes);
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_sswwc_radix_partition_v2g_int32_int32(
        RadixPartitionArgs args) {
  gpu_chunked_sswwc_radix_partition_v2g<int, int>(args);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_sswwc_radix_partition_v2g_int64_int64(
        RadixPartitionArgs args) {
  gpu_chunked_sswwc_radix_partition_v2g<long long, long long>(args);
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" __launch_bounds__(512, 1) __global__
    void gpu_chunked_hsswwc_radix_partition_v4_int32_int32(
        RadixPartitionArgs args, uint32_t shared_mem_bytes) {
  gpu_chunked_hsswwc_radix_partition_v4<int, int>(args, shared_mem_bytes);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" __launch_bounds__(512, 1) __global__
    void gpu_chunked_hsswwc_radix_partition_v4_int64_int64(
        RadixPartitionArgs args, uint32_t shared_mem_bytes) {
  gpu_chunked_hsswwc_radix_partition_v4<long long, long long>(args,
                                                              shared_mem_bytes);
}
