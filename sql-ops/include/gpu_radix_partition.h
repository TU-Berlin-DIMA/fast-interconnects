/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019-2021 Clemens Lutz, German Research Center for Artificial
 * Intelligence
 * Author: Clemens Lutz, DFKI GmbH <clemens.lutz@dfki.de>
 */

#ifndef GPU_RADIX_PARTITION_H
#define GPU_RADIX_PARTITION_H

#include <constants.h>
#include "prefix_scan_state.h"

#include <cstdint>

#ifndef LASWWC_TUPLES_PER_THREAD
#define LASWWC_TUPLES_PER_THREAD 5U
#endif

#ifndef ALIGN_BYTES
#define ALIGN_BYTES 128U
#endif

#ifndef CUDA_MODIFIER
#define CUDA_MODIFIER __device__
#endif

#define __UINT_MAX__ static_cast<unsigned int>(__INT_MAX__ * 2U + 1U)

struct PrefixSumArgs {
  // Inputs
  const void *const __restrict__ partition_attr;
  std::size_t const data_length;
  std::size_t const canonical_chunk_length;
  uint32_t const padding_length;
  uint32_t const radix_bits;
  uint32_t const ignore_bits;

  // State
  ScanState<unsigned long long> *const prefix_scan_state;
  unsigned long long *const __restrict__ tmp_partition_offsets;

  // Outputs
  // unsigned long long *const __restrict__ sorted_partition_offsets;
  unsigned long long *const __restrict__ partition_offsets;
};

struct PrefixSumAndCopyWithPayloadArgs {
  // Inputs
  const void *const __restrict__ src_partition_attr;
  const void *const __restrict__ src_payload_attr;
  std::size_t const data_length;
  std::size_t const canonical_chunk_length;
  uint32_t const padding_length;
  uint32_t const radix_bits;
  uint32_t const ignore_bits;

  // State
  ScanState<unsigned long long> *const prefix_scan_state;
  unsigned long long *const __restrict__ tmp_partition_offsets;

  // Outputs
  void *const __restrict__ dst_partition_attr;
  void *const __restrict__ dst_payload_attr;
  unsigned long long *const __restrict__ partition_offsets;
};

struct PrefixSumAndTransformArgs {
  // Inputs
  uint32_t partition_id;
  const void *const __restrict__ src_relation;
  const unsigned long long *const __restrict__ src_offsets;
  uint32_t src_chunks;
  uint32_t const src_radix_bits;
  std::size_t const data_length;
  uint32_t const padding_length;
  uint32_t const radix_bits;
  uint32_t const ignore_bits;

  // State
  ScanState<unsigned long long> *const prefix_scan_state;
  unsigned long long *const __restrict__ tmp_partition_offsets;

  // Outputs
  void *const __restrict__ dst_partition_attr;
  void *const __restrict__ dst_payload_attr;
  unsigned long long *const __restrict__ partition_offsets;
};

// Arguments to the partitioning function.
//
// Note that the struct's layout must be kept in sync with its counterpart in
// Rust.
struct RadixPartitionArgs {
  // Inputs
  const void *const __restrict__ join_attr_data;
  const void *const __restrict__ payload_attr_data;
  std::size_t const data_length;
  uint32_t const padding_length;
  uint32_t const radix_bits;
  uint32_t const ignore_bits;
  const unsigned long long *const __restrict__ partition_offsets;

  // State
  uint32_t *const __restrict__ tmp_partition_offsets;
  char *const __restrict__ l2_cache_buffers;
  char *const __restrict__ device_memory_buffers;
  uint64_t const device_memory_buffer_bytes;

  // Outputs
  void *const __restrict__ partitioned_relation;
};

// Computes the partition ID of a given key.
template <typename T, typename B>
CUDA_MODIFIER unsigned int key_to_partition(T key, unsigned long long mask,
                                            B bits) {
  return static_cast<unsigned int>(
      (static_cast<unsigned long long>(key) & mask) >> bits);
}

#endif /* GPU_RADIX_PARTITION_H */
