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

#ifndef GPU_RADIX_PARTITION_H
#define GPU_RADIX_PARTITION_H

#include <cstdint>

#include "prefix_scan_state.h"

#ifndef TUPLES_PER_THREAD
#define TUPLES_PER_THREAD 5U
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
  uint32_t const padding_length;
  uint32_t const radix_bits;

  // State
  ScanState<unsigned long long> *const prefix_scan_state;
  unsigned long long *const __restrict__ tmp_partition_offsets;

  // Outputs
  // unsigned long long *const __restrict__ sorted_partition_offsets;
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
  const unsigned long long *const __restrict__ partition_offsets;

  // State
  char *const __restrict__ device_memory_buffers;
  uint64_t const device_memory_buffer_bytes;

  // Outputs
  void *const __restrict__ partitioned_relation;
};

// Computes the partition ID of a given key.
template <typename T, typename B>
CUDA_MODIFIER uint32_t key_to_partition(T key, uint64_t mask, B bits) {
  return static_cast<uint32_t>((static_cast<uint64_t>(key) & mask) >> bits);
}

#endif /* GPU_RADIX_PARTITION_H */
