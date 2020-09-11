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

#define CUDA_MODIFIER

#include <gpu_radix_partition.h>

#include <cstdint>
#include <vector>

using namespace std;

// Chunked histogram and offset computation
//
// Used for GPU radix partitioning.
template <typename K>
void cpu_chunked_prefix_sum(PrefixSumArgs &args, uint32_t const chunk_id,
                            uint32_t const /* num_chunks */) {
  const size_t fanout = 1UL << args.radix_bits;
  const size_t mask = fanout - 1;

  vector<unsigned int> tmp_partition_offsets(fanout);

  size_t partitioned_data_offset =
      (args.canonical_chunk_length + args.padding_length * fanout) * chunk_id;
  auto partition_attr =
      static_cast<const K *const __restrict__>(args.partition_attr);

  // Ensure counters are all zeroed
  for (size_t i = 0; i < fanout; ++i) {
    tmp_partition_offsets[i] = 0;
  }

  // Compute local histograms per partition for chunk
  for (size_t i = 0; i < args.data_length; ++i) {
    auto key = partition_attr[i];
    auto p_index = key_to_partition(key, mask, 0);
    tmp_partition_offsets[p_index] += 1;
  }

  // Compute offsets with exclusive prefix sum
  uint64_t offset = partitioned_data_offset + args.padding_length;
  for (uint32_t i = 0; i < fanout; ++i) {
    // Add data offset onto partitions offsets and write out the final offsets
    // to device memory.
    args.partition_offsets[i] = offset;

    // Update offset
    offset += static_cast<uint64_t>(tmp_partition_offsets[i]);
    offset += args.padding_length;
  }
}

// Exports the histogram function for 4-byte keys.
extern "C" void cpu_chunked_prefix_sum_int32(PrefixSumArgs *const args,
                                             uint32_t const chunk_id,
                                             uint32_t const num_chunks) {
  cpu_chunked_prefix_sum<int>(*args, chunk_id, num_chunks);
}

// Exports the histogram function for 8-byte keys.
extern "C" void cpu_chunked_prefix_sum_int64(PrefixSumArgs *const args,
                                             uint32_t const chunk_id,
                                             uint32_t const num_chunks) {
  cpu_chunked_prefix_sum<long long>(*args, chunk_id, num_chunks);
}
