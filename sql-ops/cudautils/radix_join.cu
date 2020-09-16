/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2020 Clemens Lutz, German Research Center for Artificial
 * Intelligence
 * Author: Clemens Lutz, DFKI GmbH <clemens.lutz@dfki.de>
 */

#define CUDA_MODIFIER __device__
// #define DEBUG

#include <gpu_common.h>
#include <gpu_radix_partition.h>

#include <cassert>
#include <cstdint>

#ifdef DEBUG
#include <cstdio>
#endif

using namespace std;

// Arguments to the join-aggregate function.
//
// Note that the struct's layout must be kept in sync with its counterpart in
// Rust.
struct JoinAggregateArgs {
  void const *const build_rel;
  uint64_t const *const build_rel_partition_offsets;
  void const *const probe_rel;
  uint64_t const *const probe_rel_partition_offsets;
  uint64_t *const aggregation_result;
  uint32_t *const task_assignments;
  uint32_t const build_rel_length;
  uint32_t const probe_rel_length;
  uint32_t const build_rel_padding_length;
  uint32_t const probe_rel_padding_length;
  uint32_t const radix_bits;
  uint32_t const ht_entries;
};

// Assign tasks to thread blocks
//
// Each task is a contiguous sequence of partitions. This assumes that there
// are more partitions than thread blocks. If the assumption does not hold,
// then some thread blocks are assigned an empty task (i.e.,
// task_assignments[i] == task_assignments[i+1]). The last array index is a
// sentinal value that is set to the fanout.
extern "C" __global__ void gpu_radix_join_assign_tasks(JoinAggregateArgs args) {
  // FIXME: parallelize task assignment
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const uint32_t fanout = 1U << args.radix_bits;
    const uint32_t probe_tuples =
        args.probe_rel_length - args.probe_rel_padding_length * fanout;
    const uint32_t avg_task_size = (probe_tuples + gridDim.x - 1U) / gridDim.x;

    args.task_assignments[0] = 0U;
    uint32_t task_id = 1U;
    uint32_t task_size = 0U;
    for (uint32_t p = 0U; p < fanout && task_id < gridDim.x; p += 1U) {
      uint32_t probe_upper = (p + 1 < fanout)
                                 ? args.probe_rel_partition_offsets[p + 1U] -
                                       args.probe_rel_padding_length
                                 : args.probe_rel_length;
      uint32_t probe_size = static_cast<uint32_t>(
          probe_upper - args.probe_rel_partition_offsets[p]);

      task_size += probe_size;
      if (task_size >= avg_task_size) {
        args.task_assignments[task_id] = p + 1U;

#ifdef DEBUG
        printf("Assigning partitions [%u, %u] to block %d\n",
               args.task_assignments[task_id - 1],
               args.task_assignments[task_id], task_id);
#endif

        task_size = 0U;
        task_id += 1;
      }
    }

    // assign an empty task if fanout < gridDim.x
    // and initialize sentinal value at task_assignments[gridDim.x]
    for (uint32_t tid = task_id; tid <= gridDim.x; tid += 1U) {
      args.task_assignments[tid] = fanout;
    }
  }
}

// FIXME: handle chunking in the 2nd partitioning pass
template <typename K, typename PI, typename PO>
__device__ void gpu_radix_join_aggregate_smem_perfect(JoinAggregateArgs &args) {
  extern __shared__ uint32_t shared_mem[];

  const uint32_t fanout = 1U << args.radix_bits;
  const uint64_t mask = ~static_cast<uint64_t>(fanout - 1U);

  HtEntry<K, PI> *const __restrict__ hash_table =
      reinterpret_cast<HtEntry<K, PI> *>(shared_mem);

  uint64_t sum = 0;

  for (uint32_t p = args.task_assignments[blockIdx.x];
       p < args.task_assignments[blockIdx.x + 1U]; p += 1U) {
    Tuple<K, PI> const *const __restrict__ build_rel =
        reinterpret_cast<Tuple<K, PI> const *>(args.build_rel) +
        args.build_rel_partition_offsets[p];
    Tuple<K, PO> const *const __restrict__ probe_rel =
        reinterpret_cast<Tuple<K, PO> const *>(args.probe_rel) +
        args.probe_rel_partition_offsets[p];

    uint32_t build_upper = (p + 1U < fanout)
                               ? args.build_rel_partition_offsets[p + 1U] -
                                     args.build_rel_padding_length
                               : args.build_rel_length;
    uint32_t build_size = static_cast<uint32_t>(
        build_upper - args.build_rel_partition_offsets[p]);

    assert(build_size <= args.ht_entries &&
           "Build-side relation is larger than hash table");

    uint32_t probe_upper = (p + 1U < fanout)
                               ? args.probe_rel_partition_offsets[p + 1U] -
                                     args.probe_rel_padding_length
                               : args.probe_rel_length;
    uint32_t probe_size = static_cast<uint32_t>(
        probe_upper - args.probe_rel_partition_offsets[p]);

#ifdef DEBUG
    if (threadIdx.x == 0) {
      printf("part: %d, fanout: %d, build_size: %d, probe_size: %d\n", p,
             fanout, build_size, probe_size);
    }
#endif

    // Initialize hash table
    for (uint32_t i = threadIdx.x; i < args.ht_entries; i += blockDim.x) {
      hash_table[i] = {null_key<K>(), 0};
    }

    __syncthreads();

    // Build
    for (uint32_t i = threadIdx.x; i < build_size; i += blockDim.x) {
      Tuple<K, PI> tuple = build_rel[i];
      auto ht_index = key_to_partition(tuple.key, mask, args.radix_bits);

#ifdef DEBUG
      assert(ht_index < args.ht_entries && "Invalid hash table index");
#endif

      hash_table[ht_index] = {tuple.key, tuple.value};
    }

    __syncthreads();

    // Probe
    for (uint32_t i = threadIdx.x; i < probe_size; i += blockDim.x) {
      Tuple<K, PO> tuple = probe_rel[i];
      auto ht_index = key_to_partition(tuple.key, mask, args.radix_bits);

#ifdef DEBUG
      assert(ht_index < args.ht_entries && "Invalid hash table index");
#endif

      if (hash_table[ht_index].key == tuple.key) {
        sum += tuple.value;
      }
#ifdef DEBUG
      else {
        printf(
            "tid: %u, part: %u, ht_index: %u, build_key: %u, probe_key: %u\n",
            threadIdx.x, p, ht_index, hash_table[ht_index].key, tuple.key);
      }
#endif
    }

    __syncthreads();
  }

  args.aggregation_result[blockDim.x * blockIdx.x + threadIdx.x] = sum;
}

extern "C" __global__ void gpu_join_aggregate_smem_perfect_i32_i32_i32(
    JoinAggregateArgs args) {
  gpu_radix_join_aggregate_smem_perfect<int, int, int>(args);
}

extern "C" __global__ void gpu_join_aggregate_smem_perfect_i64_i64_i64(
    JoinAggregateArgs args) {
  gpu_radix_join_aggregate_smem_perfect<long long, long long, long long>(args);
}
