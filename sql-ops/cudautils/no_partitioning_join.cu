/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 *
 * Note that parts of this code are based on the Hawk query compiler by
 * Sebastian Breß et al.
 * See https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ for details.
 */

/*
 * Assumptions:
 *
 * Hash table's size is 2^x - 1
 * Key with value==-1 is reserved for NULL values
 * Key and payload are int64_t
 * Hash table is initialized with all entries set to -1
 */

#include <gpu_common.h>

#define OPTIMIZE_INT32
/* #define PREDICATED_AGGREGATION */

/*
 * Note: uint64_t in cstdint header doesn't match atomicCAS()
 */
typedef unsigned int uint32_t;
typedef unsigned long long int uint64_t;

__device__ void gpu_ht_insert_linearprobing_int32(
    HtEntry<int, int> *const __restrict__ hash_table,
    unsigned int log2_hash_table_entries, int key, int payload) {
  uint64_t index = hash<int>(key, log2_hash_table_entries);

  uint64_t hash_table_entries = 1ULL << log2_hash_table_entries;
  uint64_t hash_table_mask = hash_table_entries - 1ULL;

  for (uint64_t i = 0; i < hash_table_entries;
       ++i, index = (index + 1ULL) & hash_table_mask) {
#ifdef OPTIMIZE_INT32
    uint64_t null_key_64 = static_cast<uint64_t>(null_key<long long>());
    // Negative int32_t is promoted when cast to uint64_t,
    // need to drop the sign flags added by the cast
    uint64_t entry = (static_cast<uint64_t>(key) & 0x00000000FFFFFFFF) |
                     (static_cast<uint64_t>(payload) << 32);
    uint64_t old_entry = atomicCAS(
        reinterpret_cast<uint64_t *>(&hash_table[index]), null_key_64, entry);
    if (old_entry == static_cast<uint64_t>(null_key<long long>())) {
      return;
    }
#else
    unsigned int null_key_u = static_cast<unsigned int>(null_key<int>());
    int old = static_cast<int>(
        atomicCAS(reinterpret_cast<unsigned int *>(&hash_table[index].key),
                  null_key_u, static_cast<unsigned int>(key)));
    if (old == null_key<int>()) {
      hash_table[index].value = payload;
      return;
    }
#endif
  }
}

__device__ void gpu_ht_insert_linearprobing_int64(
    HtEntry<long long, long long> *const __restrict__ hash_table,
    unsigned int log2_hash_table_entries, long long key, long long payload) {
  uint64_t index =
      static_cast<uint64_t>(hash<long long>(key, log2_hash_table_entries));

  uint64_t hash_table_entries = 1ULL << log2_hash_table_entries;
  uint64_t hash_table_mask = hash_table_entries - 1ULL;

  for (uint64_t i = 0; i < hash_table_entries;
       ++i, index = (index + 1ULL) & hash_table_mask) {
    unsigned long long int null_key_u =
        static_cast<unsigned long long>(null_key<long long>());
    long long old = static_cast<long long>(atomicCAS(
        reinterpret_cast<unsigned long long int *>(&hash_table[index].key),
        null_key_u, static_cast<unsigned long long int>(key)));
    if (old == null_key<long long>()) {
      hash_table[index].value = payload;
      return;
    }
  }
}

extern "C" __global__ void gpu_ht_build_linearprobing_int32(
    HtEntry<int, int> *const __restrict__ hash_table,
    uint64_t const hash_table_entries,
    const int *const __restrict__ join_attr_data,
    const int *const __restrict__ payload_attr_data,
    uint64_t const data_length) {
  const uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_threads = blockDim.x * gridDim.x;
  const unsigned int log2_hash_table_entries =
      log2_floor_power_of_two(hash_table_entries);

  for (uint64_t tuple_id = global_idx; tuple_id < data_length;
       tuple_id += global_threads) {
    gpu_ht_insert_linearprobing_int32(hash_table, log2_hash_table_entries,
                                      join_attr_data[tuple_id],
                                      payload_attr_data[tuple_id]);
  }
}

extern "C" __global__ void gpu_ht_build_linearprobing_int64(
    HtEntry<long long, long long> *const __restrict__ hash_table,
    uint64_t const hash_table_entries,
    const long long *const __restrict__ join_attr_data,
    const long long *const __restrict__ payload_attr_data,
    uint64_t const data_length) {
  const uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_threads = blockDim.x * gridDim.x;
  const unsigned int log2_hash_table_entries =
      log2_floor_power_of_two(hash_table_entries);

  for (uint64_t tuple_id = global_idx; tuple_id < data_length;
       tuple_id += global_threads) {
    gpu_ht_insert_linearprobing_int64(hash_table, log2_hash_table_entries,
                                      join_attr_data[tuple_id],
                                      payload_attr_data[tuple_id]);
  }
}

__device__ bool gpu_ht_findkey_linearprobing_int32(
    const HtEntry<int, int> *const __restrict__ hash_table,
    unsigned int log2_hash_table_entries, int key, int *found_payload,
    uint64_t *__restrict__ last_index, bool use_last_index) {
  uint64_t index = 0;
  if (use_last_index) {
    index = *last_index;
    index += 1ULL;
  } else {
    index = hash<int>(key, log2_hash_table_entries);
  }

  uint64_t hash_table_entries = 1ULL << log2_hash_table_entries;
  uint64_t hash_table_mask = hash_table_entries - 1ULL;

  for (uint64_t i = 0; i < hash_table_mask + 1ULL;
       ++i, index = (index + 1ULL) & hash_table_mask) {
    HtEntry<int, int> entry;
    entry.load(hash_table[index]);

    if (entry.key == key) {
      *found_payload = entry.value;
      *last_index = index;
      return true;
    } else if (entry.key == null_key<int>()) {
      return false;
    }
  }

  return false;
}

__device__ bool gpu_ht_findkey_linearprobing_int64(
    const HtEntry<long long, long long> *const __restrict__ hash_table,
    unsigned int log2_hash_table_entries, long long key,
    long long *found_payload, uint64_t *__restrict__ last_index,
    bool use_last_index) {
  uint64_t index = 0;
  if (use_last_index) {
    index = *last_index;
    index += 1ULL;
  } else {
    index = hash<long long>(key, log2_hash_table_entries);
  }

  uint64_t hash_table_entries = 1ULL << log2_hash_table_entries;
  uint64_t hash_table_mask = hash_table_entries - 1ULL;

  for (uint64_t i = 0; i < hash_table_mask + 1ULL;
       ++i, index = (index + 1ULL) & hash_table_mask) {
    HtEntry<long long, long long> entry;
    entry.load(hash_table[index]);

    if (entry.key == key) {
      *found_payload = entry.value;
      *last_index = index;
      return true;
    } else if (entry.key == null_key<long long>()) {
      return false;
    }
  }

  return false;
}

extern "C" __global__ void gpu_ht_probe_aggregate_linearprobing_int32(
    const HtEntry<int, int> *const __restrict__ hash_table,
    uint64_t const hash_table_entries,
    const int *const __restrict__ join_attr_data,
    const int *const __restrict__ payload_attr_data, uint64_t const data_length,
    uint64_t *__restrict__ aggregation_result) {
  const uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_threads = blockDim.x * gridDim.x;
  const unsigned int log2_hash_table_entries =
      log2_floor_power_of_two(hash_table_entries);

  for (uint64_t tuple_id = global_idx; tuple_id < data_length;
       tuple_id += global_threads) {
    int hash_table_payload = 0;
    uint64_t hash_table_last_index = 0;
    bool hash_table_use_last_index = false;
    while (gpu_ht_findkey_linearprobing_int32(
        hash_table, log2_hash_table_entries, join_attr_data[tuple_id],
        &hash_table_payload, &hash_table_last_index,
        hash_table_use_last_index)) {
      hash_table_use_last_index = true;
      aggregation_result[global_idx] += payload_attr_data[tuple_id];
    }
  }
}

extern "C" __global__ void gpu_ht_probe_aggregate_linearprobing_int64(
    const HtEntry<long long, long long> *const __restrict__ hash_table,
    uint64_t const hash_table_entries,
    const long long *const __restrict__ join_attr_data,
    const long long *const __restrict__ payload_attr_data,
    uint64_t const data_length, uint64_t *__restrict__ aggregation_result) {
  const uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_threads = blockDim.x * gridDim.x;
  const unsigned int log2_hash_table_entries =
      log2_floor_power_of_two(hash_table_entries);

  for (uint64_t tuple_id = global_idx; tuple_id < data_length;
       tuple_id += global_threads) {
    long long hash_table_payload = 0;
    uint64_t hash_table_last_index = 0;
    bool hash_table_use_last_index = false;
    while (gpu_ht_findkey_linearprobing_int64(
        hash_table, log2_hash_table_entries, join_attr_data[tuple_id],
        &hash_table_payload, &hash_table_last_index,
        hash_table_use_last_index)) {
      hash_table_use_last_index = true;
      aggregation_result[global_idx] += payload_attr_data[tuple_id];
    }
  }
}

extern "C" __global__ void gpu_ht_build_perfect_int32(
    HtEntry<int, int> *const __restrict__ hash_table,
    uint64_t const /* hash_table_entries */,
    const int *const __restrict__ join_attribute_data,
    const int *const __restrict__ payload_attributed_data,
    uint64_t const data_length) {
  const uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_threads = blockDim.x * gridDim.x;

  for (uint64_t i = global_idx; i < data_length; i += global_threads) {
    HtEntry<int, int> tuple;
    tuple.key = join_attribute_data[i];
    tuple.value = payload_attributed_data[i];

    tuple.store(hash_table[tuple.key]);
  }
}

extern "C" __global__ void gpu_ht_build_selective_perfect_int32(
    HtEntry<int, int> *const __restrict__ hash_table,
    uint64_t const /* hash_table_entries */,
    const int *const __restrict__ join_attribute_data,
    const int *const __restrict__ payload_attributed_data,
    uint64_t const data_length) {
  const uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_threads = blockDim.x * gridDim.x;

  for (uint64_t i = global_idx; i < data_length; i += global_threads) {
    int key = join_attribute_data[i];
    if (key != null_key<int>()) {
      HtEntry<int, int> tuple;
      tuple.key = key;
      tuple.value = payload_attributed_data[i];

      tuple.store(hash_table[key]);
    }
  }
}

extern "C" __global__ void gpu_ht_build_perfect_int64(
    HtEntry<long long, long long> *const __restrict__ hash_table,
    uint64_t const /* hash_table_entries */,
    const long long *const __restrict__ join_attribute_data,
    const long long *const __restrict__ payload_attributed_data,
    uint64_t const data_length) {
  const uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_threads = blockDim.x * gridDim.x;

  for (uint64_t i = global_idx; i < data_length; i += global_threads) {
    HtEntry<long long, long long> tuple;
    tuple.key = join_attribute_data[i];
    tuple.value = payload_attributed_data[i];

    tuple.store(hash_table[tuple.key]);
  }
}

extern "C" __global__ void gpu_ht_build_selective_perfect_int64(
    HtEntry<long long, long long> *const __restrict__ hash_table,
    uint64_t const /* hash_table_entries */,
    const long long *const __restrict__ join_attribute_data,
    const long long *const __restrict__ payload_attributed_data,
    uint64_t const data_length) {
  const uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_threads = blockDim.x * gridDim.x;

  for (uint64_t i = global_idx; i < data_length; i += global_threads) {
    long long key = join_attribute_data[i];
    if (key != null_key<long long>()) {
      HtEntry<long long, long long> tuple;
      tuple.key = key;
      tuple.value = payload_attributed_data[i];

      tuple.store(hash_table[key]);
    }
  }
}

extern "C" __global__ void gpu_ht_probe_aggregate_perfect_int32(
    const HtEntry<int, int> *const __restrict__ hash_table,
    uint64_t const /* hash_table_entries */,
    const int *const __restrict__ join_attribute_data,
    const int *const __restrict__ payload_attribute_data,
    uint64_t const data_length, uint64_t *__restrict__ aggregation_result) {
  const uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_threads = blockDim.x * gridDim.x;

  for (uint64_t i = global_idx; i < data_length; i += global_threads) {
    int key = join_attribute_data[i];

#ifdef PREDICATED_AGGREGATION
    int condition = hash_table[key].key == key;
    condition = (condition << 31) >> 31;

    int payload = condition & payload_attribute_data[i];
    aggregation_result[global_idx] += static_cast<uint64_t>(payload);
#else
    if (hash_table[key].key == key) {
      aggregation_result[global_idx] += payload_attribute_data[i];
    }
#endif /* PREDICATED_AGGREGATION */
  }
}

extern "C" __global__ void gpu_ht_probe_aggregate_perfect_int64(
    const HtEntry<long long, long long> *const __restrict__ hash_table,
    uint64_t const /* hash_table_entries */,
    const long long *const __restrict__ join_attribute_data,
    const long long *const __restrict__ payload_attribute_data,
    uint64_t const data_length, uint64_t *__restrict__ aggregation_result) {
  const uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_threads = blockDim.x * gridDim.x;

  for (uint64_t i = global_idx; i < data_length; i += global_threads) {
    long long key = join_attribute_data[i];

#ifdef PREDICATED_AGGREGATION
    long long condition = hash_table[key].key == key;
    condition = (condition << 63) >> 63;

    long long payload = condition & payload_attribute_data[i];
    aggregation_result[global_idx] += static_cast<uint64_t>(payload);
#else
    if (hash_table[key].key == key) {
      aggregation_result[global_idx] += payload_attribute_data[i];
    }
#endif /* PREDICATED_AGGREGATION */
  }
}
