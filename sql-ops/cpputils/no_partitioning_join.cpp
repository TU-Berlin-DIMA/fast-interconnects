// Copyright 2018-2022 Clemens Lutz
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

/*
 * Assumptions:
 *
 * Hash table's size is 2^x - 1
 * Key with value==-1 is reserved for NULL values
 * Key and payload are int64_t
 * Hash table is initialized with all entries set to -1
 */

#define CUDA_MODIFIER
#include <gpu_common.h>

#include <atomic>
#include <cstdint>

template <typename T>
void cpu_ht_insert_linearprobing(HtEntry<T, T> *const __restrict__ hash_table,
                                 unsigned int log2_hash_table_entries, T key,
                                 T payload) {
  uint64_t index = hash<T>(key, log2_hash_table_entries);

  uint64_t hash_table_entries = 1ULL << log2_hash_table_entries;
  uint64_t hash_table_mask = hash_table_entries - 1ULL;

  for (uint64_t i = 0; i < hash_table_entries;
       ++i, index = (index + 1ULL) & hash_table_mask) {
    T old = hash_table[index].key;
    if (old == null_key<T>()) {
      T expected = null_key<T>();
      bool is_inserted = std::atomic_compare_exchange_strong(
          (std::atomic<T> *)&hash_table[index].key, &expected, key);
      if (is_inserted) {
        hash_table[index].value = payload;
        return;
      }
    }
  }
}

// extern "C"
template <typename T>
void cpu_ht_build_linearprobing(HtEntry<T, T> *const __restrict__ hash_table,
                                uint64_t const hash_table_entries,
                                const T *const __restrict__ join_attr_data,
                                const T *const __restrict__ payload_attr_data,
                                uint64_t const data_length) {
  const unsigned int log2_hash_table_entries =
      log2_floor_power_of_two(hash_table_entries);

  for (uint64_t tuple_id = 0; tuple_id < data_length; ++tuple_id) {
    cpu_ht_insert_linearprobing(hash_table, log2_hash_table_entries,
                                join_attr_data[tuple_id],
                                payload_attr_data[tuple_id]);
  }
}

extern "C" void cpu_ht_build_linearprobing_int32(
    HtEntry<int, int> *const __restrict__ hash_table,
    uint64_t const hash_table_entries,
    const int *const __restrict__ join_attr_data,
    const int *const __restrict__ payload_attr_data,
    uint64_t const data_length) {
  cpu_ht_build_linearprobing(hash_table, hash_table_entries, join_attr_data,
                             payload_attr_data, data_length);
}

extern "C" void cpu_ht_build_linearprobing_int64(
    HtEntry<long long, long long> *const __restrict__ hash_table,
    uint64_t const hash_table_entries,
    const long long *const __restrict__ join_attr_data,
    const long long *const __restrict__ payload_attr_data,
    uint64_t const data_length) {
  cpu_ht_build_linearprobing(hash_table, hash_table_entries, join_attr_data,
                             payload_attr_data, data_length);
}

template <typename T>
bool cpu_ht_findkey_linearprobing(
    HtEntry<T, T> const *const __restrict__ hash_table,
    unsigned int log2_hash_table_entries, T key, T const **found_payload,
    uint64_t *__restrict__ last_index, bool use_last_index) {
  uint64_t hash_table_entries = 1ULL << log2_hash_table_entries;
  uint64_t hash_table_mask = hash_table_entries - 1ULL;

  uint64_t index = 0;
  if (use_last_index) {
    index = *last_index;
    index = (index + 1ULL) & hash_table_mask;
  } else {
    index = hash<T>(key, log2_hash_table_entries);
  }

  for (uint64_t i = 0; i < hash_table_mask + 1ULL;
       ++i, index = (index + 1ULL) & hash_table_mask) {
    if (hash_table[index].key == key) {
      *found_payload = &hash_table[index].value;
      *last_index = index;
      return true;
    } else if (hash_table[index].key == null_key<T>()) {
      return false;
    }
  }

  return false;
}

template <typename T>
void cpu_ht_probe_aggregate_linearprobing(
    HtEntry<T, T> const *const __restrict__ hash_table,
    uint64_t const hash_table_entries,
    const T *const __restrict__ join_attr_data,
    const T *const __restrict__ payload_attr_data, uint64_t const data_length,
    uint64_t *const __restrict__ aggregation_result) {
  const unsigned int log2_hash_table_entries =
      log2_floor_power_of_two(hash_table_entries);

  for (uint64_t tuple_id = 0; tuple_id < data_length; ++tuple_id) {
    T const *hash_table_payload = nullptr;
    uint64_t hash_table_last_index = 0;
    bool hash_table_use_last_index = false;
    while (cpu_ht_findkey_linearprobing(
        hash_table, log2_hash_table_entries, join_attr_data[tuple_id],
        &hash_table_payload, &hash_table_last_index,
        hash_table_use_last_index)) {
      hash_table_use_last_index = true;
      *aggregation_result += payload_attr_data[tuple_id];
    }
  }
}

extern "C" void cpu_ht_probe_aggregate_linearprobing_int32(
    HtEntry<int, int> const *const __restrict__ hash_table,
    uint64_t const hash_table_entries,
    const int *const __restrict__ join_attr_data,
    const int *const __restrict__ payload_attr_data, uint64_t const data_length,
    uint64_t *const __restrict__ aggregation_result) {
  cpu_ht_probe_aggregate_linearprobing(hash_table, hash_table_entries,
                                       join_attr_data, payload_attr_data,
                                       data_length, aggregation_result);
}

extern "C" void cpu_ht_probe_aggregate_linearprobing_int64(
    HtEntry<long long, long long> const *const __restrict__ hash_table,
    uint64_t const hash_table_entries,
    const long long *const __restrict__ join_attr_data,
    const long long *const __restrict__ payload_attr_data,
    uint64_t const data_length,
    uint64_t *const __restrict__ aggregation_result) {
  cpu_ht_probe_aggregate_linearprobing(hash_table, hash_table_entries,
                                       join_attr_data, payload_attr_data,
                                       data_length, aggregation_result);
}

template <typename T>
void cpu_ht_build_perfect(HtEntry<T, T> *const __restrict__ hash_table,
                          uint64_t const /* hash_table_entries */,
                          const T *const __restrict__ join_attribute_data,
                          const T *const __restrict__ payload_attributed_data,
                          uint64_t const data_length) {
  for (uint64_t tuple_id = 0; tuple_id < data_length; ++tuple_id) {
    T key = join_attribute_data[tuple_id];
    T val = payload_attributed_data[tuple_id];
    hash_table[key].key = key;
    hash_table[key].value = val;
  }
}

template <typename T>
void cpu_ht_build_selective_perfect(
    HtEntry<T, T> *const __restrict__ hash_table,
    uint64_t const /* hash_table_entries */,
    const T *const __restrict__ join_attribute_data,
    const T *const __restrict__ payload_attributed_data,
    uint64_t const data_length) {
  for (uint64_t tuple_id = 0; tuple_id < data_length; ++tuple_id) {
    T key = join_attribute_data[tuple_id];
    if (key != null_key<T>()) {
      T val = payload_attributed_data[tuple_id];
      hash_table[key].key = key;
      hash_table[key].value = val;
    }
  }
}

extern "C" void cpu_ht_build_perfect_int32(
    HtEntry<int, int> *const __restrict__ hash_table,
    uint64_t const hash_table_entries,
    const int *const __restrict__ join_attribute_data,
    const int *const __restrict__ payload_attributed_data,
    uint64_t const data_length) {
  cpu_ht_build_perfect(hash_table, hash_table_entries, join_attribute_data,
                       payload_attributed_data, data_length);
}

extern "C" void cpu_ht_build_selective_perfect_int32(
    HtEntry<int, int> *const __restrict__ hash_table,
    uint64_t const hash_table_entries,
    const int *const __restrict__ join_attribute_data,
    const int *const __restrict__ payload_attributed_data,
    uint64_t const data_length) {
  cpu_ht_build_selective_perfect(hash_table, hash_table_entries,
                                 join_attribute_data, payload_attributed_data,
                                 data_length);
}

extern "C" void cpu_ht_build_perfect_int64(
    HtEntry<long long, long long> *const __restrict__ hash_table,
    uint64_t const hash_table_entries,
    const long long *const __restrict__ join_attribute_data,
    const long long *const __restrict__ payload_attributed_data,
    uint64_t const data_length) {
  cpu_ht_build_perfect(hash_table, hash_table_entries, join_attribute_data,
                       payload_attributed_data, data_length);
}

extern "C" void cpu_ht_build_selective_perfect_int64(
    HtEntry<long long, long long> *const __restrict__ hash_table,
    uint64_t const hash_table_entries,
    const long long *const __restrict__ join_attribute_data,
    const long long *const __restrict__ payload_attributed_data,
    uint64_t const data_length) {
  cpu_ht_build_selective_perfect(hash_table, hash_table_entries,
                                 join_attribute_data, payload_attributed_data,
                                 data_length);
}

template <typename T>
void cpu_ht_probe_aggregate_perfect(
    const HtEntry<T, T> *const __restrict__ hash_table,
    uint64_t const /* hash_table_entries */,
    const T *const __restrict__ join_attribute_data,
    const T *const __restrict__ payload_attribute_data,
    uint64_t const data_length, uint64_t *__restrict__ aggregation_result) {
  for (uint64_t tuple_id = 0; tuple_id < data_length; ++tuple_id) {
    T key = join_attribute_data[tuple_id];
    if (hash_table[key].key == key) {
      *aggregation_result += payload_attribute_data[tuple_id];
    }
  }
}

extern "C" void cpu_ht_probe_aggregate_perfect_int32(
    const HtEntry<int, int> *const __restrict__ hash_table,
    uint64_t const hash_table_entries,
    const int *const __restrict__ join_attribute_data,
    const int *const __restrict__ payload_attribute_data,
    uint64_t const data_length, uint64_t *__restrict__ aggregation_result) {
  cpu_ht_probe_aggregate_perfect(hash_table, hash_table_entries,
                                 join_attribute_data, payload_attribute_data,
                                 data_length, aggregation_result);
}

extern "C" void cpu_ht_probe_aggregate_perfect_int64(
    const HtEntry<long long, long long> *const __restrict__ hash_table,
    uint64_t const hash_table_entries,
    const long long *const __restrict__ join_attribute_data,
    const long long *const __restrict__ payload_attribute_data,
    uint64_t const data_length, uint64_t *__restrict__ aggregation_result) {
  cpu_ht_probe_aggregate_perfect(hash_table, hash_table_entries,
                                 join_attribute_data, payload_attribute_data,
                                 data_length, aggregation_result);
}
