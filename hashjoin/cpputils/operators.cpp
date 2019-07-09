/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
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

/* See Richter et al., Seven-Dimensional Analysis of Hashing Methods
 * Multiply-shift hash function
 * Requirement: hash factor is an odd 64-bit integer
*/
#define HASH_FACTOR 123456789123456789ull

#define NULL_KEY ((long)0xFFFFFFFFFFFFFFFFL)

#include <atomic>
#include <cstdint>

template<typename T>
void cpu_ht_insert_linearprobing(
        std::atomic<T> * __restrict__ hash_table,
        uint64_t hash_table_mask,
        T key,
        T payload
        )
{
    uint64_t index = 0;
    index = key;

    index *= HASH_FACTOR;
    for (uint64_t i = 0; i < hash_table_mask + 1; ++i, index += 2) {
        index &= hash_table_mask;
        index &= ~1ul;

        T old = hash_table[index];
        if (old == NULL_KEY) {
            uint64_t expected = (uint64_t) NULL_KEY;
            bool is_inserted = std::atomic_compare_exchange_strong(
                    (std::atomic<uint64_t>*) &hash_table[index],
                    &expected,
                    (uint64_t) key
                    );
            if (is_inserted) {
                hash_table[index + 1] = payload;
                return;
            }
        }
    }
}

// extern "C"
template<typename T>
void cpu_ht_build_linearprobing(
        std::atomic<T> *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const T *const __restrict__ join_attr_data,
        const T *const __restrict__ payload_attr_data,
        uint64_t const data_length
        )
{
    for (
            uint64_t tuple_id = 0;
            tuple_id < data_length;
            ++tuple_id
        )
    {
        cpu_ht_insert_linearprobing(
                hash_table,
                hash_table_entries - 1,
                join_attr_data[tuple_id],
                payload_attr_data[tuple_id]
                );
    }
}

extern "C"
void cpu_ht_build_linearprobing_int32(
        std::atomic<int32_t> *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const int32_t *const __restrict__ join_attr_data,
        const int32_t *const __restrict__ payload_attr_data,
        uint64_t const data_length
        )
{
    cpu_ht_build_linearprobing(hash_table, hash_table_entries, join_attr_data, payload_attr_data, data_length);
}

extern "C"
void cpu_ht_build_linearprobing_int64(
        std::atomic<int64_t> *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const int64_t *const __restrict__ join_attr_data,
        const int64_t *const __restrict__ payload_attr_data,
        uint64_t const data_length
        )
{
    cpu_ht_build_linearprobing(hash_table, hash_table_entries, join_attr_data, payload_attr_data, data_length);
}

template<typename T>
bool cpu_ht_findkey_linearprobing(
        T const *const __restrict__ hash_table,
        uint64_t hash_table_mask,
        T key,
        T const **found_payload,
        uint64_t * __restrict__ last_index,
        bool use_last_index
        )
{
    uint64_t index = 0;
    if (use_last_index) {
        index = *last_index;
        index += 2;
    } else {
        index = key;
        index *= HASH_FACTOR;
    }

    for (uint64_t i = 0; i < hash_table_mask + 1; ++i, index += 2) {
        index &= hash_table_mask;
        index &= ~1ul;

        if (hash_table[index] == key) {
            *found_payload = &hash_table[index + 1];
            *last_index = index;
            return true;
        } else if (hash_table[index] == NULL_KEY) {
            return false;
        }
    }

    return false;
}

template<typename T>
void cpu_ht_probe_aggregate_linearprobing(
        T const *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const T *const __restrict__ join_attr_data,
        const T *const __restrict__ /* payload_attr_data */,
        uint64_t const data_length,
        uint64_t *const __restrict__ aggregation_result
        )
{
    for (
            uint64_t tuple_id = 0;
            tuple_id < data_length;
            ++tuple_id
        )
    {
        T const *hash_table_payload = nullptr;
        uint64_t hash_table_last_index = 0;
        bool hash_table_use_last_index = false;
        while (cpu_ht_findkey_linearprobing(
                    hash_table, hash_table_entries - 1,
                    join_attr_data[tuple_id], &hash_table_payload,
                    &hash_table_last_index,
                    hash_table_use_last_index)) {

            hash_table_use_last_index = true;
            *aggregation_result += 1;
        }
    }
}

extern "C"
void cpu_ht_probe_aggregate_linearprobing_int32(
        int32_t const *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const int32_t *const __restrict__ join_attr_data,
        const int32_t *const __restrict__ payload_attr_data,
        uint64_t const data_length,
        uint64_t *const __restrict__ aggregation_result
        )
{
    cpu_ht_probe_aggregate_linearprobing(
            hash_table,
            hash_table_entries,
            join_attr_data,
            payload_attr_data,
            data_length,
            aggregation_result
            );
}

extern "C"
void cpu_ht_probe_aggregate_linearprobing_int64(
        int64_t const *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const int64_t *const __restrict__ join_attr_data,
        const int64_t *const __restrict__ payload_attr_data,
        uint64_t const data_length,
        uint64_t *const __restrict__ aggregation_result
        )
{
    cpu_ht_probe_aggregate_linearprobing(
            hash_table,
            hash_table_entries,
            join_attr_data,
            payload_attr_data,
            data_length,
            aggregation_result
            );
}

template<typename T>
void cpu_ht_build_perfect(
        T *const __restrict__ hash_table,
        uint64_t const /* hash_table_entries */,
        const T *const __restrict__ join_attribute_data,
        const T *const __restrict__ payload_attributed_data,
        uint64_t const data_length
        )
{
    for (uint64_t tuple_id = 0; tuple_id < data_length; ++tuple_id) {
        T key = join_attribute_data[tuple_id];
        T val = payload_attributed_data[tuple_id];
        hash_table[key] = key;
        hash_table[key + 1] = val;
    }
}

extern "C"
void cpu_ht_build_perfect_int32(
        int32_t *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const int32_t *const __restrict__ join_attribute_data,
        const int32_t *const __restrict__ payload_attributed_data,
        uint64_t const data_length
        )
{
    cpu_ht_build_perfect(
            hash_table,
            hash_table_entries,
            join_attribute_data,
            payload_attributed_data,
            data_length
            );
}

extern "C"
void cpu_ht_build_perfect_int64(
        int64_t *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const int64_t *const __restrict__ join_attribute_data,
        const int64_t *const __restrict__ payload_attributed_data,
        uint64_t const data_length
        )
{
    cpu_ht_build_perfect(
            hash_table,
            hash_table_entries,
            join_attribute_data,
            payload_attributed_data,
            data_length
            );
}

template<typename T>
void cpu_ht_probe_aggregate_perfect(
        const T *const __restrict__ hash_table,
        uint64_t const /* hash_table_entries */,
        const T *const __restrict__ join_attribute_data,
        const T *const __restrict__ /* payload_attribute_data */,
        uint64_t const data_length,
        uint64_t * __restrict__ aggregation_result
        )
{
    for (uint64_t tuple_id = 0; tuple_id < data_length; ++tuple_id) {
        T key = join_attribute_data[tuple_id];
        if (hash_table[key] != NULL_KEY) {
            *aggregation_result += 1;
        }
    }
}

extern "C"
void cpu_ht_probe_aggregate_perfect_int32(
        const int32_t *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const int32_t *const __restrict__ join_attribute_data,
        const int32_t *const __restrict__ payload_attribute_data,
        uint64_t const data_length,
        uint64_t * __restrict__ aggregation_result
        )
{
    cpu_ht_probe_aggregate_perfect(
            hash_table,
            hash_table_entries,
            join_attribute_data,
            payload_attribute_data,
            data_length,
            aggregation_result
            );
}

extern "C"
void cpu_ht_probe_aggregate_perfect_int64(
        const int64_t *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const int64_t *const __restrict__ join_attribute_data,
        const int64_t *const __restrict__ payload_attribute_data,
        uint64_t const data_length,
        uint64_t * __restrict__ aggregation_result
        )
{
    cpu_ht_probe_aggregate_perfect(
            hash_table,
            hash_table_entries,
            join_attribute_data,
            payload_attribute_data,
            data_length,
            aggregation_result
            );
}

