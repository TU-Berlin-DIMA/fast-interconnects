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

void cpu_ht_insert_linearprobing(
        std::atomic<int64_t> * __restrict__ hash_table,
        uint64_t hash_table_mask,
        int64_t key,
        int64_t payload
        )
{
    uint64_t index = 0;
    index = key;

    index *= HASH_FACTOR;
    for (uint64_t i = 0; i < hash_table_mask + 1; ++i, index += 2) {
        index &= hash_table_mask;
        index &= ~1ul;

        int64_t old = hash_table[index];
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

extern "C"
void cpu_ht_build_linearprobing(
        std::atomic<int64_t> *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const int64_t *const __restrict__ join_attr_data,
        const int64_t *const __restrict__ payload_attr_data,
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

bool cpu_ht_findkey_linearprobing(
        int64_t const *const __restrict__ hash_table,
        uint64_t hash_table_mask,
        int64_t key,
        int64_t const **found_payload,
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

extern "C"
void cpu_ht_probe_aggregate_linearprobing(
        int64_t const *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const int64_t *const __restrict__ join_attr_data,
        const int64_t *const __restrict__ /* payload_attr_data */,
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
        int64_t const *hash_table_payload = nullptr;
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
