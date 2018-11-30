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
    uint32_t index = 0;
    index = key;

    index *= HASH_FACTOR;
    for (uint32_t i = 0; i < hash_table_mask + 1; ++i, index += 2) {
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
        uint64_t const num_elements,
        uint64_t * __restrict__ result_size,
        const int64_t *const __restrict__ selection_column_data,
        const int64_t *const __restrict__ join_column_data,
        uint64_t const hash_table_size,
        std::atomic<int64_t> * __restrict__ hash_table
        )
{
    int64_t write_pos = 0;

    for (
            uint32_t tuple_id_RT1 = 0;
            tuple_id_RT1 < num_elements;
            ++tuple_id_RT1
        )
    {
        if (selection_column_data[tuple_id_RT1] > 1) {
            cpu_ht_insert_linearprobing(
                    hash_table,
                    hash_table_size - 1,
                    join_column_data[tuple_id_RT1],
                    write_pos
                    );

            write_pos++;
        }
    }

    result_size[0] = write_pos;
}

bool cpu_ht_findkey_linearprobing(
        int64_t const *const __restrict__ hash_table,
        uint64_t hash_table_mask,
        int64_t key,
        int64_t const **found_payload,
        uint32_t * __restrict__ last_index,
        bool use_last_index
        )
{
    uint32_t index = 0;
    if (use_last_index) {
        index = *last_index;
        index += 2;
    } else {
        index = key;
        index *= HASH_FACTOR;
    }

    for (uint32_t i = 0; i < hash_table_mask + 1; ++i, index += 2) {
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
        uint64_t const num_elements,
        const int64_t *const __restrict__ filter_attribute_data,
        const int64_t *const __restrict__ join_attribute_data,
        const int64_t *const __restrict__ hash_table,
        uint64_t const hash_table_length,
        uint64_t *aggregation_result
        )
{
    for (
            uint32_t tuple_id_ST1 = 0;
            tuple_id_ST1 < num_elements;
            ++tuple_id_ST1
        )
    {
        if (((filter_attribute_data[tuple_id_ST1] > 1))) {
            int64_t const *hash_table_payload = nullptr;
            uint32_t hash_table_last_index = 0;
            bool hash_table_use_last_index = false;
            while (cpu_ht_findkey_linearprobing(
                        hash_table, hash_table_length - 1,
                        join_attribute_data[tuple_id_ST1], &hash_table_payload,
                        &hash_table_last_index,
                        hash_table_use_last_index)) {

                hash_table_use_last_index = true;
                *aggregation_result += 1;
            }
        }
    }
}
