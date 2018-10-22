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
 *
 * TODO:
 * - put constants into #define statements
 * - use HtEntry struct for key/payload pairs and adjust insert/probe logic
 */

typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long int uint64_t;
typedef unsigned long long int TID;
/* typedef char int8_t; */
typedef short int int16_t;
typedef int int32_t;
/* typedef long long int int64_t; */
/* #define C_MIN(a, b) (a = (a < b ? a : b)) */
/* #define C_MIN_uint32t(a, b) C_MIN(a, b) */
/* #define C_MIN_uint64t(a, b) C_MIN(a, b) */
/* #define C_MIN_double(a, b) C_MIN(a, b) */
/* #define C_MAX(a, b) (a = (a > b ? a : b)) */
/* #define C_MAX_uint32_t(a, b) C_MAX(a, b) */
/* #define C_MAX_uint64_t(a, b) C_MAX(a, b) */
/* #define C_MAX_double(a, b) C_MAX(a, b) */
/* #define C_SUM(a, b) (a += b) */
/* #define C_SUM_uint32_t(a, b) C_SUM(a, b) */
/* #define C_SUM_uint64_t(a, b) C_SUM(a, b) */
/* #define C_SUM_float(a, b) C_SUM(a, b) */
/* #define C_SUM_double(a, b) C_SUM(a, b) */
/* #ifndef NULL */
/* #define NULL 0 */
/* #endif */

/* __inline__ */
__device__
void insertIntoLinearProbingHTht_RT1_RT_XT1_build_mode(
        int64_t * __restrict__ hash_table,
        uint64_t hash_table_mask,
        int64_t key,
        int64_t payload
        )
{
    uint32_t index = 0;
    index = key;

    // Question: What makes a good hash value?
    index *= 123456789123456789ul;
    for (uint32_t i = 0; i < hash_table_mask + 1; ++i, index += 2) {
        // effectively index = index % ht_size
        index &= hash_table_mask;

        // ensures index is a multiple of 2, i.e., index = index - (index % 2)
        // This is done because each key/payload pair occupies 2 slots in ht array
        // Question: does this lead to unnecessary hash conflicts?
        //   Answer: yes, but only for 1 entry. Multiple entries are on same probe chain anyways
        index &= ~1ul;

        //    int64_t null_key = 0xFFFFFFFFFFFFFFFF;
        //    int64_t old = hash_table[index];
        //    if (old == key) {
        //      return;
        //    } else if (old == null_key) {
        //      old = atom_cmpxchg(&hash_table[index], null_key, key);
        //      if (old == null_key || old == key) {
        //        hash_table[index + 1] = payload;
        //        return;
        //      }
        //    }

        int64_t null_key = 0xFFFFFFFFFFFFFFFF;
        int64_t old = hash_table[index];
        if (old == null_key) {
            old = (int64_t)atomicCAS((uint64_t*)&hash_table[index], (uint64_t)null_key, (uint64_t)key);
            if (old == null_key) {
                hash_table[index + 1] = payload;
                return;
            }
        }
    }
}

extern "C"
__global__
void build_pipeline_kernel(
        uint64_t const num_elements,
        uint64_t * __restrict__ result_size,
        const int64_t *const __restrict__ selection_column_data,
        const int64_t *const __restrict__ join_column_data,
        uint64_t const ht_RT1_RT_XT1_build_mode_length,
        int64_t * __restrict__ ht_RT1_RT_XT1_build_mode
        )
{
    const uint32_t global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    const uint32_t number_of_threads = blockDim.x * gridDim.x;

    int64_t write_pos = 0;
    uint32_t tuple_id_RT1 = global_idx;
    while (tuple_id_RT1 < num_elements) {
        if (selection_column_data[tuple_id_RT1] > 1) {
            insertIntoLinearProbingHTht_RT1_RT_XT1_build_mode(
                    ht_RT1_RT_XT1_build_mode,
                    ht_RT1_RT_XT1_build_mode_length - 1,
                    join_column_data[tuple_id_RT1],
                    write_pos
                    );

            write_pos++;
        }
        tuple_id_RT1 += number_of_threads;
    }

    result_size[0] = write_pos;
}

__device__
bool findKeyLinearProbingHThash_table(
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
        index *= 123456789123456789ul;
    }

    int64_t null_key = 0xFFFFFFFFFFFFFFFF;
    for (uint32_t i = 0; i < hash_table_mask + 1; ++i, index += 2) {
        index &= hash_table_mask;
        index &= ~1ul;
        //printf("Key: %ld  Hash Table Bucket: %ld\n", key, hash_table[index]);

        if (hash_table[index] == key) {
            *found_payload = &hash_table[index + 1];
            *last_index = index;
            return true;
        } else if (hash_table[index] == null_key) {
            return false;
        }
    }

    return false;
}

extern "C"
__global__
void aggregation_kernel(
        uint64_t const num_elements,
        const int64_t *const __restrict__ array_ST1_ST_BT1,
        const int64_t *const __restrict__ array_ST1_ST_YT1,
        const int64_t *const __restrict__ hash_table,
        uint64_t const hash_table_length,
        uint64_t * __restrict__ COUNT_OF_ST_BT1_COUNT
        )
{
    const uint32_t global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    const uint32_t number_of_threads = blockDim.x * gridDim.x;
    uint32_t tuple_id_ST1 = global_idx;

    while (tuple_id_ST1 < num_elements) {
        if (((array_ST1_ST_BT1[tuple_id_ST1] > 1))) {
            /* TID tuple_id_RT = 0; */
            int64_t const *hash_table_payload = nullptr;
            uint32_t hash_table_last_index = 0;
            bool hash_table_use_last_index = false;
            while (findKeyLinearProbingHThash_table(
                        hash_table, hash_table_length - 1,
                        array_ST1_ST_YT1[tuple_id_ST1], &hash_table_payload,
                        &hash_table_last_index,
                        hash_table_use_last_index)) {

                hash_table_use_last_index = true;
                COUNT_OF_ST_BT1_COUNT[global_idx] += 1;
            }
        }
        tuple_id_ST1 += number_of_threads;
    }
}
