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
 * - use HtEntry struct for key/payload pairs and adjust insert/probe logic
 */

/* See Richter et al., Seven-Dimensional Analysis of Hashing Methods
 * Multiply-shift hash function
 * Requirement: hash factor is an odd 64-bit integer
*/
#define HASH_FACTOR 123456789123456789ull

#define NULL_KEY_32 0xFFFFFFFF
#define NULL_KEY_64 0xFFFFFFFFFFFFFFFFll

#define OPTIMIZE_INT32

/*
 * Note: uint64_t in cstdint header doesn't match atomicCAS()
 */
typedef unsigned int uint32_t;
typedef unsigned long long int uint64_t;

__device__
void gpu_ht_insert_linearprobing_int32(
        int32_t * __restrict__ hash_table,
        uint64_t hash_table_mask,
        int32_t key,
        int32_t payload
        )
{
    uint64_t index = 0;
    index = key;

    index *= HASH_FACTOR;
    for (uint64_t i = 0; i < hash_table_mask + 1; ++i, index += 2) {
        // Effectively index = index % ht_size
        index &= hash_table_mask;

        // Effectively a index = index % 2
        // This is done because each key/payload pair occupies 2 slots in ht array
        index &= ~1ul;

#ifdef OPTIMIZE_INT32
        uint64_t null_key_64 = NULL_KEY_64;
        int32_t old_key = hash_table[index];
        uint64_t entry = (((uint64_t)key) << 32) | ((uint64_t)payload);
        if (old_key == NULL_KEY_32) {
            uint64_t old_entry = atomicCAS((uint64_t*)&hash_table[index], null_key_64, entry);
            old_key = old_entry >> 32;
            if (old_key == NULL_KEY_32) {
                return;
            }
        }
#else
        unsigned int null_key = NULL_KEY_32;
        int32_t old = hash_table[index];
        if (old == NULL_KEY_32) {
            old = (int32_t)atomicCAS((unsigned int*)&hash_table[index], null_key, (unsigned int)key);
            if (old == NULL_KEY_32) {
                hash_table[index + 1] = payload;
                return;
            }
        }
#endif
    }
}

__device__
void gpu_ht_insert_linearprobing_int64(
        int64_t * __restrict__ hash_table,
        uint64_t hash_table_mask,
        int64_t key,
        int64_t payload
        )
{
    uint64_t index = 0;
    index = key;

    index *= HASH_FACTOR;
    for (uint64_t i = 0; i < hash_table_mask + 1; ++i, index += 2) {
        // Effectively index = index % ht_size
        index &= hash_table_mask;

        // Effectively a index = index % 2
        // This is done because each key/payload pair occupies 2 slots in ht array
        index &= ~1ul;

        unsigned long long int null_key = NULL_KEY_64;
        int64_t old = hash_table[index];
        if (old == NULL_KEY_64) {
            old = (int64_t)atomicCAS((unsigned long long int*)&hash_table[index], null_key, (unsigned long long int)key);
            if (old == NULL_KEY_64) {
                hash_table[index + 1] = payload;
                return;
            }
        }
    }
}

extern "C"
__global__
void gpu_ht_build_linearprobing_int32(
        int32_t *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const int32_t *const __restrict__ join_attr_data,
        const int32_t *const __restrict__ payload_attr_data,
        uint64_t const data_length
        )
{
    const uint32_t global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    const uint32_t global_threads = blockDim.x * gridDim.x;

    for (
            uint64_t tuple_id = global_idx;
            tuple_id < data_length;
            tuple_id += global_threads
        )
    {
        gpu_ht_insert_linearprobing_int32(
                hash_table,
                hash_table_entries - 1,
                join_attr_data[tuple_id],
                payload_attr_data[tuple_id]
                );
    }
}

extern "C"
__global__
void gpu_ht_build_linearprobing_int64(
        int64_t *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const int64_t *const __restrict__ join_attr_data,
        const int64_t *const __restrict__ payload_attr_data,
        uint64_t const data_length
        )
{
    const uint32_t global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    const uint32_t global_threads = blockDim.x * gridDim.x;

    for (
            uint64_t tuple_id = global_idx;
            tuple_id < data_length;
            tuple_id += global_threads
        )
    {
        gpu_ht_insert_linearprobing_int64(
                hash_table,
                hash_table_entries - 1,
                join_attr_data[tuple_id],
                payload_attr_data[tuple_id]
                );
    }
}

__device__
bool gpu_ht_findkey_linearprobing_int32(
        int32_t const *const __restrict__ hash_table,
        uint64_t hash_table_mask,
        int32_t key,
        int32_t const **found_payload,
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
        } else if (hash_table[index] == NULL_KEY_32) {
            return false;
        }
    }

    return false;
}

__device__
bool gpu_ht_findkey_linearprobing_int64(
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
        } else if (hash_table[index] == NULL_KEY_64) {
            return false;
        }
    }

    return false;
}

extern "C"
__global__
void gpu_ht_probe_aggregate_linearprobing_int32(
        int32_t const *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const int32_t *const __restrict__ join_attr_data,
        const int32_t *const __restrict__ /* payload_attr_data */,
        uint64_t const data_length,
        uint64_t * __restrict__ aggregation_result
        )
{
    const uint32_t global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    const uint32_t global_threads = blockDim.x * gridDim.x;

    for (
            uint64_t tuple_id = global_idx;
            tuple_id < data_length;
            tuple_id += global_threads
        )
    {
        int32_t const *hash_table_payload = nullptr;
        uint64_t hash_table_last_index = 0;
        bool hash_table_use_last_index = false;
        while (
                gpu_ht_findkey_linearprobing_int32(
                    hash_table,
                    hash_table_entries - 1,
                    join_attr_data[tuple_id],
                    &hash_table_payload,
                    &hash_table_last_index,
                    hash_table_use_last_index)
              )
        {
            hash_table_use_last_index = true;
            aggregation_result[global_idx] += 1;
        }
    }
}

extern "C"
__global__
void gpu_ht_probe_aggregate_linearprobing_int64(
        int64_t const *const __restrict__ hash_table,
        uint64_t const hash_table_entries,
        const int64_t *const __restrict__ join_attr_data,
        const int64_t *const __restrict__ /* payload_attr_data */,
        uint64_t const data_length,
        uint64_t * __restrict__ aggregation_result
        )
{
    const uint32_t global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    const uint32_t global_threads = blockDim.x * gridDim.x;

    for (
            uint64_t tuple_id = global_idx;
            tuple_id < data_length;
            tuple_id += global_threads
        )
    {
        int64_t const *hash_table_payload = nullptr;
        uint64_t hash_table_last_index = 0;
        bool hash_table_use_last_index = false;
        while (
                gpu_ht_findkey_linearprobing_int64(
                    hash_table,
                    hash_table_entries - 1,
                    join_attr_data[tuple_id],
                    &hash_table_payload,
                    &hash_table_last_index,
                    hash_table_use_last_index)
              )
        {
            hash_table_use_last_index = true;
            aggregation_result[global_idx] += 1;
        }
    }
}

extern "C"
__global__
void gpu_ht_build_perfect_int32(
        int32_t *const __restrict__ hash_table,
        uint64_t const /* hash_table_entries */,
        const int32_t *const __restrict__ join_attribute_data,
        const int32_t *const __restrict__ payload_attributed_data,
        uint64_t const data_length
        )
{
    const uint32_t global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    const uint32_t global_threads = blockDim.x * gridDim.x;

    for (uint64_t i = global_idx; i < data_length; i += global_threads) {
        int32_t key = join_attribute_data[i];
        int32_t val = payload_attributed_data[i];
        hash_table[key] = key;
        hash_table[key + 1] = val;
    }
}

extern "C"
__global__
void gpu_ht_build_perfect_int64(
        int64_t *const __restrict__ hash_table,
        uint64_t const /* hash_table_entries */,
        const int64_t *const __restrict__ join_attribute_data,
        const int64_t *const __restrict__ payload_attributed_data,
        uint64_t const data_length
        )
{
    const uint32_t global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    const uint32_t global_threads = blockDim.x * gridDim.x;

    for (uint64_t i = global_idx; i < data_length; i += global_threads) {
        int64_t key = join_attribute_data[i];
        int64_t val = payload_attributed_data[i];
        hash_table[key] = key;
        hash_table[key + 1] = val;
    }
}

extern "C"
__global__
void gpu_ht_probe_aggregate_perfect_int32(
        const int32_t *const __restrict__ hash_table,
        uint64_t const /* hash_table_entries */,
        const int32_t *const __restrict__ join_attribute_data,
        const int32_t *const __restrict__ /* payload_attributed_data */,
        uint64_t const data_length,
        uint64_t * __restrict__ aggregation_result
        )
{
    const uint32_t global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    const uint32_t global_threads = blockDim.x * gridDim.x;

    for (uint64_t i = global_idx; i < data_length; i += global_threads) {
        int32_t key = join_attribute_data[i];
        if (hash_table[key] != NULL_KEY_32) {
            aggregation_result[global_idx] += 1;
        }
    }
}

extern "C"
__global__
void gpu_ht_probe_aggregate_perfect_int64(
        const int64_t *const __restrict__ hash_table,
        uint64_t const /* hash_table_entries */,
        const int64_t *const __restrict__ join_attribute_data,
        const int64_t *const __restrict__ /* payload_attributed_data */,
        uint64_t const data_length,
        uint64_t * __restrict__ aggregation_result
        )
{
    const uint32_t global_idx = blockIdx.x *blockDim.x + threadIdx.x;
    const uint32_t global_threads = blockDim.x * gridDim.x;

    for (uint64_t i = global_idx; i < data_length; i += global_threads) {
        int64_t key = join_attribute_data[i];
        if (hash_table[key] != NULL_KEY_64) {
            aggregation_result[global_idx] += 1;
        }
    }
}
