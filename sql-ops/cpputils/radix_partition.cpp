/*
 * Copyright (c) 2014 Cagri Balkesen, ETH Zurich
 * Copyright (c) 2014 Claude Barthels, ETH Zurich
 * Copyright (c) 2019-2021 Clemens Lutz, German Research Center for Artificial
 * Intelligence
 *
 * Original sources by Cagri Balkesen and Claude Barthels are copyrighted under
 * the MIT license.
 *
 * Modications by Clemens Lutz are copyrighted under the Apache License 2.0.
 *
 * MIT license:
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Apache License 2.0:
 *
 * Author: Clemens Lutz, DFKI GmbH <clemens.lutz@dfki.de>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <constants.h>

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__ALTIVEC__)
#include <altivec.h>
#include <ppc_intrinsics.h>

#ifdef bool
// Workaround for AltiVec redefinition of bool.
// See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58241
// and https://bugzilla.redhat.com/show_bug.cgi?id=1394505
#undef bool
#endif
#endif

#include <cassert>
#include <cstdint>
#include <cstring>

// Defines the cache-line size; usually this should be passed via the build
// script.
#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64U
#endif

// Defines the software write-combine buffer size; usually this should be passed
// via the build script.
#ifndef SWWC_BUFFER_SIZE
#define SWWC_BUFFER_SIZE CACHE_LINE_SIZE
#endif

using namespace std;

// Arguments to the prefix sum function.
//
// Note that the struct's layout must be kept in sync with its counterpart in
// Rust.
struct PrefixSumArgs {
  // Inputs
  const void *const __restrict__ partition_attr;
  size_t const data_length;
  size_t const canonical_chunk_length;
  uint32_t const padding_length;
  uint32_t const radix_bits;
  uint32_t const ignore_bits;

  // State
  unsigned int *const __restrict__ tmp_partition_offsets;

  // Outputs
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
  size_t const data_length;
  size_t const padding_length;
  uint32_t const radix_bits;
  uint32_t const ignore_bits;
  const unsigned long long *const __restrict__ partition_offsets;

  // State
  unsigned long long *const __restrict__ tmp_partition_offsets;
  void *const __restrict__ write_combine_buffer;

  // Outputs
  void *const __restrict__ partitioned_relation;
};

// A key-value tuple.
//
// Note that the struct's layout must be kept in sync with its counterpart in
// Rust.
template <typename K, typename V>
struct Tuple {
  K key;
  V value;
};

// A set of buffers used for software write-combinining.
//
// Supports two views of its data. The purpose is to align each buffer to the
// cache-line size. This requires periodic overwriting of `meta.slot` when a
// buffer becomes full. After emptying the buffer, the slot's value must be
// restored.
template <typename T, uint32_t size>
union WriteCombineBuffer {
  struct {
    T data[size / sizeof(T)];
  } tuples;

  struct {
    T data[(size - sizeof(uint64_t)) / sizeof(T)];
    char _padding[(size - sizeof(uint64_t)) -
                  (((size - sizeof(uint64_t)) / sizeof(T)) * sizeof(T))];
    uint64_t slot;  // Padding makes `slot` 8-byte aligned if sizeof(T) % 8 != 0
  } meta;

  // Computes the number of tuples contained in a buffer.
  static constexpr size_t tuples_per_buffer() { return size / sizeof(T); }
} __attribute__((packed));

// Computes the partition ID of a given key.
template <typename T, typename M, typename B>
M key_to_partition(T key, M mask, B bits) {
  return (static_cast<M>(key) & mask) >> bits;
}

#if defined(__ALTIVEC__)
template <typename T, typename M, typename B>
vector M key_to_partition_simd(vector T key, M mask, B bits) {
  return (reinterpret_cast<M>(key) & mask) >> bits;
}
#endif

// Flushes a SWWC buffer from cache to memory.
//
// If possible, uses non-temporal SIMD writes, that require vector-length
// alignment. This in turn requires padding, because the front of a buffer may
// contain invalid data on the first flush.
void flush_buffer(void *const __restrict__ dst,
                  const void *const __restrict__ src) {
  auto byte_dst = static_cast<char *>(dst);
  auto byte_src = static_cast<const char *>(src);

#if defined(__AVX512F__)
  for (size_t i = 0; i < SWWC_BUFFER_SIZE; i += 64) {
    auto avx_dst = reinterpret_cast<__m512i *>(byte_dst + i);
    auto avx_src = reinterpret_cast<const __m512i *>(byte_src + i);

    _mm512_stream_si512(avx_dst, *avx_src);
  }
#elif defined(__AVX__)
  for (size_t i = 0; i < SWWC_BUFFER_SIZE; i += 32) {
    auto avx_dst = reinterpret_cast<__m256i *>(byte_dst + i);
    auto avx_src = reinterpret_cast<const __m256i *>(byte_src + i);

    _mm256_stream_si256(avx_dst, *avx_src);
  }
#elif defined(__SSE2__)
  for (size_t i = 0; i < SWWC_BUFFER_SIZE; i += 16) {
    auto sse_dst = reinterpret_cast<__m128i *>(byte_dst + i);
    auto sse_src = reinterpret_cast<const __m128i *>(byte_src + i);

    _mm_stream_si128(sse_dst, *sse_src);
  }
#elif defined(__ALTIVEC__)
  // vec_ld: 128-bit vector load; requires requires 16-byte alignment
  // vec_st: 128-bit vector store; requires requires 16-byte alignment
  // See also:
  // https://gcc.gcc.gnu.narkive.com/cJndcMpR/vec-ld-versus-vec-vsx-ld-on-power8
  //
  // _mm_stream_si128 wraps dcbtstt
  // See also:
  // https://github.com/gcc-mirror/gcc/blob/74c5e5f5bf7f2f13718008421cdf53bb0a814f4c/gcc/config/rs6000/emmintrin.h#L2249
  for (size_t i = 0; i < SWWC_BUFFER_SIZE; i += 64) {
    int *vsx_dst = reinterpret_cast<int *>(byte_dst + i);
    const int *vsx_src = reinterpret_cast<const int *>(byte_src + i);

    vector int tmp0 = vec_ld(0, vsx_src);
    vector int tmp1 = vec_ld(16, vsx_src);
    vector int tmp2 = vec_ld(32, vsx_src);
    vector int tmp3 = vec_ld(48, vsx_src);

    vec_st(tmp0, 0, vsx_dst);
    vec_st(tmp1, 16, vsx_dst);
    vec_st(tmp2, 32, vsx_dst);
    vec_st(tmp3, 48, vsx_dst);
  }
#else
  memcpy(byte_dst, byte_src, SWWC_BUFFER_SIZE);
#endif
}

// Chunked histogram and offset computation.
//
// See the Rust module for details.
template <typename K, typename M>
void cpu_chunked_prefix_sum(PrefixSumArgs &args, uint32_t const chunk_id,
                            uint32_t const /* num_chunks */) {
  const size_t fanout = 1UL << args.radix_bits;
  const M mask = static_cast<M>(fanout - 1UL);

  auto partition_attr =
      static_cast<const K *const __restrict__>(args.partition_attr);

  // Ensure counters are all zeroed
  for (size_t i = 0; i < fanout; ++i) {
    args.tmp_partition_offsets[i] = 0;
  }

  // Compute local histograms per partition for chunk
#pragma GCC unroll 16
  for (size_t i = 0; i < args.data_length; ++i) {
    auto key = partition_attr[i];
    M p_index = key_to_partition(key, mask, 0);
    args.tmp_partition_offsets[p_index] += 1;
  }

  // Compute offsets with exclusive prefix sum
  size_t partitioned_data_offset =
      (args.canonical_chunk_length + args.padding_length * fanout) * chunk_id;
  uint64_t offset = partitioned_data_offset + args.padding_length;
  for (uint32_t i = 0; i < fanout; ++i) {
    // Add data offset onto partitions offsets and write out the final offsets
    // to device memory.
    args.partition_offsets[i] = offset;

    // Update offset
    offset += static_cast<uint64_t>(args.tmp_partition_offsets[i]);
    offset += args.padding_length;
  }
}

#if defined(__ALTIVEC__)
// Chunked histogram and offset computation with SIMD optimizations.
//
// See the Rust module for details.
template <typename K, typename M>
void cpu_chunked_prefix_sum_simd(PrefixSumArgs &args, uint32_t const chunk_id,
                                 uint32_t const /* num_chunks */) {
  constexpr size_t vec_len = sizeof(vector int) / sizeof(K);
  const size_t fanout = 1UL << args.radix_bits;
  const M mask = static_cast<M>(fanout - 1UL);

  auto partition_attr =
      static_cast<const K *const __restrict__>(args.partition_attr);

  assert(((size_t)partition_attr) % vec_len == 0U &&
         "128-bit intrinsics require 16-byte alignment");

  const vector M mask_vsx = vec_splats(mask);
  const vector M ignore_bits_vsx = vec_splats(static_cast<M>(0U));
  size_t i;

  // Ensure counters are all zeroed
  for (size_t i = 0; i < fanout; ++i) {
    args.tmp_partition_offsets[i] = 0;
  }

  // Compute local histograms per partition
  i = 0;
  if (args.data_length > vec_len * 4) {
    for (; i < (args.data_length - vec_len * 4); i += vec_len * 4) {
      const K *const base = &partition_attr[i];

      vector K key0 = vec_ld(0, base);
      vector K key1 = vec_ld(16, base);
      vector K key2 = vec_ld(32, base);
      vector K key3 = vec_ld(48, base);

      vector M p_index0 =
          key_to_partition_simd(key0, mask_vsx, ignore_bits_vsx);
      vector M p_index1 =
          key_to_partition_simd(key1, mask_vsx, ignore_bits_vsx);
      vector M p_index2 =
          key_to_partition_simd(key2, mask_vsx, ignore_bits_vsx);
      vector M p_index3 =
          key_to_partition_simd(key3, mask_vsx, ignore_bits_vsx);

      for (uint32_t v = 0; v < vec_len; ++v) {
        args.tmp_partition_offsets[p_index0[v]] += 1;
        args.tmp_partition_offsets[p_index1[v]] += 1;
        args.tmp_partition_offsets[p_index2[v]] += 1;
        args.tmp_partition_offsets[p_index3[v]] += 1;
      }
    }
  }
  for (; i < args.data_length; ++i) {
    auto key = partition_attr[i];
    auto p_index = key_to_partition(key, mask, 0);
    args.tmp_partition_offsets[p_index] += 1;
  }

  // Compute offsets with exclusive prefix sum
  size_t partitioned_data_offset =
      (args.canonical_chunk_length + args.padding_length * fanout) * chunk_id;
  uint64_t offset = partitioned_data_offset + args.padding_length;
  for (uint32_t i = 0; i < fanout; ++i) {
    // Add data offset onto partitions offsets and write out the final offsets
    // to device memory.
    args.partition_offsets[i] = offset;

    // Update offset
    offset += static_cast<uint64_t>(args.tmp_partition_offsets[i]);
    offset += args.padding_length;
  }
}
#endif /* defined(__ALTIVEC__) */

// Chunked radix partitioning.
//
// See the Rust module for details.
template <typename K, typename V, typename M>
void cpu_chunked_radix_partition(RadixPartitionArgs &args) {
  auto join_attr_data =
      static_cast<const K *const __restrict__>(args.join_attr_data);
  auto payload_attr_data =
      static_cast<const V *const __restrict__>(args.payload_attr_data);
  auto partitioned_relation =
      static_cast<Tuple<K, V> *const __restrict__>(args.partitioned_relation);
  auto tmp_partition_offsets = args.tmp_partition_offsets;

  const size_t fanout = 1UL << args.radix_bits;
  const M mask = static_cast<M>(fanout - 1UL);
  const size_t partitioned_data_offset = args.partition_offsets[0];

  // Load partition offsets.
  for (size_t i = 0; i < fanout; ++i) {
    tmp_partition_offsets[i] =
        args.partition_offsets[i] - partitioned_data_offset;
  }

  // Partition.
#pragma GCC unroll 16
  for (size_t i = 0; i < args.data_length; ++i) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    M p_index = key_to_partition(tuple.key, mask, 0);
    auto &offset = tmp_partition_offsets[p_index];
    partitioned_relation[offset] = tuple;
    offset += 1;
  }
}

template <typename K, typename V, typename M>
void buffer_tuple(Tuple<K, V> *const __restrict__ partitioned_relation,
                  WriteCombineBuffer<Tuple<K, V>, SWWC_BUFFER_SIZE>
                      *const __restrict__ buffers,
                  M p_index, K key, V payload) {
  constexpr size_t tuples_per_buffer =
      WriteCombineBuffer<Tuple<K, V>, SWWC_BUFFER_SIZE>::tuples_per_buffer();

  Tuple<K, V> tuple;
  tuple.key = key;
  tuple.value = payload;

  auto &buffer = buffers[p_index];

  size_t slot = buffer.meta.slot;
  size_t buffer_slot = slot % tuples_per_buffer;

  // `buffer.meta.slot` is overwritten on buffer_slot == (tuples_per_buffer -
  // 1), and restored after the buffer flush.
  buffer.tuples.data[buffer_slot] = tuple;

  // Flush buffer
  // Can occur on partially filled buffer due to cache-line alignment,
  // because first output slot might not be at offset % tuples_per_buffer == 0
  if (buffer_slot + 1 == tuples_per_buffer) {
    flush_buffer(partitioned_relation + (slot + 1) - tuples_per_buffer,
                 buffer.tuples.data);
  }

  // Restore `buffer.meta.slot` after overwriting it above, and increment its
  // value.
  buffer.meta.slot = slot + 1;
}

// Chunked radix partitioning with software write-combining.
//
// See the Rust module for details.
template <typename K, typename V, typename M>
void cpu_chunked_radix_partition_swwc(RadixPartitionArgs &args) {
  constexpr size_t tuples_per_buffer =
      WriteCombineBuffer<Tuple<K, V>, SWWC_BUFFER_SIZE>::tuples_per_buffer();

  assert(reinterpret_cast<size_t>(args.write_combine_buffer) % 64UL == 0 &&
         "512-bit intrinsics require 64-byte alignment");
  assert(args.padding_length % tuples_per_buffer == 0 &&
         "Padding must be a multiple of the buffer length");

  auto join_attr_data =
      static_cast<const K *const __restrict__>(args.join_attr_data);
  auto payload_attr_data =
      static_cast<const V *const __restrict__>(args.payload_attr_data);
  auto partitioned_relation =
      static_cast<Tuple<K, V> *const __restrict__>(args.partitioned_relation);
  auto buffers = static_cast<
      WriteCombineBuffer<Tuple<K, V>, SWWC_BUFFER_SIZE> *const __restrict__>(
      args.write_combine_buffer);

  const size_t fanout = 1UL << args.radix_bits;
  const M mask = static_cast<M>(fanout - 1UL);
  const size_t partitioned_data_offset = args.partition_offsets[0];

  // Load partition offsets.
  for (size_t i = 0; i < fanout; ++i) {
    buffers[i].meta.slot = args.partition_offsets[i] - partitioned_data_offset;
  }

  // Partition into software write combine buffers.
#pragma GCC unroll 16
  for (size_t i = 0; i < args.data_length; ++i) {
    K key = join_attr_data[i];
    V pay = payload_attr_data[i];

    M p_index = key_to_partition(key, mask, 0);

    buffer_tuple<K, V, M>(partitioned_relation, buffers, p_index, key, pay);
  }

  // Flush remainders of all buffers.
  for (size_t i = 0; i < fanout; ++i) {
    size_t slot = buffers[i].meta.slot;
    size_t remaining = slot % tuples_per_buffer;

    for (size_t j = slot - remaining, k = 0; k < remaining; ++j, ++k) {
      partitioned_relation[j] = buffers[i].tuples.data[k];
    }
  }
}

#if defined(__ALTIVEC__)
// Chunked radix partitioning with software write-combining and SIMD
// optimizations.
//
// See the Rust module for details.
template <typename K, typename V, typename M>
void cpu_chunked_radix_partition_swwc_simd(RadixPartitionArgs &args) {
  constexpr size_t tuples_per_buffer =
      WriteCombineBuffer<Tuple<K, V>, SWWC_BUFFER_SIZE>::tuples_per_buffer();
  constexpr size_t vec_len = sizeof(vector int) / sizeof(K);

  assert(reinterpret_cast<size_t>(args.write_combine_buffer) %
                 SWWC_BUFFER_SIZE ==
             0 &&
         "SWWC buffer not sufficiently aligned");
  assert(args.padding_length % tuples_per_buffer == 0 &&
         "Padding must be a multiple of the buffer length");

  auto join_attr_data =
      static_cast<const K *const __restrict__>(args.join_attr_data);
  auto payload_attr_data =
      static_cast<const V *const __restrict__>(args.payload_attr_data);
  auto partitioned_relation =
      static_cast<Tuple<K, V> *const __restrict__>(args.partitioned_relation);
  auto buffers = static_cast<
      WriteCombineBuffer<Tuple<K, V>, SWWC_BUFFER_SIZE> *const __restrict__>(
      args.write_combine_buffer);

  assert(((size_t)join_attr_data) % vec_len == 0U &&
         "Key column should be aligned to ALIGN_BYTES for best performance");
  assert(
      ((size_t)payload_attr_data) % vec_len == 0U &&
      "Payload column should be aligned to ALIGN_BYTES for best performance");

  const size_t fanout = 1UL << args.radix_bits;
  const M mask = static_cast<M>(fanout - 1UL);
  const size_t partitioned_data_offset = args.partition_offsets[0];

  const vector M mask_vsx = vec_splats(mask);
  const vector M ignore_bits_vsx = vec_splats(static_cast<M>(0U));
  size_t i;

  // Load partition offsets
  for (size_t i = 0; i < fanout; ++i) {
    buffers[i].meta.slot = args.partition_offsets[i] - partitioned_data_offset;
  }

  // Partition into software write combine buffers
  i = 0;
  if (args.data_length > vec_len * 4) {
    for (; i < (args.data_length - vec_len * 4); i += vec_len * 4) {
      const K *const key_base = &join_attr_data[i];
      const V *const pay_base = &payload_attr_data[i];

      vector K key0 = vec_ld(0, key_base);
      vector K key1 = vec_ld(16, key_base);
      vector K key2 = vec_ld(32, key_base);
      vector K key3 = vec_ld(48, key_base);

      vector V pay0 = vec_ld(0, pay_base);
      vector V pay1 = vec_ld(16, pay_base);
      vector V pay2 = vec_ld(32, pay_base);
      vector V pay3 = vec_ld(48, pay_base);

      vector M p_index0 =
          key_to_partition_simd(key0, mask_vsx, ignore_bits_vsx);
      vector M p_index1 =
          key_to_partition_simd(key1, mask_vsx, ignore_bits_vsx);
      vector M p_index2 =
          key_to_partition_simd(key2, mask_vsx, ignore_bits_vsx);
      vector M p_index3 =
          key_to_partition_simd(key3, mask_vsx, ignore_bits_vsx);

      // FIXME: check if GCC unrolls this loop, otherwise force unroll
      for (uint32_t v = 0; v < vec_len; ++v) {
        buffer_tuple<K, V, M>(partitioned_relation, buffers, p_index0[v],
                              key0[v], pay0[v]);
        buffer_tuple<K, V, M>(partitioned_relation, buffers, p_index1[v],
                              key1[v], pay1[v]);
        buffer_tuple<K, V, M>(partitioned_relation, buffers, p_index2[v],
                              key2[v], pay2[v]);
        buffer_tuple<K, V, M>(partitioned_relation, buffers, p_index3[v],
                              key3[v], pay3[v]);
      }
    }
  }
  for (; i < args.data_length; ++i) {
    K key = join_attr_data[i];
    V payload = payload_attr_data[i];

    M p_index = key_to_partition(key, mask, 0);
    buffer_tuple<K, V, M>(partitioned_relation, buffers, p_index, key, payload);
  }

  // Flush remainders of all buffers.
  for (size_t i = 0; i < fanout; ++i) {
    size_t slot = buffers[i].meta.slot;
    size_t remaining = slot % tuples_per_buffer;

    for (size_t j = slot - remaining, k = 0; k < remaining; ++j, ++k) {
      partitioned_relation[j] = buffers[i].tuples.data[k];
    }
  }
}
#endif /* defined(__ALTIVEC__) */

// Exports the the size of all SWWC buffers.
extern "C" size_t cpu_swwc_buffer_bytes() { return SWWC_BUFFER_SIZE; }

// Exports the prefix sum function for 4-byte keys.
extern "C" void cpu_chunked_prefix_sum_int32(PrefixSumArgs *const args,
                                             uint32_t const chunk_id,
                                             uint32_t const num_chunks) {
  cpu_chunked_prefix_sum<int, unsigned>(*args, chunk_id, num_chunks);
}

// Exports the prefix sum function for 8-byte keys.
extern "C" void cpu_chunked_prefix_sum_int64(PrefixSumArgs *const args,
                                             uint32_t const chunk_id,
                                             uint32_t const num_chunks) {
  cpu_chunked_prefix_sum<long long, unsigned long long>(*args, chunk_id,
                                                        num_chunks);
}

#if defined(__ALTIVEC__)
// Exports the SIMD prefix sum function for 4-byte keys.
extern "C" void cpu_chunked_prefix_sum_simd_int32(PrefixSumArgs *const args,
                                                  uint32_t const chunk_id,
                                                  uint32_t const num_chunks) {
  cpu_chunked_prefix_sum_simd<int, unsigned>(*args, chunk_id, num_chunks);
}

// Exports the SIMD prefix sum function for 8-byte keys.
extern "C" void cpu_chunked_prefix_sum_simd_int64(PrefixSumArgs *const args,
                                                  uint32_t const chunk_id,
                                                  uint32_t const num_chunks) {
  cpu_chunked_prefix_sum_simd<long long, unsigned long long>(*args, chunk_id,
                                                             num_chunks);
}
#else  // define dummy function symbols
// Exports the SIMD prefix sum function for 4-byte keys.
extern "C" void cpu_chunked_prefix_sum_simd_int32(PrefixSumArgs *const,
                                                  uint32_t const,
                                                  uint32_t const) {}

// Exports the SIMD prefix sum function for 8-byte keys.
extern "C" void cpu_chunked_prefix_sum_simd_int64(PrefixSumArgs *const,
                                                  uint32_t const,
                                                  uint32_t const) {}
#endif /* defined(__ALTIVEC__) */

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" void cpu_chunked_radix_partition_int32_int32(
    RadixPartitionArgs *args) {
  cpu_chunked_radix_partition<int, int, unsigned>(*args);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" void cpu_chunked_radix_partition_int64_int64(
    RadixPartitionArgs *args) {
  cpu_chunked_radix_partition<long long, long long, unsigned long long>(*args);
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" void cpu_chunked_radix_partition_swwc_int32_int32(
    RadixPartitionArgs *args) {
  cpu_chunked_radix_partition_swwc<int, int, unsigned>(*args);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" void cpu_chunked_radix_partition_swwc_int64_int64(
    RadixPartitionArgs *args) {
  cpu_chunked_radix_partition_swwc<long long, long long, unsigned long long>(
      *args);
}

#if defined(__ALTIVEC__)
// Exports the partitioning function for 8-byte key/value tuples.
extern "C" void cpu_chunked_radix_partition_swwc_simd_int32_int32(
    RadixPartitionArgs *args) {
  cpu_chunked_radix_partition_swwc_simd<int, int, unsigned>(*args);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" void cpu_chunked_radix_partition_swwc_simd_int64_int64(
    RadixPartitionArgs *args) {
  cpu_chunked_radix_partition_swwc_simd<long long, long long,
                                        unsigned long long>(*args);
}
#else  // define dummy function symbols
extern "C" void cpu_chunked_radix_partition_swwc_simd_int32_int32(
    RadixPartitionArgs * /* args */) {}
extern "C" void cpu_chunked_radix_partition_swwc_simd_int64_int64(
    RadixPartitionArgs * /* args */) {}
#endif /* defined(__ALTIVEC__) */
