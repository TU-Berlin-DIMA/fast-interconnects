/*
 * Copyright 2019 Clemens Lutz, German Research Center for Artificial Intelligence
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
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

//! Radix partition operators for CPU and GPU.
//!
//! # Overview
//!
//! CPU and GPU partitioning operators are compatible. The devices can cooperate
//! to partition a relation in parallel. Note that this only holds for equivalent
//! operator algorithms, i.e., `chunked_radix_partition_swwc` cannot be combined
//! with `chunked_radix_partition`.
//!
//! Provided is one radix partitioning algorithm, chunked radix partitioning with
//! software write-combine buffering (SWWC). This algorithm is described by Schuh
//! et al. in Section 6 of "An Experimental Comparison of Thirteen Relational
//! Equi-Joins in Main Memory".
//!
//! # Thread-safety
//!
//! The radix partitioning operators are designed to be thread-safe. Although the
//! input data can be shared between threads, threads should typically work on
//! disjuct input partitions for correct results. In contrast, each thread must
//! have exclusive ownership of its output and intermediate state buffers.
//!
//! # Padding
//!
//! It is important to note that partitions are padded to, at minimum, the
//! cache-line size. This is necessary for SWWC buffering because cache-lines
//! are written back to memory as a whole. However, partition offsets are not
//! naturally aligned, because partitions can have any size. Therefore,
//! all partitions are padded in front by, at minimum, the length of a cache-line.
//! The cache-alignment is also necessary for non-temporal SIMD writes, which
//! must be aligned to their SIMD vector length.
//!
//! # Copyright notes
//!
//! The C/C++ CPU code is based on [code kindly published by Cagri Balkesen and
//! Claude Barthels][mchj]. As such, we adhere to their copyright and license
//! (MIT) in derived code. Modifications are licensed under Apache License 2.0.
//!
//! [mchj]: https://www.systems.ethz.ch/sites/default/files/file/PublishedCode/multicore-distributed-hashjoins-0_1.zip

use crate::error::{ErrorKind, Result};
use numa_gpu::runtime::allocator::DerefMemAllocFn;
use numa_gpu::runtime::memory::DerefMem;
use rustacuda::memory::DeviceCopy;
use std::ffi::c_void;
use std::ops::{Index, IndexMut};
use std::{mem, ptr};

extern "C" {
    fn cpu_swwc_buffer_bytes() -> usize;
    fn cpu_chunked_radix_partition_int32_int32(args: *mut RadixPartitionArgs);
    fn cpu_chunked_radix_partition_int64_int64(args: *mut RadixPartitionArgs);
    fn cpu_chunked_radix_partition_swwc_int32_int32(args: *mut RadixPartitionArgs);
    fn cpu_chunked_radix_partition_swwc_int64_int64(args: *mut RadixPartitionArgs);
}

/// Compute the fanout (i.e., the number of partitions) from the number of radix
/// bits.
fn fanout(radix_bits: u32) -> usize {
    1 << radix_bits
}

/// Arguments to the C/C++ partitioning function.
///
/// Note that the struct's layout must be kept in sync with its counterpart in
/// C/C++.
#[repr(C)]
#[derive(Debug)]
struct RadixPartitionArgs {
    // Inputs
    partition_attr_data: *const c_void,
    payload_attr_data: *const c_void,
    data_len: usize,
    padding_len: usize,
    radix_bits: u32,

    // State
    tmp_partition_offsets: *mut u64,
    write_combine_buffer: *mut c_void,

    // Outputs
    partition_offsets: *mut u64,
    partitioned_relation: *mut c_void,
}

/// A key-value tuple.
///
/// The partitioned relation is stored as a collection of `Tuple<K, V>`.
///
/// Note that the struct's layout must be kept in sync with its counterpart in
/// C/C++.
#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct Tuple<Key: Sized, Value: Sized> {
    key: Key,
    value: Value,
}

unsafe impl<K, V> DeviceCopy for Tuple<K, V>
where
    K: DeviceCopy,
    V: DeviceCopy,
{
}

/// A radix-partitioned relation, optionally with padding in front of each
/// partition.
///
/// # Invariants
///
/// The `radix_bits` must match in `WriteCombineBuffer` and `CpuRadixPartitioner`.
#[derive(Debug)]
pub struct PartitionedRelation<T: DeviceCopy> {
    relation: DerefMem<T>,
    offsets: DerefMem<u64>,
    radix_bits: u32,
}

impl<T: DeviceCopy> PartitionedRelation<T> {
    /// Creates a new partitioned relation, and automatically includes the
    /// necessary padding and metadata.
    pub fn new(
        len: usize,
        radix_bits: u32,
        partition_alloc_fn: DerefMemAllocFn<T>,
        offsets_alloc_fn: DerefMemAllocFn<u64>,
    ) -> Self {
        let padding_len = WriteCombineBuffer::tuples_per_buffer::<T>();
        let num_partitions = fanout(radix_bits);
        let relation_len = len + num_partitions * padding_len;

        let relation = partition_alloc_fn(relation_len);
        let offsets = offsets_alloc_fn(num_partitions);

        Self {
            relation,
            offsets,
            radix_bits,
        }
    }

    /// Returns the total number of elements in the relation (excluding padding).
    pub fn len(&self) -> usize {
        let num_partitions = fanout(self.radix_bits);

        self.relation.len() - num_partitions * self.padding_len()
    }

    /// Returns the number of partitions.
    pub fn partitions(&self) -> usize {
        fanout(self.radix_bits)
    }

    /// Returns the number of padding elements per partition.
    fn padding_len(&self) -> usize {
        WriteCombineBuffer::tuples_per_buffer::<T>()
    }
}

/// Returns the specified partition as a subslice of the relation.
impl<T: DeviceCopy> Index<usize> for PartitionedRelation<T> {
    type Output = [T];

    fn index(&self, i: usize) -> &Self::Output {
        let begin = self.offsets[i] as usize;
        let end = if i + 1 < self.offsets.len() {
            self.offsets[i + 1] as usize - self.padding_len()
        } else {
            self.relation.len()
        };

        &self.relation[begin..end]
    }
}

/// Returns the specified partition as a mutable subslice of the relation.
impl<T: DeviceCopy> IndexMut<usize> for PartitionedRelation<T> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        let begin = self.offsets[i] as usize;
        let end = if i + 1 < self.offsets.len() {
            self.offsets[i + 1] as usize - self.padding_len()
        } else {
            self.relation.len()
        };

        &mut self.relation[begin..end]
    }
}

/// A set of buffers used for software write-combining.
///
/// The original code by Cagri Balkesen allocates these SWWC buffers on the stack.
/// In contrast, this implementation allocates the SWWC buffers on the heap,
/// because new CPUs have very large L3 caches (> 100 MB). Allocating on the
/// stack risks a stack overflow on these CPUs. This can occur when using a large
/// fanout (i.e., a high number of radix bits).
///
/// # Invariants
///
/// * The `radix_bits` must match in `PartitionedRelation` and `CpuRadixPartitioner`.
///
/// * The backing memory must be aligned to the cache-line size of the machine.
///   Hint: `DerefMem::Numa` alignes to the page size, which is a multiple of
///   the cache-line size
#[derive(Debug)]
struct WriteCombineBuffer {
    buffers: DerefMem<u64>,
}

impl WriteCombineBuffer {
    /// Creates a new set of SWWC buffers.
    fn new(radix_bits: u32, alloc_fn: DerefMemAllocFn<u64>) -> Self {
        let buffer_bytes = unsafe { cpu_swwc_buffer_bytes() };
        let bytes = buffer_bytes * fanout(radix_bits);
        let buffers = alloc_fn(bytes / mem::size_of::<u64>());

        Self { buffers }
    }

    /// Computes the number of tuples per SWWC buffer.
    ///
    /// Note that `WriteCombineBuffer` contains one SWWC buffer per
    fn tuples_per_buffer<T: Sized>() -> usize {
        let buffer_bytes = unsafe { cpu_swwc_buffer_bytes() };
        buffer_bytes / mem::size_of::<T>()
    }
}

/// Specifies that the implementing type can be used as partitioning key in
/// `CpuRadixPartitioner`.
///
/// `CpuRadixPartitionable` is a trait for which specialized implementations
/// exist for each implementing type (currently i32 and i64). Specialization is
/// necessary because each type requires a different C++ function to be called.
///
/// See `CudaHashJoinable` for more details on the design decision.
pub trait CpuRadixPartitionable: Sized + DeviceCopy {
    fn partition_impl(
        rp: &mut CpuRadixPartitioner,
        partition_attr: &[Self],
        payload_attr: &[Self],
        partitioned_relation: &mut PartitionedRelation<Tuple<Self, Self>>,
    ) -> Result<()>;
}

/// Specifies the radix partition algorithm.
#[derive(Copy, Clone, Debug)]
pub enum CpuRadixPartitionAlgorithm {
    /// Chunked radix partition.
    Chunked,

    /// Chunked radix partition with software write-combining.
    ChunkedSwwc,
}

/// Mutable internal state of the partition functions.
///
/// The state is reusable as long as the radix bits remain unchanged between
/// runs.
#[derive(Debug)]
enum RadixPartitionState {
    Chunked(DerefMem<u64>),
    ChunkedSwwc(WriteCombineBuffer),
}

/// A CPU radix partitioner that provides partitioning functions.
#[derive(Debug)]
pub struct CpuRadixPartitioner {
    radix_bits: u32,
    state: RadixPartitionState,
}

impl CpuRadixPartitioner {
    /// Creates a new CPU radix partitioner.
    pub fn new(
        algorithm: CpuRadixPartitionAlgorithm,
        radix_bits: u32,
        alloc_fn: DerefMemAllocFn<u64>,
    ) -> Self {
        let num_partitions = fanout(radix_bits);

        let state = match algorithm {
            CpuRadixPartitionAlgorithm::Chunked => {
                RadixPartitionState::Chunked(alloc_fn(num_partitions))
            }
            CpuRadixPartitionAlgorithm::ChunkedSwwc => {
                RadixPartitionState::ChunkedSwwc(WriteCombineBuffer::new(radix_bits, alloc_fn))
            }
        };

        Self { radix_bits, state }
    }

    /// Radix-partitions a relation by its key attribute.
    ///
    /// See the module-level documentation for details on the algorithm.
    pub fn partition<T: DeviceCopy + CpuRadixPartitionable>(
        &mut self,
        partition_attr: &[T],
        payload_attr: &[T],
        partitioned_relation: &mut PartitionedRelation<Tuple<T, T>>,
    ) -> Result<()> {
        T::partition_impl(self, partition_attr, payload_attr, partitioned_relation)
    }
}

macro_rules! impl_cpu_radix_partition_for_type {
    ($Type:ty, $Suffix:expr) => {
        impl CpuRadixPartitionable for $Type {
            paste::item! {
                fn partition_impl(
                    rp: &mut CpuRadixPartitioner,
                    partition_attr: &[$Type],
                    payload_attr: &[$Type],
                    partitioned_relation: &mut PartitionedRelation<Tuple<$Type, $Type>>,
                    ) -> Result<()>
                {
                    if partition_attr.len() != payload_attr.len() {
                        Err(ErrorKind::InvalidArgument(
                                "Partition and payload attributes have different sizes"
                                .to_string()
                            ))?;
                    }
                    if partitioned_relation.radix_bits != rp.radix_bits {
                        Err(ErrorKind::InvalidArgument(
                                "PartitionedRelation has mismatching radix bits"
                                .to_string()
                            ))?;
                    }

                    let data_len = partition_attr.len();
                    let (partition_fn, tmp_partition_offsets, write_combine_buffer):
                        (
                            unsafe extern "C" fn(*mut RadixPartitionArgs),
                            *mut u64,
                            *mut c_void,
                        ) = match rp.state
                    {
                        RadixPartitionState::Chunked(ref mut offsets) =>
                            (
                                [<cpu_chunked_radix_partition_ $Suffix _ $Suffix>],
                                offsets.as_mut_ptr(),
                                ptr::null_mut(),
                            ),
                        RadixPartitionState::ChunkedSwwc(ref mut swwc) =>
                            (
                                [<cpu_chunked_radix_partition_swwc_ $Suffix _ $Suffix>],
                                ptr::null_mut(),
                                swwc.buffers.as_mut_slice().as_mut_ptr() as *mut c_void,
                            ),
                    };

                    let mut args = RadixPartitionArgs {
                        partition_attr_data: partition_attr.as_ptr() as *const c_void,
                        payload_attr_data: payload_attr.as_ptr() as *const c_void,
                        data_len,
                        padding_len: partitioned_relation.padding_len(),
                        radix_bits: rp.radix_bits,
                        tmp_partition_offsets,
                        write_combine_buffer,
                        partition_offsets: partitioned_relation.offsets.as_mut_ptr(),
                        partitioned_relation: partitioned_relation.relation
                            .as_mut_ptr() as *mut c_void,
                    };

                    unsafe {
                        partition_fn(
                            &mut args as *mut RadixPartitionArgs
                        );
                    }

                    Ok(())
                }
            }
        }
    };
}

impl_cpu_radix_partition_for_type!(i32, int32);
impl_cpu_radix_partition_for_type!(i64, int64);

#[cfg(test)]
mod tests {
    use super::*;
    use datagen::relation::UniformRelation;
    use numa_gpu::runtime::allocator::{Allocator, DerefMemType};
    use std::collections::hash_map::{Entry, HashMap};
    use std::error::Error;
    use std::iter;
    use std::mem::size_of;
    use std::ops::RangeInclusive;
    use std::result::Result;

    macro_rules! test_cpu_seq {
        ($fn_suffix:ident, $type:ty, $tuples:expr, $key_range:expr, $algorithm:expr, $radix_bits:expr) => {
            paste::item! {

                #[test]
                fn [<cpu_tuple_loss_or_duplicates_ $fn_suffix>]() -> Result<(), Box<dyn Error>> {
                    const PAYLOAD_RANGE: RangeInclusive<usize> = 1..=10000;
                    const NUMA_NODE: u16 = 0;

                    let mut data_key: Vec<$type> = vec![0; $tuples];
                    let mut data_pay: Vec<$type> = vec![0; $tuples];

                    UniformRelation::gen_primary_key(&mut data_key)?;
                    UniformRelation::gen_attr(&mut data_pay, PAYLOAD_RANGE)?;

                    let mut original_tuples: HashMap<_, _> = data_key
                        .iter()
                        .zip(data_pay.iter().zip(std::iter::repeat(0)))
                        .collect();

                    let mut partitioned_relation = PartitionedRelation::new(
                        $tuples,
                        $radix_bits,
                        Allocator::deref_mem_alloc_fn(DerefMemType::NumaMem(NUMA_NODE)),
                        Allocator::deref_mem_alloc_fn(DerefMemType::NumaMem(NUMA_NODE)),
                    );

                    let mut partitioner = CpuRadixPartitioner::new(
                        $algorithm,
                        $radix_bits,
                        Allocator::deref_mem_alloc_fn::<u64>(DerefMemType::NumaMem(NUMA_NODE))
                    );

                    partitioner.partition(
                        &data_key,
                        &data_pay,
                        &mut partitioned_relation,
                    )?;

                    partitioned_relation.relation
                        .as_slice()
                        .iter()
                        .for_each(|Tuple { key, value }| {
                            let entry = original_tuples.entry(key);
                            match entry {
                                entry @ Entry::Occupied(_) => {
                                    let key = *entry.key();
                                    entry.and_modify(|(original_value, counter)| {
                                        assert_eq!(
                                            value, *original_value,
                                            "Invalid payload: {}; expected: {}",
                                            value, *original_value
                                            );
                                        assert_eq!(*counter, 0, "Duplicate key: {}", key);
                                        *counter = *counter + 1;
                                    });
                                }
                                entry @ Entry::Vacant(_) => {
                                    // skip padding entries
                                    if **entry.key() != 0 {
                                        assert!(false, "Invalid key: {}", entry.key());
                                    }
                                }
                            };
                        });

                    original_tuples.iter().for_each(|(&key, &(_, counter))| {
                        assert_eq!(
                            counter, 1,
                            "Key {} occurs {} times; expected exactly once",
                            key, counter
                        );
                    });

                    Ok(())
                }

                #[test]
                fn [<cpu_verify_partitions_ $fn_suffix>]() -> Result<(), Box<dyn Error>> {
                    const PAYLOAD_RANGE: RangeInclusive<usize> = 1..=10000;
                    const NUMA_NODE: u16 = 0;

                    let mut data_key: Vec<$type> = vec![0; $tuples];
                    let mut data_pay: Vec<$type> = vec![0; $tuples];

                    UniformRelation::gen_attr(&mut data_key, $key_range)?;
                    UniformRelation::gen_attr(&mut data_pay, PAYLOAD_RANGE)?;

                    let mut partitioned_relation = PartitionedRelation::new(
                        $tuples,
                        $radix_bits,
                        Allocator::deref_mem_alloc_fn(DerefMemType::NumaMem(NUMA_NODE)),
                        Allocator::deref_mem_alloc_fn(DerefMemType::NumaMem(NUMA_NODE)),
                    );

                    let mut partitioner = CpuRadixPartitioner::new(
                        $algorithm,
                        $radix_bits,
                        Allocator::deref_mem_alloc_fn::<u64>(DerefMemType::NumaMem(NUMA_NODE))
                    );

                    partitioner.partition(
                        &data_key,
                        &data_pay,
                        &mut partitioned_relation,
                    )?;

                    let mask = fanout($radix_bits) - 1;
                    (0..partitioned_relation.partitions())
                        .flat_map(|i| {
                            iter::repeat(i)
                                .zip(partitioned_relation[i].iter())
                        })
                    .for_each(|(i, &tuple)| {
                        let dst_partition = (tuple.key) & mask as $type;
                        assert_eq!(
                            dst_partition, i as $type,
                            "Wrong partitioning detected: key {} in partition {}; expected partition {}",
                            tuple.key, i, dst_partition
                            );
                    });

                    Ok(())
                }
            }
        }
    }

    test_cpu_seq!(
        i32_small_data,
        i32,
        15,
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::Chunked,
        4
    );

    test_cpu_seq!(
        i32_1_bit,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::Chunked,
        1
    );

    test_cpu_seq!(
        i32_2_bits,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::Chunked,
        2
    );

    test_cpu_seq!(
        i32_12_bits,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::Chunked,
        12
    );

    test_cpu_seq!(
        i32_13_bits,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::Chunked,
        13
    );

    test_cpu_seq!(
        i32_14_bits,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::Chunked,
        14
    );

    test_cpu_seq!(
        i32_15_bits,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::Chunked,
        15
    );

    test_cpu_seq!(
        i32_16_bits,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::Chunked,
        16
    );

    test_cpu_seq!(
        i32_17_bits,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::Chunked,
        17
    );

    test_cpu_seq!(
        i32_less_tuples_than_partitions,
        i32,
        (32 << 5) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::Chunked,
        17
    );

    test_cpu_seq!(
        i32_non_power_2_data_len,
        i32,
        ((32 << 10) - 7) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::Chunked,
        10
    );

    test_cpu_seq!(
        i64_17_bits,
        i64,
        (64 << 20) / size_of::<i64>(),
        1..=(64 << 20),
        CpuRadixPartitionAlgorithm::Chunked,
        17
    );

    test_cpu_seq!(
        swwc_i32_small_data,
        i32,
        15,
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::ChunkedSwwc,
        4
    );

    test_cpu_seq!(
        swwc_i32_1_bit,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::ChunkedSwwc,
        1
    );

    test_cpu_seq!(
        swwc_i32_2_bits,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::ChunkedSwwc,
        2
    );

    test_cpu_seq!(
        swwc_i32_12_bits,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::ChunkedSwwc,
        12
    );

    test_cpu_seq!(
        swwc_i32_13_bits,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::ChunkedSwwc,
        13
    );

    test_cpu_seq!(
        swwc_i32_14_bits,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::ChunkedSwwc,
        14
    );

    test_cpu_seq!(
        swwc_i32_15_bits,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::ChunkedSwwc,
        15
    );

    test_cpu_seq!(
        swwc_i32_16_bits,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::ChunkedSwwc,
        16
    );

    test_cpu_seq!(
        swwc_i32_17_bits,
        i32,
        (32 << 20) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::ChunkedSwwc,
        17
    );

    test_cpu_seq!(
        swwc_i32_less_tuples_than_partitions,
        i32,
        (32 << 5) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::ChunkedSwwc,
        17
    );

    test_cpu_seq!(
        swwc_i32_non_power_2_data_len,
        i32,
        ((32 << 10) - 7) / size_of::<i32>(),
        1..=(32 << 20),
        CpuRadixPartitionAlgorithm::ChunkedSwwc,
        10
    );

    test_cpu_seq!(
        swwc_i64_17_bits,
        i64,
        (64 << 20) / size_of::<i64>(),
        1..=(64 << 20),
        CpuRadixPartitionAlgorithm::ChunkedSwwc,
        17
    );
}
