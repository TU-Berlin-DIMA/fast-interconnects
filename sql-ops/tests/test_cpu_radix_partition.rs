/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2021 Clemens Lutz, German Research Center for Artificial
 * Intelligence
 * Author: Clemens Lutz, DFKI GmbH <clemens.lutz@dfki.de>
 */

pub mod radix_partition;

use datagen::relation::UniformRelation;
use itertools::izip;
use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemType};
use radix_partition::{tuple_loss_or_duplicates, verify_partitions};
use rustacuda::memory::DeviceCopy;
use sql_ops::partition::cpu_radix_partition::{
    CpuHistogramAlgorithm, CpuRadixPartitionAlgorithm, CpuRadixPartitionable, CpuRadixPartitioner,
};
use sql_ops::partition::{
    PartitionOffsets, PartitionedRelation, RadixBits, RadixPartitionInputChunkable, RadixPass,
    Tuple,
};
use std::error::Error;
use std::mem::size_of;
use std::result::Result;

fn run_cpu_partitioning<T, KeyGenFn, PayGenFn, ValidatorFn>(
    tuples: usize,
    key_gen: Box<KeyGenFn>,
    pay_gen: Box<PayGenFn>,
    prefix_sum_algorithm: CpuHistogramAlgorithm,
    partition_algorithm: CpuRadixPartitionAlgorithm,
    radix_bits: RadixBits,
    threads: u32,
    mut validator: Box<ValidatorFn>,
) -> Result<(), Box<dyn Error>>
where
    T: Clone + Default + DeviceCopy + CpuRadixPartitionable,
    KeyGenFn: FnOnce(&mut [T]) -> Result<(), Box<dyn Error>>,
    PayGenFn: FnOnce(&mut [T]) -> Result<(), Box<dyn Error>>,
    ValidatorFn: FnMut(
        RadixPass,
        &RadixBits,
        &[T],
        &[T],
        &PartitionedRelation<Tuple<T, T>>,
        Option<u32>,
    ) -> Result<(), Box<dyn Error>>,
{
    let mut data_key: Vec<T> = vec![T::default(); tuples];
    let mut data_pay: Vec<T> = vec![T::default(); tuples];

    key_gen(data_key.as_mut_slice())?;
    pay_gen(data_pay.as_mut_slice())?;

    let mut partition_offsets = PartitionOffsets::new(
        prefix_sum_algorithm.into(),
        threads,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        Allocator::mem_alloc_fn(MemType::SysMem),
    );

    let mut partitioned_relation = PartitionedRelation::new(
        tuples,
        prefix_sum_algorithm.into(),
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        threads,
        Allocator::mem_alloc_fn(MemType::SysMem),
        Allocator::mem_alloc_fn(MemType::SysMem),
    );

    unsafe {
        partitioned_relation
            .as_raw_relation_mut_slice()?
            .iter_mut()
            .for_each(|x| *x = Tuple::default());
    }

    let mut partitioner = CpuRadixPartitioner::new(
        prefix_sum_algorithm,
        partition_algorithm,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        DerefMemType::SysMem,
    );

    let data_key_chunks = data_key.as_slice().input_chunks::<T>(threads)?;

    for (key_chunk, offsets_chunk) in
        izip!(data_key_chunks.into_iter(), partition_offsets.chunks_mut(),)
    {
        partitioner.prefix_sum(key_chunk, offsets_chunk)?;
    }

    let data_key_chunks = data_key.as_slice().input_chunks::<T>(threads)?;
    let data_pay_chunks = data_pay.as_slice().input_chunks::<T>(threads)?;

    for (key_chunk, pay_chunk, offsets_chunk, partitioned_chunk) in izip!(
        data_key_chunks.into_iter(),
        data_pay_chunks.into_iter(),
        partition_offsets.chunks_mut(),
        partitioned_relation.chunks_mut()
    ) {
        partitioner.partition(key_chunk, pay_chunk, offsets_chunk, partitioned_chunk)?;
    }

    validator(
        RadixPass::First,
        &radix_bits,
        data_key.as_slice(),
        data_pay.as_slice(),
        &partitioned_relation,
        None,
    )?;

    Ok(())
}

// ======================== Chunked NC ========================

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_i32_small_data() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        15,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(4),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_i32_small_data() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        15,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(4),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_i32_0_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(0),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_i32_0_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(0),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_i32_1_bit() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(1),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_i32_1_bit() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(1),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(12),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(12),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_i32_13_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(13),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_i32_13_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(13),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_i32_14_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(14),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_i32_14_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(14),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_i32_15_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(15),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_i32_15_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(15),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_i32_16_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(16),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_i32_16_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(16),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_i32_17_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(17),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_i32_17_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(17),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_i32_less_tuples_than_partitions(
) -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 5) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(17),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_i32_less_tuples_than_partitions() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 5) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(17),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_i32_non_power_2_data_len() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 10) / size_of::<i32>() - 7,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_i32_non_power_2_data_len() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 10) / size_of::<i32>() - 7,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_i64_17_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (64 << 20) / size_of::<i64>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i64>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i64>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(17),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

// #[test]
// fn cpu_verify_partitions_chunked_i64_17_bits() -> Result<(), Box<dyn Error>> {
//     run_cpu_partitioning(
//         (64 << 20) / size_of::<i64>(),
//         Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i64>(keys, 1..=(32 << 20))?)),
//         Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i64>(pays, 1..=10000)?)),
//         CpuHistogramAlgorithm::Chunked,
//         CpuRadixPartitionAlgorithm::NC,
//         RadixBits::from(17),
//         4,
//         Box::new(&verify_partitions),
//     )
// }

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_simd_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(12),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_simd_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::NC,
        RadixBits::from(12),
        4,
        Box::new(&verify_partitions),
    )
}

// ======================== Chunked SWWC ========================

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_i32_small_data() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        15,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(4),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_swwc_i32_small_data() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        15,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(4),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_i32_0_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(0),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_swwc_i32_0_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(0),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_i32_1_bit() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(1),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_swwc_i32_1_bit() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(1),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(2),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_swwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(2),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(12),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_swwc_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(12),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_i32_13_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(13),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_swwc_i32_13_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(13),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_i32_14_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(14),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_swwc_i32_14_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(14),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_i32_15_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(15),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_swwc_i32_15_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(15),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_i32_16_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(16),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_swwc_i32_16_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(16),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_i32_17_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(17),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_swwc_i32_17_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(17),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_i32_less_tuples_than_partitions(
) -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 5) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(17),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_swwc_i32_less_tuples_than_partitions() -> Result<(), Box<dyn Error>>
{
    run_cpu_partitioning(
        (32 << 5) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(17),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_i32_non_power_2_data_len() -> Result<(), Box<dyn Error>>
{
    run_cpu_partitioning(
        (32 << 10) / size_of::<i32>() - 7,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(10),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn cpu_verify_partitions_chunked_swwc_i32_non_power_2_data_len() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 10) / size_of::<i32>() - 7,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(10),
        4,
        Box::new(&verify_partitions),
    )
}

#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_i64_17_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (64 << 20) / size_of::<i64>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i64>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i64>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::Swwc,
        RadixBits::from(17),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

// ======================== Chunked SWWC SIMD ========================

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_simd_i32_small_data() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        15,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(4),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_swwc_simd_i32_small_data() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        15,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(4),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_simd_i32_0_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(0),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_swwc_simd_i32_0_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(0),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_simd_i32_1_bit() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(1),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_swwc_simd_i32_1_bit() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(1),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_simd_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(2),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_swwc_simd_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(2),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_simd_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(12),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_swwc_simd_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(12),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_simd_i32_13_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(13),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_swwc_simd_i32_13_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(13),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_simd_i32_14_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(14),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_swwc_simd_i32_14_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(14),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_simd_i32_15_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(15),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_swwc_simd_i32_15_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(15),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_simd_i32_16_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(16),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_swwc_simd_i32_16_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(16),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_simd_i32_17_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(17),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_swwc_simd_i32_17_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(17),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_simd_i32_less_tuples_than_partitions(
) -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 5) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(17),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_swwc_simd_i32_less_tuples_than_partitions(
) -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 5) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(17),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_simd_i32_non_power_2_data_len(
) -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 10) / size_of::<i32>() - 7,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(10),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_swwc_simd_i32_non_power_2_data_len() -> Result<(), Box<dyn Error>>
{
    run_cpu_partitioning(
        (32 << 10) / size_of::<i32>() - 7,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(10),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_swwc_simd_i64_17_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (64 << 20) / size_of::<i64>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i64>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i64>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::Chunked,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(17),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

// ======================== Chunked SIMD SWWC SIMD ========================

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_simd_swwc_simd_i32_small_data() -> Result<(), Box<dyn Error>>
{
    run_cpu_partitioning(
        15,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(4),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_simd_swwc_simd_i32_small_data() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        15,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(4),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_simd_swwc_simd_i32_0_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(0),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_simd_swwc_simd_i32_0_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(0),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_simd_swwc_simd_i32_1_bit() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(1),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_simd_swwc_simd_i32_1_bit() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(1),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_simd_swwc_simd_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(2),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_simd_swwc_simd_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(2),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_simd_swwc_simd_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(12),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_simd_swwc_simd_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(12),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_simd_swwc_simd_i32_13_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(13),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_simd_swwc_simd_i32_13_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(13),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_simd_swwc_simd_i32_14_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(14),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_simd_swwc_simd_i32_14_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(14),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_simd_swwc_simd_i32_15_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(15),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_simd_swwc_simd_i32_15_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(15),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_simd_swwc_simd_i32_16_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(16),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_simd_swwc_simd_i32_16_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(16),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_simd_swwc_simd_i32_17_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(17),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_simd_swwc_simd_i32_17_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 20) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(17),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_simd_swwc_simd_i32_less_tuples_than_partitions(
) -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 5) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(17),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_simd_swwc_simd_i32_less_tuples_than_partitions(
) -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 5) / size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(17),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_simd_swwc_simd_i32_non_power_2_data_len(
) -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 10) / size_of::<i32>() - 7,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i32>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(10),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_verify_partitions_chunked_simd_swwc_simd_i32_non_power_2_data_len(
) -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (32 << 10) / size_of::<i32>() - 7,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr::<i32>(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i32>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(10),
        4,
        Box::new(&verify_partitions),
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn cpu_tuple_loss_or_duplicates_chunked_simd_swwc_simd_i64_17_bits() -> Result<(), Box<dyn Error>> {
    run_cpu_partitioning(
        (64 << 20) / size_of::<i64>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key::<i64>(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr::<i64>(pays, 1..=10000)?)),
        CpuHistogramAlgorithm::ChunkedSimd,
        CpuRadixPartitionAlgorithm::SwwcSimd,
        RadixBits::from(17),
        4,
        Box::new(&tuple_loss_or_duplicates),
    )
}
