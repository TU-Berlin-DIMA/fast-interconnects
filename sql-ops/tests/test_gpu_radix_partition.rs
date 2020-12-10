/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2020 Clemens Lutz, German Research Center for Artificial
 * Intelligence
 * Author: Clemens Lutz, DFKI GmbH <clemens.lutz@dfki.de>
 */

use datagen::relation::UniformRelation;
use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemType};
use numa_gpu::runtime::memory::{LaunchableMem, Mem};
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::LockedBuffer;
use rustacuda::stream::{Stream, StreamFlags};
use sql_ops::partition::gpu_radix_partition::{
    GpuHistogramAlgorithm, GpuRadixPartitionAlgorithm, GpuRadixPartitioner,
    RadixPartitionInputChunkable,
};
use sql_ops::partition::{PartitionOffsets, PartitionedRelation, RadixBits, RadixPass, Tuple};
use std::collections::hash_map::{Entry, HashMap};
use std::error::Error;
use std::iter;
use std::mem;
use std::result::Result;

fn key_to_partition(key: i32, radix_bits: &RadixBits, radix_pass: RadixPass) -> u32 {
    let ignore_bits = radix_bits.pass_ignore_bits(radix_pass);
    let fanout = radix_bits.pass_fanout(radix_pass).unwrap();
    let mask = (fanout - 1) << ignore_bits;
    let partition = (key as u32 & mask) >> ignore_bits;
    partition
}

fn run_gpu_partitioning<KeyGenFn, PayGenFn, ValidatorFn>(
    tuples: usize,
    key_gen: Box<KeyGenFn>,
    pay_gen: Box<PayGenFn>,
    histogram_algorithm: GpuHistogramAlgorithm,
    partition_algorithm: GpuRadixPartitionAlgorithm,
    radix_bits: RadixBits,
    grid_size: GridSize,
    block_size: BlockSize,
    mut validator: Box<ValidatorFn>,
) -> Result<(), Box<dyn Error>>
where
    KeyGenFn: FnOnce(&mut [i32]) -> Result<(), Box<dyn Error>>,
    PayGenFn: FnOnce(&mut [i32]) -> Result<(), Box<dyn Error>>,
    ValidatorFn: FnMut(
        RadixPass,
        &RadixBits,
        &[i32],
        &[i32],
        &PartitionedRelation<Tuple<i32, i32>>,
        Option<u32>,
    ) -> Result<(), Box<dyn Error>>,
{
    const DMEM_BUFFER_BYTES: usize = 8 * 1024;

    let _context = rustacuda::quick_init()?;

    let mut data_key = Allocator::alloc_deref_mem::<i32>(DerefMemType::CudaPinnedMem, tuples);
    let mut data_pay = Allocator::alloc_deref_mem::<i32>(DerefMemType::CudaPinnedMem, tuples);

    key_gen(data_key.as_mut_slice())?;
    pay_gen(data_pay.as_mut_slice())?;

    // Ensure that the allocated memory is zeroed
    let alloc_fn = Box::new(|len: usize| {
        let mut mem = Allocator::alloc_deref_mem(DerefMemType::CudaUniMem, len);
        mem.iter_mut().for_each(|x| *x = Default::default());
        mem.into()
    });

    let mut partition_offsets = PartitionOffsets::new(
        histogram_algorithm,
        grid_size.x,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    let mut partitioned_relation = PartitionedRelation::new(
        tuples,
        histogram_algorithm,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        grid_size.x,
        alloc_fn,
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    let mut partitioner = GpuRadixPartitioner::new(
        histogram_algorithm,
        partition_algorithm,
        radix_bits.into(),
        &grid_size,
        &block_size,
        DMEM_BUFFER_BYTES,
    )?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    match histogram_algorithm {
        GpuHistogramAlgorithm::CpuChunked => {
            let key_slice = data_key.as_slice();
            let key_chunks = key_slice.input_chunks::<i32>(&partitioner)?;
            let offset_chunks = partition_offsets.chunks_mut();

            for (key_chunk, mut offset_chunk) in key_chunks.iter().zip(offset_chunks) {
                partitioner.cpu_prefix_sum(RadixPass::First, key_chunk, &mut offset_chunk)?;
            }
        }
        _ => {
            partitioner.prefix_sum(
                RadixPass::First,
                data_key.as_launchable_slice(),
                &mut partition_offsets,
                &stream,
            )?;
        }
    }

    partitioner.partition(
        RadixPass::First,
        data_key.as_launchable_slice(),
        data_pay.as_launchable_slice(),
        partition_offsets,
        &mut partitioned_relation,
        &stream,
    )?;

    stream.synchronize()?;

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

fn run_gpu_partitioning_and_copy_with_payload<KeyGenFn, PayGenFn, ValidatorFn>(
    tuples: usize,
    key_gen: Box<KeyGenFn>,
    pay_gen: Box<PayGenFn>,
    histogram_algorithm: GpuHistogramAlgorithm,
    partition_algorithm: GpuRadixPartitionAlgorithm,
    radix_bits: RadixBits,
    grid_size: GridSize,
    block_size: BlockSize,
    mut validator: Box<ValidatorFn>,
) -> Result<(), Box<dyn Error>>
where
    KeyGenFn: FnOnce(&mut [i32]) -> Result<(), Box<dyn Error>>,
    PayGenFn: FnOnce(&mut [i32]) -> Result<(), Box<dyn Error>>,
    ValidatorFn: FnMut(
        RadixPass,
        &RadixBits,
        &[i32],
        &[i32],
        &PartitionedRelation<Tuple<i32, i32>>,
        &[i32],
        &[i32],
    ) -> Result<(), Box<dyn Error>>,
{
    const DMEM_BUFFER_BYTES: usize = 8 * 1024;

    let _context = rustacuda::quick_init()?;

    let mut data_key = Allocator::alloc_deref_mem::<i32>(DerefMemType::CudaPinnedMem, tuples);
    let mut data_pay = Allocator::alloc_deref_mem::<i32>(DerefMemType::CudaPinnedMem, tuples);

    key_gen(data_key.as_mut_slice())?;
    pay_gen(data_pay.as_mut_slice())?;

    // Ensure that the allocated memory is zeroed
    let alloc_fn = Box::new(|len: usize| {
        let mut mem = Allocator::alloc_deref_mem(DerefMemType::CudaUniMem, len);
        mem.iter_mut().for_each(|x| *x = Default::default());
        mem.into()
    });

    let mut partition_offsets = PartitionOffsets::new(
        histogram_algorithm,
        grid_size.x,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    let mut partitioned_relation = PartitionedRelation::new(
        tuples,
        histogram_algorithm,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        grid_size.x,
        alloc_fn,
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    let mut partitioner = GpuRadixPartitioner::new(
        histogram_algorithm,
        partition_algorithm,
        radix_bits,
        &grid_size,
        &block_size,
        DMEM_BUFFER_BYTES,
    )?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let mut cached_key: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;
    let mut cached_pay: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;

    partitioner.prefix_sum_and_copy_with_payload(
        RadixPass::First,
        data_key.as_launchable_slice(),
        data_pay.as_launchable_slice(),
        cached_key.as_launchable_mut_slice(),
        cached_pay.as_launchable_mut_slice(),
        &mut partition_offsets,
        &stream,
    )?;

    partitioner.partition(
        RadixPass::First,
        cached_key.as_launchable_slice(),
        cached_pay.as_launchable_slice(),
        partition_offsets,
        &mut partitioned_relation,
        &stream,
    )?;

    stream.synchronize()?;

    validator(
        RadixPass::First,
        &radix_bits,
        data_key.as_slice(),
        data_pay.as_slice(),
        &partitioned_relation,
        cached_key.as_slice(),
        cached_pay.as_slice(),
    )?;

    Ok(())
}

fn run_gpu_two_pass_partitioning<KeyGenFn, PayGenFn, ValidatorFn>(
    tuples: usize,
    key_gen: Box<KeyGenFn>,
    pay_gen: Box<PayGenFn>,
    histogram_algorithm: GpuHistogramAlgorithm,
    partition_algorithm: GpuRadixPartitionAlgorithm,
    histogram_algorithm_2nd: GpuHistogramAlgorithm,
    partition_algorithm_2nd: GpuRadixPartitionAlgorithm,
    radix_bits: RadixBits,
    grid_size: GridSize,
    block_size: BlockSize,
    mut validator: Box<ValidatorFn>,
) -> Result<(), Box<dyn Error>>
where
    KeyGenFn: FnOnce(&mut [i32]) -> Result<(), Box<dyn Error>>,
    PayGenFn: FnOnce(&mut [i32]) -> Result<(), Box<dyn Error>>,
    ValidatorFn: FnMut(
        RadixPass,
        &RadixBits,
        &PartitionedRelation<Tuple<i32, i32>>,
        u32,
        &PartitionedRelation<Tuple<i32, i32>>,
        &[i32],
        &[i32],
    ) -> Result<(), Box<dyn Error>>,
{
    const DMEM_BUFFER_BYTES: usize = 8 * 1024;

    let _context = rustacuda::quick_init()?;

    let mut data_key: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;
    let mut data_pay: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;

    key_gen(data_key.as_mut_slice())?;
    pay_gen(data_pay.as_mut_slice())?;

    let mut partition_offsets = PartitionOffsets::new(
        histogram_algorithm,
        grid_size.x,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    let mut partitioned_relation = PartitionedRelation::new(
        tuples,
        histogram_algorithm,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        grid_size.x,
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    let mut partitioner = GpuRadixPartitioner::new(
        histogram_algorithm,
        partition_algorithm,
        radix_bits,
        &grid_size,
        &block_size,
        DMEM_BUFFER_BYTES,
    )?;

    let mut partitioner_2nd = GpuRadixPartitioner::new(
        histogram_algorithm_2nd,
        partition_algorithm_2nd,
        radix_bits,
        &grid_size,
        &block_size,
        DMEM_BUFFER_BYTES,
    )?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let data_key = Mem::CudaPinnedMem(data_key);
    let data_pay = Mem::CudaPinnedMem(data_pay);

    partitioner.prefix_sum(
        RadixPass::First,
        data_key.as_launchable_slice(),
        &mut partition_offsets,
        &stream,
    )?;

    partitioner.partition(
        RadixPass::First,
        data_key.as_launchable_slice(),
        data_pay.as_launchable_slice(),
        partition_offsets,
        &mut partitioned_relation,
        &stream,
    )?;
    stream.synchronize()?;

    // TODO: compute size of largest partition from partition_offsets
    let mut cached_key: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;
    let mut cached_pay: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;

    for partition_id in 0..radix_bits.pass_fanout(RadixPass::First).unwrap() {
        // Ensure that padded entries are zero for testing
        cached_key.iter_mut().for_each(|x| *x = 0);
        cached_pay.iter_mut().for_each(|x| *x = 0);

        let mut partition_offsets_2nd = PartitionOffsets::new(
            histogram_algorithm_2nd,
            grid_size.x,
            radix_bits.pass_radix_bits(RadixPass::Second).unwrap(),
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
        );

        let partition_len = partitioned_relation.partition_len(partition_id)?;

        let mut partitioned_relation_2nd = PartitionedRelation::new(
            partition_len,
            histogram_algorithm_2nd,
            radix_bits.pass_radix_bits(RadixPass::Second).unwrap(),
            grid_size.x,
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
            Allocator::mem_alloc_fn(MemType::CudaUniMem),
        );

        let cached_key_slice = &mut cached_key.as_mut_slice()[0..partition_len];
        let cached_pay_slice = &mut cached_pay.as_mut_slice()[0..partition_len];

        partitioner_2nd.prefix_sum_and_transform(
            RadixPass::Second,
            partition_id,
            &partitioned_relation,
            cached_key_slice.as_launchable_mut_slice(),
            cached_pay_slice.as_launchable_mut_slice(),
            &mut partition_offsets_2nd,
            &stream,
        )?;

        partitioner_2nd.partition(
            RadixPass::Second,
            cached_key_slice.as_launchable_slice(),
            cached_pay_slice.as_launchable_slice(),
            partition_offsets_2nd,
            &mut partitioned_relation_2nd,
            &stream,
        )?;

        stream.synchronize()?;

        validator(
            RadixPass::Second,
            &radix_bits,
            &partitioned_relation,
            partition_id,
            &partitioned_relation_2nd,
            cached_key_slice,
            cached_pay_slice,
        )?;
    }

    Ok(())
}

fn gpu_tuple_loss_or_duplicates(
    _radix_pass: RadixPass,
    _radix_bits: &RadixBits,
    data_key: &[i32],
    data_pay: &[i32],
    partitioned_relation: &PartitionedRelation<Tuple<i32, i32>>,
    partition_id: Option<u32>,
) -> Result<(), Box<dyn Error>> {
    let mut original_tuples: HashMap<_, _> = data_key
        .iter()
        .cloned()
        .zip(data_pay.iter().cloned().zip(std::iter::repeat(0)))
        .collect();

    let relation = unsafe { partitioned_relation.as_raw_relation_slice()? };

    relation.iter().cloned().for_each(|Tuple { key, value }| {
        let entry = original_tuples.entry(key);
        match entry {
            entry @ Entry::Occupied(_) => {
                let key = *entry.key();
                entry.and_modify(|(original_value, counter)| {
                    let id_str = partition_id
                        .map_or_else(|| "".to_string(), |id| format!(" in partition {}", id));
                    assert_eq!(
                        value, *original_value,
                        "Invalid payload{}: {}; expected: {}",
                        id_str, value, *original_value
                    );
                    assert_eq!(*counter, 0, "Duplicate key: {}", key);
                    *counter = *counter + 1;
                });
            }
            entry @ Entry::Vacant(_) => {
                // skip padding entries
                if *entry.key() != 0 {
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

fn gpu_verify_partitions(
    radix_pass: RadixPass,
    radix_bits: &RadixBits,
    _data_key: &[i32],
    _data_pay: &[i32],
    partitioned_relation: &PartitionedRelation<Tuple<i32, i32>>,
    partition_id: Option<u32>,
) -> Result<(), Box<dyn Error>> {
    (0..partitioned_relation.num_chunks())
        .flat_map(|c| iter::repeat(c).zip(0..partitioned_relation.fanout()))
        .flat_map(|(c, p)| iter::repeat((c, p)).zip(partitioned_relation[(c, p)].iter()))
        .enumerate()
        .for_each(|(i, ((c, p), &tuple))| {
            let dst_partition = key_to_partition(tuple.key, radix_bits, radix_pass);
            let id_str = partition_id.map_or_else(|| "".to_string(), |id| format!("{}:", id));
            assert_eq!(
                dst_partition, p,
                "Wrong partitioning detected in chunk {} at position {}: \
                key {} in partition {}{}; expected partition {}{}",
                c, i, tuple.key, id_str, p, id_str, dst_partition
            );
        });

    Ok(())
}

fn gpu_check_copy_with_payload(
    _radix_pass: RadixPass,
    _radix_bits: &RadixBits,
    data_key: &[i32],
    data_pay: &[i32],
    _partitioned_relation: &PartitionedRelation<Tuple<i32, i32>>,
    cached_key: &[i32],
    cached_pay: &[i32],
) -> Result<(), Box<dyn Error>> {
    data_key
        .iter()
        .zip(cached_key.iter())
        .enumerate()
        .for_each(|(i, (&original, &cached))| {
            assert_eq!(
                original, cached,
                "Wrong key detected at position {}: {}",
                i, cached
            );
        });

    data_pay
        .iter()
        .zip(cached_pay.iter())
        .enumerate()
        .for_each(|(i, (&original, &cached))| {
            assert_eq!(
                original, cached,
                "Wrong payload detected at position {}: {}",
                i, cached
            );
        });

    Ok(())
}

fn gpu_two_pass_tuple_loss_or_duplicates(
    radix_pass: RadixPass,
    radix_bits: &RadixBits,
    _partitioned_relation_1st: &PartitionedRelation<Tuple<i32, i32>>,
    partition_id: u32,
    partitioned_relation_2nd: &PartitionedRelation<Tuple<i32, i32>>,
    cached_key_slice: &[i32],
    cached_pay_slice: &[i32],
) -> Result<(), Box<dyn Error>> {
    gpu_tuple_loss_or_duplicates(
        radix_pass,
        radix_bits,
        cached_key_slice,
        cached_pay_slice,
        partitioned_relation_2nd,
        Some(partition_id),
    )
}

fn gpu_two_pass_verify_partitions(
    radix_pass: RadixPass,
    radix_bits: &RadixBits,
    _partitioned_relation_1st: &PartitionedRelation<Tuple<i32, i32>>,
    partition_id: u32,
    partitioned_relation_2nd: &PartitionedRelation<Tuple<i32, i32>>,
    cached_key_slice: &[i32],
    cached_pay_slice: &[i32],
) -> Result<(), Box<dyn Error>> {
    gpu_verify_partitions(
        radix_pass,
        radix_bits,
        cached_key_slice,
        cached_pay_slice,
        partitioned_relation_2nd,
        Some(partition_id),
    )
}

fn gpu_verify_transformed_input(
    _radix_pass: RadixPass,
    _radix_bits: &RadixBits,
    partitioned_relation_1st: &PartitionedRelation<Tuple<i32, i32>>,
    partition_id: u32,
    _partitioned_relation_2nd: &PartitionedRelation<Tuple<i32, i32>>,
    cached_key_slice: &[i32],
    cached_pay_slice: &[i32],
) -> Result<(), Box<dyn Error>> {
    (0..partitioned_relation_1st.num_chunks())
        .flat_map(|c| partitioned_relation_1st[(c, partition_id)].iter())
        .zip(cached_key_slice.iter())
        .zip(cached_pay_slice.iter())
        .enumerate()
        .for_each(|(i, ((&tuple, &key), &pay))| {
            assert_eq!(
                tuple.key, key,
                "Wrong key detected in partition {} at position {}: {}",
                partition_id, i, key
            );
            assert_eq!(
                tuple.value, pay,
                "Wrong payload detected in partition {} at position {}: {}",
                partition_id, i, pay
            );
        });

    Ok(())
}

#[test]
fn gpu_tuple_loss_or_duplicates_cpu_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::CpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_cpu_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::CpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_small_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        100,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(4),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_small_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        100,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(4),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_chunked_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(12),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(12),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_contiguous_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_contiguous_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_contiguous_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_contiguous_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_contiguous_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(12),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_contiguous_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(12),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_contiguous_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_contiguous_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_laswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_laswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_laswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_chunked_laswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_laswwc_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_laswwc_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_sswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::SSWWC,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_sswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::SSWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_sswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::SSWWC,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_chunked_sswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::SSWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_sswwc_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::SSWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_sswwc_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::SSWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_sswwc_non_temporal_i32_10_bits(
) -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::SSWWCNT,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_sswwc_v2_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_chunked_sswwc_v2_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_sswwc_v2_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWC,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_hsswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWC,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_chunked_hsswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_hsswwc_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v2_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv2,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v2_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv2,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_hsswwc_v2_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv2,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_chunked_hsswwc_v2_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv2,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v2_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv2,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_hsswwc_v2_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv2,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v3_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv3,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v3_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv3,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_hsswwc_v3_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv3,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_chunked_hsswwc_v3_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv3,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v3_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv3,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_hsswwc_v3_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv3,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v4_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv4,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v4_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv4,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_hsswwc_v4_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv4,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_chunked_hsswwc_v4_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv4,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v4_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv4,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_hsswwc_v4_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::HSSWWCv4,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&gpu_verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_copy_with_payload_contiguous_i32_2_bits(
) -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning_and_copy_with_payload(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(|pass, rb: &_, dk: &_, dp: &_, pr: &_, _: &_, _: &_| {
            gpu_tuple_loss_or_duplicates(pass, rb, dk, dp, pr, None)
        }),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_copy_with_payload_contiguous_i32_10_bits(
) -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning_and_copy_with_payload(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(|pass, rb: &_, dk: &_, dp: &_, pr: &_, _: &_, _: &_| {
            gpu_tuple_loss_or_duplicates(pass, rb, dk, dp, pr, None)
        }),
    )
}

#[test]
fn gpu_verify_partitions_copy_with_payload_contiguous_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning_and_copy_with_payload(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(|pass, rb: &_, dk: &_, dp: &_, pr: &_, _: &_, _: &_| {
            gpu_verify_partitions(pass, rb, dk, dp, pr, None)
        }),
    )
}

#[test]
fn gpu_verify_partitions_copy_with_payload_contiguous_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning_and_copy_with_payload(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(|pass, rb: &_, dk: &_, dp: &_, pr: &_, _: &_, _: &_| {
            gpu_verify_partitions(pass, rb, dk, dp, pr, None)
        }),
    )
}

#[test]
fn gpu_check_copy_with_payload_contiguous_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning_and_copy_with_payload(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_check_copy_with_payload),
    )
}

#[test]
fn gpu_check_copy_with_payload_contiguous_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning_and_copy_with_payload(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&gpu_check_copy_with_payload),
    )
}

#[test]
fn gpu_loss_or_duplicates_two_pass_chunked_contiguous_i32_0_0_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(0), Some(0), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_two_pass_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_loss_or_duplicates_two_pass_chunked_contiguous_i32_2_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(2), Some(2), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_two_pass_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_loss_or_duplicates_two_pass_chunked_contiguous_i32_10_10_bits() -> Result<(), Box<dyn Error>>
{
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(10), Some(10), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_two_pass_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_loss_or_duplicates_two_pass_contiguous_contiguous_i32_2_2_bits() -> Result<(), Box<dyn Error>>
{
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(2), Some(2), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_two_pass_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_two_pass_chunked_contiguous_i32_0_0_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(0), Some(0), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_two_pass_verify_partitions),
    )
}

#[test]
fn gpu_verify_two_pass_chunked_contiguous_i32_2_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(2), Some(2), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_two_pass_verify_partitions),
    )
}

#[test]
fn gpu_verify_two_pass_chunked_contiguous_i32_10_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(10), Some(10), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_two_pass_verify_partitions),
    )
}

#[test]
fn gpu_verify_two_pass_contiguous_contiguous_i32_2_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(2), Some(2), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_two_pass_verify_partitions),
    )
}

#[test]
fn gpu_transform_two_pass_chunked_contiguous_i32_0_0_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(0), Some(0), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_verify_transformed_input),
    )
}

#[test]
fn gpu_transform_two_pass_chunked_contiguous_i32_2_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(2), Some(2), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_verify_transformed_input),
    )
}

#[test]
fn gpu_transform_two_pass_chunked_contiguous_i32_10_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(10), Some(10), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_verify_transformed_input),
    )
}

#[test]
fn gpu_transform_two_pass_contiguous_contiguous_i32_2_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(2), Some(2), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_verify_transformed_input),
    )
}

#[test]
fn gpu_loss_or_duplicates_two_pass_laswwc_i32_6_6_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::new(Some(6), Some(6), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_two_pass_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_two_pass_laswwc_i32_6_6_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::new(Some(6), Some(6), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_two_pass_verify_partitions),
    )
}

#[test]
fn gpu_loss_or_duplicates_two_pass_sswwc_i32_6_6_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::SSWWC,
        RadixBits::new(Some(6), Some(6), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_two_pass_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_two_pass_sswwc_i32_6_6_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::SSWWC,
        RadixBits::new(Some(6), Some(6), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_two_pass_verify_partitions),
    )
}

#[test]
fn gpu_loss_or_duplicates_two_pass_sswwc_v2_i32_6_6_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::new(Some(6), Some(6), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_two_pass_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_two_pass_sswwc_v2_i32_6_6_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 1..=(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 1..=10000)?)),
        GpuHistogramAlgorithm::GpuChunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::GpuContiguous,
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::new(Some(6), Some(6), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&gpu_two_pass_verify_partitions),
    )
}
