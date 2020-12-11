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
use numa_gpu::runtime::memory::Mem;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::stream::{Stream, StreamFlags};
use sql_ops::join::cuda_radix_join::CudaRadixJoin;
use sql_ops::join::HashingScheme;
use sql_ops::partition::gpu_radix_partition::{
    GpuHistogramAlgorithm, GpuRadixPartitionAlgorithm, GpuRadixPartitioner,
};
use sql_ops::partition::RadixPass;
use sql_ops::partition::{PartitionOffsets, PartitionedRelation};
use std::error::Error;
use std::result::Result;

fn gpu_verify_join_aggregate(
    build_tuples: usize,
    probe_tuples: usize,
    hashing_scheme: HashingScheme,
    radix_bits: u32,
    grid_size: GridSize,
    block_size: BlockSize,
) -> Result<(), Box<dyn Error>> {
    let histogram_algorithm = GpuHistogramAlgorithm::GpuChunked;
    let partition_algorithm = GpuRadixPartitionAlgorithm::NC;
    let num_chunks = GridSize::from(1); // one contiguous chunk

    let _ctx = rustacuda::quick_init()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let alloc_fn = Allocator::deref_mem_alloc_fn::<i32>(DerefMemType::CudaUniMem);

    let mut inner_rel_key = alloc_fn(build_tuples);
    let mut inner_rel_pay = alloc_fn(build_tuples);
    let mut outer_rel_key = alloc_fn(probe_tuples);
    let mut outer_rel_pay = alloc_fn(probe_tuples);

    UniformRelation::gen_primary_key(&mut inner_rel_key, None)?;
    UniformRelation::gen_foreign_key_from_primary_key(&mut outer_rel_key, &inner_rel_key);

    inner_rel_pay
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = (i + 1) as i32);
    outer_rel_pay
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = (i + 1) as i32);

    let mut inner_rel_partition_offsets = PartitionOffsets::new(
        histogram_algorithm,
        num_chunks.x,
        radix_bits.into(),
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    let mut outer_rel_partition_offsets = PartitionOffsets::new(
        histogram_algorithm,
        num_chunks.x,
        radix_bits.into(),
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    let mut inner_rel_partitions = PartitionedRelation::new(
        inner_rel_key.len(),
        histogram_algorithm,
        radix_bits.into(),
        num_chunks.x,
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    let mut outer_rel_partitions = PartitionedRelation::new(
        outer_rel_key.len(),
        histogram_algorithm,
        radix_bits.into(),
        num_chunks.x,
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    let mut result_sums =
        Allocator::alloc_mem(MemType::CudaUniMem, (grid_size.x * block_size.x) as usize);
    let mut task_assignments =
        Allocator::alloc_mem(MemType::CudaDevMem, (grid_size.x + 1) as usize);

    // Initialize result
    if let Mem::CudaUniMem(ref mut c) = result_sums {
        c.iter_mut().map(|sum| *sum = 0).for_each(drop);
    }

    let mut radix_partitioner = GpuRadixPartitioner::new(
        histogram_algorithm,
        partition_algorithm,
        radix_bits.into(),
        &num_chunks,
        &block_size,
        0,
    )?;

    let radix_join = CudaRadixJoin::new(
        RadixPass::First,
        radix_bits.into(),
        hashing_scheme,
        &grid_size,
        &block_size,
    )?;

    radix_partitioner.prefix_sum(
        RadixPass::First,
        inner_rel_key.as_launchable_slice(),
        &mut inner_rel_partition_offsets,
        &stream,
    )?;
    radix_partitioner.prefix_sum(
        RadixPass::First,
        outer_rel_key.as_launchable_slice(),
        &mut outer_rel_partition_offsets,
        &stream,
    )?;
    radix_partitioner.partition(
        RadixPass::First,
        inner_rel_key.as_launchable_slice(),
        inner_rel_pay.as_launchable_slice(),
        inner_rel_partition_offsets,
        &mut inner_rel_partitions,
        &stream,
    )?;
    radix_partitioner.partition(
        RadixPass::First,
        outer_rel_key.as_launchable_slice(),
        outer_rel_pay.as_launchable_slice(),
        outer_rel_partition_offsets,
        &mut outer_rel_partitions,
        &stream,
    )?;
    radix_join.join(
        &inner_rel_partitions,
        &outer_rel_partitions,
        &mut result_sums.as_launchable_mut_slice(),
        &mut task_assignments.as_launchable_mut_slice(),
        &stream,
    )?;

    stream.synchronize()?;
    let result_sum = if let Mem::CudaUniMem(ref r) = result_sums {
        r.iter().sum()
    } else {
        0
    };

    assert_eq!(
        (probe_tuples as i64 * (probe_tuples as i64 + 1)) / 2,
        result_sum
    );

    Ok(())
}

#[test]
fn gpu_verify_join_aggregate_smem_perfect_i32_0_bits() -> Result<(), Box<dyn Error>> {
    gpu_verify_join_aggregate(
        6100,
        6100,
        HashingScheme::Perfect,
        0,
        GridSize::from(1),
        BlockSize::from(128),
    )
}

#[test]
fn gpu_verify_join_aggregate_smem_perfect_i32_1_bits() -> Result<(), Box<dyn Error>> {
    gpu_verify_join_aggregate(
        6100 * 2_usize.pow(1),
        6100 * 2_usize.pow(1),
        HashingScheme::Perfect,
        1,
        GridSize::from(1),
        BlockSize::from(128),
    )
}

#[test]
fn gpu_verify_join_aggregate_smem_perfect_i32_2_bits() -> Result<(), Box<dyn Error>> {
    gpu_verify_join_aggregate(
        6100 * 2_usize.pow(2),
        6100 * 2_usize.pow(2),
        HashingScheme::Perfect,
        2,
        GridSize::from(1),
        BlockSize::from(128),
    )
}

#[test]
fn gpu_verify_join_aggregate_smem_perfect_i32_3_bits() -> Result<(), Box<dyn Error>> {
    gpu_verify_join_aggregate(
        6100 * 2_usize.pow(3),
        6100 * 2_usize.pow(3),
        HashingScheme::Perfect,
        3,
        GridSize::from(1),
        BlockSize::from(128),
    )
}

#[test]
fn gpu_verify_join_aggregate_smem_perfect_i32_4_bits() -> Result<(), Box<dyn Error>> {
    gpu_verify_join_aggregate(
        6100 * 2_usize.pow(4),
        6100 * 2_usize.pow(4),
        HashingScheme::Perfect,
        4,
        GridSize::from(1),
        BlockSize::from(128),
    )
}

#[test]
fn gpu_verify_join_aggregate_smem_perfect_i32_5_bits() -> Result<(), Box<dyn Error>> {
    gpu_verify_join_aggregate(
        6100 * 2_usize.pow(5),
        6100 * 2_usize.pow(5),
        HashingScheme::Perfect,
        5,
        GridSize::from(1),
        BlockSize::from(128),
    )
}

#[test]
fn gpu_verify_join_aggregate_smem_perfect_i32_8_bits() -> Result<(), Box<dyn Error>> {
    gpu_verify_join_aggregate(
        6100 * 2_usize.pow(8),
        6100 * 2_usize.pow(8),
        HashingScheme::Perfect,
        8,
        GridSize::from(1),
        BlockSize::from(128),
    )
}

#[test]
fn gpu_verify_join_aggregate_task_assignment_i32_2_bits_3_blocks() -> Result<(), Box<dyn Error>> {
    gpu_verify_join_aggregate(
        6100 * 2_usize.pow(2),
        6100 * 2_usize.pow(2),
        HashingScheme::Perfect,
        2,
        GridSize::from(3),
        BlockSize::from(128),
    )
}

#[test]
fn gpu_verify_join_aggregate_task_assignment_i32_8_bits_3_blocks() -> Result<(), Box<dyn Error>> {
    gpu_verify_join_aggregate(
        6100 * 2_usize.pow(8),
        6100 * 2_usize.pow(8),
        HashingScheme::Perfect,
        8,
        GridSize::from(3),
        BlockSize::from(128),
    )
}

#[test]
fn gpu_verify_join_aggregate_task_assignment_1_bits_80_blocks() -> Result<(), Box<dyn Error>> {
    gpu_verify_join_aggregate(
        6100 * 2_usize.pow(1),
        6100 * 2_usize.pow(1),
        HashingScheme::Perfect,
        1,
        GridSize::from(80),
        BlockSize::from(128),
    )
}

#[test]
fn gpu_verify_join_aggregate_task_assignment_8_bits_80_blocks() -> Result<(), Box<dyn Error>> {
    gpu_verify_join_aggregate(
        6100 * 2_usize.pow(8),
        6100 * 2_usize.pow(8),
        HashingScheme::Perfect,
        8,
        GridSize::from(80),
        BlockSize::from(128),
    )
}

#[test]
fn gpu_verify_join_aggregate_smem_bucketchaining_i32_0_bits() -> Result<(), Box<dyn Error>> {
    gpu_verify_join_aggregate(
        4096,
        4096,
        HashingScheme::BucketChaining,
        0,
        GridSize::from(1),
        BlockSize::from(128),
    )
}

#[test]
fn gpu_verify_join_aggregate_smem_bucketchaining_i32_2_bits() -> Result<(), Box<dyn Error>> {
    gpu_verify_join_aggregate(
        4096 * 2_usize.pow(1),
        4096 * 2_usize.pow(1),
        HashingScheme::BucketChaining,
        1,
        GridSize::from(1),
        BlockSize::from(128),
    )
}

#[test]
fn gpu_verify_join_aggregate_smem_bucketchaining_i32_8_bits() -> Result<(), Box<dyn Error>> {
    gpu_verify_join_aggregate(
        4096 * 2_usize.pow(8),
        4096 * 2_usize.pow(8),
        HashingScheme::BucketChaining,
        8,
        GridSize::from(1),
        BlockSize::from(128),
    )
}

#[test]
fn gpu_verify_join_aggregate_task_assignment_smem_bucketchaining() -> Result<(), Box<dyn Error>> {
    gpu_verify_join_aggregate(
        4096 * 2_usize.pow(8),
        4096 * 2_usize.pow(8),
        HashingScheme::BucketChaining,
        8,
        GridSize::from(80),
        BlockSize::from(128),
    )
}
