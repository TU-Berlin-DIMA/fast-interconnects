/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2021, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use data_store::join_data::{JoinData, JoinDataBuilder};
use datagen::relation::UniformRelation;
use num_rational::Ratio;
use numa_gpu::runtime::allocator::{DerefMemType, MemType};
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::hw_info::NvidiaDriverInfo;
use numa_gpu::runtime::numa::NodeRatio;
use once_cell::sync::Lazy;
use radix_join::error::Result as RJResult;
use radix_join::execution_methods::gpu_radix_join::gpu_radix_join;
use radix_join::measurement::harness::RadixJoinPoint;
use rustacuda::context::{Context, CurrentContext, UnownedContext};
use rustacuda::device::Device;
use rustacuda::function::{BlockSize, GridSize};
use sql_ops::join::HashingScheme;
use sql_ops::partition::gpu_radix_partition::{GpuHistogramAlgorithm, GpuRadixPartitionAlgorithm};
use sql_ops::partition::RadixBits;
use std::error::Error;
use std::mem;
use std::result::Result;

#[cfg(target_arch = "powerpc64")]
use radix_join::execution_methods::gpu_triton_join::gpu_triton_join;

static mut CUDA_CONTEXT_OWNER: Option<Context> = None;
static CUDA_CONTEXT: Lazy<UnownedContext> = Lazy::new(|| {
    let context = rustacuda::quick_init().expect("Failed to initialize CUDA context");
    let unowned = context.get_unowned();

    unsafe {
        CUDA_CONTEXT_OWNER = Some(context);
    }

    unowned
});

fn run_gpu_radix_join_validate_sum<JoinFn, PartitionsFn>(
    join_fn: JoinFn,
    partitions_fn: PartitionsFn,
    inner_relation_len: usize,
    outer_relation_len: usize,
    radix_bits: RadixBits,
    grid_size: GridSize,
    block_size: BlockSize,
    threads: usize,
    hashing_scheme: HashingScheme,
) -> Result<(), Box<dyn Error>>
where
    JoinFn: FnOnce(
        &mut JoinData<i32>,
        HashingScheme,
        [GpuHistogramAlgorithm; 2],
        [GpuRadixPartitionAlgorithm; 2],
        &RadixBits,
        usize,
        usize,
        CpuAffinity,
        MemType,
        MemType,
        (&GridSize, &BlockSize),
        (&GridSize, &BlockSize),
    ) -> RJResult<(i64, RadixJoinPoint)>,
    PartitionsFn: FnOnce(&Device) -> Result<MemType, Box<dyn Error>>,
{
    const DMEM_BUFFER_BYTES: usize = 8 * 1024;

    let prefix_sum_algorithms = [
        GpuHistogramAlgorithm::GpuChunked,
        GpuHistogramAlgorithm::GpuContiguous,
    ];
    let partition_algorithms = [
        GpuRadixPartitionAlgorithm::SSWWCv2,
        GpuRadixPartitionAlgorithm::SSWWCv2,
    ];

    CurrentContext::set_current(&*CUDA_CONTEXT)?;

    let data_gen_fn = Box::new(
        |pk_rel_key: &mut [_], pk_rel_pay: &mut [_], fk_rel_key: &mut [_], fk_rel_pay: &mut [_]| {
            UniformRelation::gen_primary_key(pk_rel_key, None)?;
            UniformRelation::gen_foreign_key_from_primary_key(fk_rel_key, pk_rel_key);

            pk_rel_pay
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = (i + 1) as i32);
            fk_rel_pay
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = (i + 1) as i32);

            Ok(())
        },
    );

    let mut data_builder = JoinDataBuilder::default();
    data_builder
        .inner_mem_type(DerefMemType::CudaPinnedMem)
        .outer_mem_type(DerefMemType::CudaPinnedMem)
        .inner_len(inner_relation_len)
        .outer_len(outer_relation_len);
    let (mut join_data, _, _) = data_builder.build_with_data_gen(data_gen_fn)?;

    let partitions_mem_type = partitions_fn(&CurrentContext::get_device()?)?;

    let (result_sum, _) = join_fn(
        &mut join_data,
        hashing_scheme,
        prefix_sum_algorithms,
        partition_algorithms,
        &radix_bits,
        DMEM_BUFFER_BYTES,
        threads,
        CpuAffinity::default(),
        partitions_mem_type,
        MemType::CudaDevMem,
        (&grid_size, &block_size),
        (&grid_size, &block_size),
    )?;

    assert_eq!(
        (outer_relation_len as i64 * (outer_relation_len as i64 + 1)) / 2,
        result_sum
    );

    Ok(())
}

fn partitions_type_normal(_: &Device) -> Result<MemType, Box<dyn Error>> {
    Ok(MemType::CudaPinnedMem)
}

#[allow(dead_code)]
fn partitions_type_cached(device: &Device) -> Result<MemType, Box<dyn Error>> {
    let cache_node = device.numa_node()?;
    let overflow_node = device.numa_memory_affinity()?;

    let mem_type = MemType::DistributedNumaMem(Box::new([
        NodeRatio {
            node: cache_node,
            ratio: Ratio::from_integer(0),
        },
        NodeRatio {
            node: overflow_node,
            ratio: Ratio::from_integer(0),
        },
    ]));

    Ok(mem_type)
}

#[test]
fn test_gpu_radix_partition_validate_sum_perfect_small_i32() -> Result<(), Box<dyn Error>> {
    run_gpu_radix_join_validate_sum(
        &gpu_radix_join::<i32>,
        &partitions_type_normal,
        100_000,
        100_000,
        RadixBits::new(Some(3), Some(3), None),
        GridSize::from(8),
        BlockSize::from(128),
        1,
        HashingScheme::Perfect,
    )
}

#[test]
fn test_gpu_radix_partition_validate_sum_bucketchaining_small_i32() -> Result<(), Box<dyn Error>> {
    run_gpu_radix_join_validate_sum(
        &gpu_radix_join::<i32>,
        &partitions_type_normal,
        100_000,
        100_000,
        RadixBits::new(Some(3), Some(3), None),
        GridSize::from(8),
        BlockSize::from(128),
        1,
        HashingScheme::BucketChaining,
    )
}

#[test]
fn test_gpu_radix_partition_validate_sum_bucketchaining_large_i32() -> Result<(), Box<dyn Error>> {
    run_gpu_radix_join_validate_sum(
        &gpu_radix_join::<i32>,
        &partitions_type_normal,
        (1 << 31) / mem::size_of::<i32>(),
        (1 << 31) / mem::size_of::<i32>(),
        RadixBits::new(Some(9), Some(9), None),
        GridSize::from(8),
        BlockSize::from(128),
        1,
        HashingScheme::BucketChaining,
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn test_gpu_triton_partition_validate_sum_perfect_small_i32() -> Result<(), Box<dyn Error>> {
    run_gpu_radix_join_validate_sum(
        &gpu_triton_join::<i32>,
        &partitions_type_cached,
        100_000,
        100_000,
        RadixBits::new(Some(3), Some(3), None),
        GridSize::from(8),
        BlockSize::from(128),
        1,
        HashingScheme::Perfect,
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn test_gpu_triton_partition_validate_sum_bucketchaining_small_i32() -> Result<(), Box<dyn Error>> {
    run_gpu_radix_join_validate_sum(
        &gpu_triton_join::<i32>,
        &partitions_type_cached,
        100_000,
        100_000,
        RadixBits::new(Some(3), Some(3), None),
        GridSize::from(8),
        BlockSize::from(128),
        1,
        HashingScheme::BucketChaining,
    )
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn test_gpu_triton_partition_validate_sum_bucketchaining_large_i32() -> Result<(), Box<dyn Error>> {
    run_gpu_radix_join_validate_sum(
        &gpu_triton_join::<i32>,
        &partitions_type_cached,
        (1 << 31) / mem::size_of::<i32>(),
        (1 << 31) / mem::size_of::<i32>(),
        RadixBits::new(Some(9), Some(9), None),
        GridSize::from(8),
        BlockSize::from(128),
        1,
        HashingScheme::BucketChaining,
    )
}
