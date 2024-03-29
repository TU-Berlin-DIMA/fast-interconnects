// Copyright 2020-2022 Clemens Lutz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

mod radix_partition;

use datagen::relation::{KeyAttribute, UniformRelation};
use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemType};
use numa_gpu::runtime::memory::{LaunchableMem, Mem};
use numa_gpu::utils::DeviceType;
use once_cell::sync::Lazy;
use radix_partition::*;
use rustacuda::context::{Context, CurrentContext, UnownedContext};
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::LockedBuffer;
use rustacuda::stream::{Stream, StreamFlags};
use sql_ops::partition::cpu_radix_partition::{
    CpuHistogramAlgorithm, CpuRadixPartitionAlgorithm, CpuRadixPartitioner,
};
use sql_ops::partition::gpu_radix_partition::{
    GpuHistogramAlgorithm, GpuRadixPartitionAlgorithm, GpuRadixPartitioner,
};
use sql_ops::partition::{
    PartitionOffsets, PartitionedRelation, RadixBits, RadixPartitionInputChunkable, RadixPass,
    Tuple,
};
use std::cmp;
use std::error::Error;
use std::mem;
use std::result::Result;

static mut CUDA_CONTEXT_OWNER: Option<Context> = None;
static CUDA_CONTEXT: Lazy<UnownedContext> = Lazy::new(|| {
    let context = rustacuda::quick_init().expect("Failed to initialize CUDA context");
    let unowned = context.get_unowned();

    unsafe {
        CUDA_CONTEXT_OWNER = Some(context);
    }

    unowned
});

fn run_gpu_partitioning<KeyGenFn, PayGenFn, ValidatorFn>(
    tuples: usize,
    key_gen: Box<KeyGenFn>,
    pay_gen: Box<PayGenFn>,
    histogram_algorithm: DeviceType<CpuHistogramAlgorithm, GpuHistogramAlgorithm>,
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

    CurrentContext::set_current(&*CUDA_CONTEXT)?;

    let mut data_key = Allocator::alloc_deref_mem::<i32>(DerefMemType::CudaPinnedMem, tuples);
    let mut data_pay = Allocator::alloc_deref_mem::<i32>(DerefMemType::CudaPinnedMem, tuples);

    key_gen(data_key.as_mut_slice())?;
    pay_gen(data_pay.as_mut_slice())?;

    // Ensure that the allocated memory is zeroed
    let alloc_fn = Box::new(|len: usize| {
        let mut mem = Allocator::alloc_deref_mem(DerefMemType::CudaUniMem, len);
        mem.iter_mut().for_each(|x| {
            *x = Tuple {
                key: i32::null_key(),
                value: i32::default(),
            }
        });
        mem.into()
    });

    let mut partition_offsets = PartitionOffsets::new(
        histogram_algorithm.either(|cpu| cpu.into(), |gpu| gpu.into()),
        grid_size.x,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    let mut partitioned_relation = PartitionedRelation::new(
        tuples,
        histogram_algorithm.either(|cpu| cpu.into(), |gpu| gpu.into()),
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        grid_size.x,
        alloc_fn,
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    let mut partitioner = GpuRadixPartitioner::new(
        histogram_algorithm.either(|cpu| cpu.into(), |gpu| gpu.into()),
        partition_algorithm,
        radix_bits.into(),
        &grid_size,
        &block_size,
        DMEM_BUFFER_BYTES,
    )?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    match histogram_algorithm {
        DeviceType::Cpu(histogram_algorithm) => {
            let key_slice = data_key.as_slice();
            let key_chunks = key_slice.input_chunks::<i32>(grid_size.x)?;
            let offset_chunks = partition_offsets.chunks_mut();

            let mut radix_prnr = CpuRadixPartitioner::new(
                histogram_algorithm,
                CpuRadixPartitionAlgorithm::NC,
                radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
                DerefMemType::SysMem,
            );

            for (key_chunk, offset_chunk) in key_chunks.into_iter().zip(offset_chunks) {
                radix_prnr.prefix_sum(key_chunk, offset_chunk)?;
            }
        }
        DeviceType::Gpu(_) => {
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
        &mut partition_offsets,
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

    CurrentContext::set_current(&*CUDA_CONTEXT)?;

    let mut data_key = Allocator::alloc_deref_mem::<i32>(DerefMemType::CudaPinnedMem, tuples);
    let mut data_pay = Allocator::alloc_deref_mem::<i32>(DerefMemType::CudaPinnedMem, tuples);

    key_gen(data_key.as_mut_slice())?;
    pay_gen(data_pay.as_mut_slice())?;

    // Ensure that the allocated memory is zeroed
    let alloc_fn = Box::new(|len: usize| {
        let mut mem = Allocator::alloc_deref_mem(DerefMemType::CudaUniMem, len);
        mem.iter_mut().for_each(|x| {
            *x = Tuple {
                key: i32::null_key(),
                value: i32::default(),
            }
        });
        mem.into()
    });

    let mut partition_offsets = PartitionOffsets::new(
        histogram_algorithm.into(),
        grid_size.x,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    let mut partitioned_relation = PartitionedRelation::new(
        tuples,
        histogram_algorithm.into(),
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
        &mut partition_offsets,
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

    CurrentContext::set_current(&*CUDA_CONTEXT)?;

    let mut data_key: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;
    let mut data_pay: LockedBuffer<i32> = LockedBuffer::new(&0, tuples)?;

    key_gen(data_key.as_mut_slice())?;
    pay_gen(data_pay.as_mut_slice())?;

    let mut partition_offsets = PartitionOffsets::new(
        histogram_algorithm.into(),
        grid_size.x,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    let mut partitioned_relation = PartitionedRelation::new(
        tuples,
        histogram_algorithm.into(),
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

    // Wait on offsets for computing max_partition_len
    stream.synchronize()?;

    let max_partition_len = (0..partition_offsets.fanout()).try_fold(0, |max, partition_id| {
        partition_offsets
            .partition_len(partition_id)
            .map(|len| cmp::max(max, len))
    })?;

    let mut cached_key: LockedBuffer<i32> = LockedBuffer::new(&0, max_partition_len)?;
    let mut cached_pay: LockedBuffer<i32> = LockedBuffer::new(&0, max_partition_len)?;

    let mut partition_offsets_2nd = PartitionOffsets::new(
        histogram_algorithm_2nd.into(),
        grid_size.x,
        radix_bits.pass_radix_bits(RadixPass::Second).unwrap(),
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    let mut partitioned_relation_2nd = PartitionedRelation::new(
        max_partition_len,
        histogram_algorithm_2nd.into(),
        radix_bits.pass_radix_bits(RadixPass::Second).unwrap(),
        grid_size.x,
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
        Allocator::mem_alloc_fn(MemType::CudaUniMem),
    );

    partitioner.partition(
        RadixPass::First,
        data_key.as_launchable_slice(),
        data_pay.as_launchable_slice(),
        &mut partition_offsets,
        &mut partitioned_relation,
        &stream,
    )?;

    for partition_id in 0..radix_bits.pass_fanout(RadixPass::First).unwrap() {
        let partition_len = partitioned_relation.partition_len(partition_id)?;
        partitioned_relation_2nd.resize(partition_len)?;

        let cached_key_slice = &mut cached_key.as_mut_slice()[0..partition_len];
        let cached_pay_slice = &mut cached_pay.as_mut_slice()[0..partition_len];

        // Ensure that padded entries are zero for testing
        unsafe {
            partitioned_relation_2nd
                .as_raw_relation_mut_slice()?
                .iter_mut()
                .for_each(|x| {
                    *x = Tuple {
                        key: i32::null_key(),
                        value: i32::default(),
                    }
                });
        }

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
            &mut partition_offsets_2nd,
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

#[test]
fn gpu_tuple_loss_or_duplicates_cpu_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Cpu(CpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_cpu_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Cpu(CpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_small_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        100,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(4),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_small_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        100,
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(4),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_chunked_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(12),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(12),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_contiguous_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Contiguous),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_contiguous_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Contiguous),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_contiguous_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Contiguous),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_contiguous_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Contiguous),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_contiguous_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Contiguous),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(12),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_contiguous_i32_12_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Contiguous),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(12),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_contiguous_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Contiguous),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_contiguous_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Contiguous),
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_laswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_laswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_laswwc_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_chunked_laswwc_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_laswwc_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_laswwc_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_contiguous_sswwc_v2_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Contiguous),
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_sswwc_v2_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_chunked_sswwc_v2_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_contiguous_sswwc_v2_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Contiguous),
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_sswwc_v2_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2g_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::SSWWCv2G,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2g_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::SSWWCv2G,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_sswwc_v2g_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::SSWWCv2G,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_chunked_sswwc_v2g_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::SSWWCv2G,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_sswwc_v2g_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::SSWWCv2G,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_sswwc_v2g_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::SSWWCv2G,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v4_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::HSSWWCv4,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v4_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::HSSWWCv4,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_contiguous_hsswwc_v4_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Contiguous),
        GpuRadixPartitionAlgorithm::HSSWWCv4,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_hsswwc_v4_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::HSSWWCv4,
        RadixBits::from(2),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_chunked_hsswwc_v4_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::HSSWWCv4,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_verify_partitions_contiguous_hsswwc_v4_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        (32 << 20) / mem::size_of::<i32>(),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Contiguous),
        GpuRadixPartitionAlgorithm::HSSWWCv4,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_chunked_hsswwc_v4_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::HSSWWCv4,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_chunked_hsswwc_v4_non_power_two() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
        GpuRadixPartitionAlgorithm::HSSWWCv4,
        RadixBits::from(10),
        GridSize::from(1),
        BlockSize::from(128),
        Box::new(&verify_partitions),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_copy_with_payload_contiguous_i32_2_bits(
) -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning_and_copy_with_payload(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(|pass, rb: &_, dk: &_, dp: &_, pr: &_, _: &_, _: &_| {
            tuple_loss_or_duplicates(pass, rb, dk, dp, pr, None)
        }),
    )
}

#[test]
fn gpu_tuple_loss_or_duplicates_copy_with_payload_contiguous_i32_10_bits(
) -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning_and_copy_with_payload(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(|pass, rb: &_, dk: &_, dp: &_, pr: &_, _: &_, _: &_| {
            tuple_loss_or_duplicates(pass, rb, dk, dp, pr, None)
        }),
    )
}

#[test]
fn gpu_verify_partitions_copy_with_payload_contiguous_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning_and_copy_with_payload(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(|pass, rb: &_, dk: &_, dp: &_, pr: &_, _: &_, _: &_| {
            verify_partitions(pass, rb, dk, dp, pr, None)
        }),
    )
}

#[test]
fn gpu_verify_partitions_copy_with_payload_contiguous_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning_and_copy_with_payload(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(|pass, rb: &_, dk: &_, dp: &_, pr: &_, _: &_, _: &_| {
            verify_partitions(pass, rb, dk, dp, pr, None)
        }),
    )
}

#[test]
fn gpu_check_copy_with_payload_contiguous_i32_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning_and_copy_with_payload(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(2),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&check_copy_with_payload),
    )
}

#[test]
fn gpu_check_copy_with_payload_contiguous_i32_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_partitioning_and_copy_with_payload(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::from(10),
        GridSize::from(10),
        BlockSize::from(128),
        Box::new(&check_copy_with_payload),
    )
}

#[test]
fn gpu_loss_or_duplicates_two_pass_chunked_contiguous_i32_0_0_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Chunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(0), Some(0), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&two_pass_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_loss_or_duplicates_two_pass_chunked_contiguous_i32_2_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Chunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(2), Some(2), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&two_pass_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_loss_or_duplicates_two_pass_chunked_contiguous_i32_10_10_bits() -> Result<(), Box<dyn Error>>
{
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Chunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(10), Some(10), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&two_pass_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_loss_or_duplicates_two_pass_contiguous_contiguous_i32_2_2_bits() -> Result<(), Box<dyn Error>>
{
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(2), Some(2), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&two_pass_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_two_pass_chunked_contiguous_i32_0_0_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Chunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(0), Some(0), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&two_pass_verify_partitions),
    )
}

#[test]
fn gpu_verify_two_pass_chunked_contiguous_i32_2_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Chunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(2), Some(2), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&two_pass_verify_partitions),
    )
}

#[test]
fn gpu_verify_two_pass_chunked_contiguous_i32_10_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Chunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(10), Some(10), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&two_pass_verify_partitions),
    )
}

#[test]
fn gpu_verify_two_pass_contiguous_contiguous_i32_2_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(2), Some(2), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&two_pass_verify_partitions),
    )
}

#[test]
fn gpu_transform_two_pass_chunked_contiguous_i32_0_0_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Chunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(0), Some(0), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&verify_transformed_input),
    )
}

#[test]
fn gpu_transform_two_pass_chunked_contiguous_i32_2_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Chunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(2), Some(2), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&verify_transformed_input),
    )
}

#[test]
fn gpu_transform_two_pass_chunked_contiguous_i32_10_10_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Chunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(10), Some(10), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&verify_transformed_input),
    )
}

#[test]
fn gpu_transform_two_pass_contiguous_contiguous_i32_2_2_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::NC,
        RadixBits::new(Some(2), Some(2), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&verify_transformed_input),
    )
}

#[test]
fn gpu_loss_or_duplicates_two_pass_laswwc_i32_6_6_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Chunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::new(Some(6), Some(6), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&two_pass_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_two_pass_laswwc_i32_6_6_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Chunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::LASWWC,
        RadixBits::new(Some(6), Some(6), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&two_pass_verify_partitions),
    )
}

#[test]
fn gpu_loss_or_duplicates_two_pass_sswwc_v2_i32_6_6_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_primary_key(keys, None)?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Chunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::new(Some(6), Some(6), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&two_pass_tuple_loss_or_duplicates),
    )
}

#[test]
fn gpu_verify_partitions_two_pass_sswwc_v2_i32_6_6_bits() -> Result<(), Box<dyn Error>> {
    run_gpu_two_pass_partitioning(
        10_usize.pow(6),
        Box::new(|keys: &mut _| Ok(UniformRelation::gen_attr(keys, 0..(32 << 20))?)),
        Box::new(|pays: &mut _| Ok(UniformRelation::gen_attr(pays, 0..10000)?)),
        GpuHistogramAlgorithm::Chunked,
        GpuRadixPartitionAlgorithm::NC,
        GpuHistogramAlgorithm::Contiguous,
        GpuRadixPartitionAlgorithm::SSWWCv2,
        RadixBits::new(Some(6), Some(6), None),
        GridSize::from(8),
        BlockSize::from(128),
        Box::new(&two_pass_verify_partitions),
    )
}
