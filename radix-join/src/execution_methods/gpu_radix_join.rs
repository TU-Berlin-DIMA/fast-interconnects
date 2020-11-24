/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2020, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use crate::error::{ErrorKind, Result};
use crate::measurement::harness::RadixJoinPoint;
use data_store::join_data::JoinData;
use numa_gpu::runtime::allocator::{self, Allocator};
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::memory::*;
use rustacuda::context::{CacheConfig, CurrentContext, SharedMemoryConfig};
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::DeviceCopy;
use rustacuda::stream::{Stream, StreamFlags};
use sql_ops::join::{cuda_radix_join, no_partitioning_join, HashingScheme};
use sql_ops::partition::gpu_radix_partition::{
    GpuHistogramAlgorithm, GpuRadixPartitionAlgorithm, GpuRadixPartitionable, GpuRadixPartitioner,
    PartitionOffsets, PartitionedRelation, RadixPartitionInputChunkable,
};
use std::convert::TryInto;
use std::sync::Arc;
use std::time::Instant;

pub fn gpu_radix_join<T>(
    data: &mut JoinData<T>,
    hashing_scheme: HashingScheme,
    histogram_algorithm: GpuHistogramAlgorithm,
    partition_algorithm: GpuRadixPartitionAlgorithm,
    radix_bits: u32,
    dmem_buffer_bytes: usize,
    threads: usize,
    cpu_affinity: CpuAffinity,
    partitions_mem_type: allocator::MemType,
    partition_fst_dim: (GridSize, BlockSize),
    _partition_snd_dim: (GridSize, BlockSize),
    join_dim: (GridSize, BlockSize),
) -> Result<RadixJoinPoint>
where
    T: Default
        + DeviceCopy
        + Sync
        + Send
        + GpuRadixPartitionable
        + no_partitioning_join::NullKey
        + no_partitioning_join::CudaHashJoinable
        + no_partitioning_join::CpuHashJoinable
        + cuda_radix_join::CudaRadixJoinable,
{
    CurrentContext::set_cache_config(CacheConfig::PreferShared)?;
    CurrentContext::set_shared_memory_config(SharedMemoryConfig::FourByteBankSize)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let boxed_cpu_affinity = Arc::new(cpu_affinity);
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .start_handler(move |tid| {
            boxed_cpu_affinity
                .clone()
                .set_affinity(tid as u16)
                .expect("Couldn't set CPU core affinity")
        })
        .build()?;

    let partitions_malloc_timer = Instant::now();

    let mut radix_prnr = GpuRadixPartitioner::new(
        histogram_algorithm,
        partition_algorithm,
        radix_bits,
        &partition_fst_dim.0,
        &partition_fst_dim.1,
        dmem_buffer_bytes,
    )?;

    let mut inner_rel_partitions = PartitionedRelation::new(
        data.build_relation_key.len(),
        histogram_algorithm,
        radix_bits,
        partition_fst_dim.0.x,
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
    );

    let mut outer_rel_partitions = PartitionedRelation::new(
        data.probe_relation_key.len(),
        histogram_algorithm,
        radix_bits,
        partition_fst_dim.0.x,
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
    );

    let mut inner_rel_partition_offsets = PartitionOffsets::new(
        histogram_algorithm,
        partition_fst_dim.0.x,
        radix_bits,
        Allocator::mem_alloc_fn(allocator::MemType::CudaUniMem),
    );

    let mut outer_rel_partition_offsets = PartitionOffsets::new(
        histogram_algorithm,
        partition_fst_dim.0.x,
        radix_bits,
        Allocator::mem_alloc_fn(allocator::MemType::CudaUniMem),
    );

    let mut result_sums = allocator::Allocator::alloc_mem(
        allocator::MemType::CudaUniMem,
        (join_dim.0.x * join_dim.1.x) as usize,
    );

    // Initialize result
    if let CudaUniMem(ref mut c) = result_sums {
        c.iter_mut().map(|sum| *sum = 0).for_each(drop);
    }

    let partitions_malloc_time = partitions_malloc_timer.elapsed();

    stream.synchronize()?;

    let prefix_sum_timer = Instant::now();

    match histogram_algorithm {
        GpuHistogramAlgorithm::CpuChunked => {
            let inner_key_slice: &[T] = (&data.build_relation_key).try_into().map_err(|_| {
                ErrorKind::RuntimeError("Failed to run CPU prefix sum on device memory".into())
            })?;
            let inner_key_chunks = inner_key_slice.input_chunks::<T>(&radix_prnr)?;
            let inner_offsets_chunks = inner_rel_partition_offsets.chunks_mut();

            let outer_key_slice: &[T] = (&data.probe_relation_key).try_into().map_err(|_| {
                ErrorKind::RuntimeError("Failed to run CPU prefix sum on device memory".into())
            })?;
            let outer_key_chunks = outer_key_slice.input_chunks::<T>(&radix_prnr)?;
            let outer_offsets_chunks = outer_rel_partition_offsets.chunks_mut();

            thread_pool.scope(|s| {
                // First outer and then inner, because inner is smaller. Both have equal amount
                // of chunks, thus inner chunks are smaller. Scheduling the smaller chunks last
                // potentially mitigates stragglers.
                for (input, mut output) in outer_key_chunks
                    .iter()
                    .zip(outer_offsets_chunks)
                    .chain(inner_key_chunks.iter().zip(inner_offsets_chunks))
                {
                    let radix_prnr_ref = &radix_prnr;
                    s.spawn(move |_| {
                        radix_prnr_ref
                            .cpu_prefix_sum(input, &mut output)
                            .expect("Failed to run CPU prefix sum");
                    })
                }
            });
        }
        _ => {
            radix_prnr.prefix_sum(
                data.build_relation_key.as_launchable_slice(),
                &mut inner_rel_partition_offsets,
                &stream,
            )?;
            radix_prnr.prefix_sum(
                data.probe_relation_key.as_launchable_slice(),
                &mut outer_rel_partition_offsets,
                &stream,
            )?;
        }
    }

    stream.synchronize()?;
    let prefix_sum_time = prefix_sum_timer.elapsed();

    let partition_timer = Instant::now();

    // Partition inner relation
    radix_prnr.partition(
        data.build_relation_key.as_launchable_slice(),
        data.build_relation_payload.as_launchable_slice(),
        inner_rel_partition_offsets,
        &mut inner_rel_partitions,
        &stream,
    )?;

    // Partition outer relation
    radix_prnr.partition(
        data.probe_relation_key.as_launchable_slice(),
        data.probe_relation_payload.as_launchable_slice(),
        outer_rel_partition_offsets,
        &mut outer_rel_partitions,
        &stream,
    )?;

    stream.synchronize()?;
    let partition_time = partition_timer.elapsed();

    let join_timer = Instant::now();

    let mut join_task_assignments =
        allocator::Allocator::alloc_mem(allocator::MemType::CudaDevMem, join_dim.0.x as usize);
    let radix_join = cuda_radix_join::CudaRadixJoin::new(hashing_scheme, join_dim)?;
    radix_join.join(
        radix_bits,
        &inner_rel_partitions,
        &outer_rel_partitions,
        &mut result_sums.as_launchable_mut_slice(),
        &mut join_task_assignments.as_launchable_mut_slice(),
        &stream,
    )?;

    stream.synchronize()?;
    let join_time = join_timer.elapsed();

    Ok(RadixJoinPoint {
        prefix_sum_ns: Some(prefix_sum_time.as_nanos() as f64),
        partition_ns: Some(partition_time.as_nanos() as f64),
        join_ns: Some(join_time.as_nanos() as f64),
        partitions_malloc_ns: Some(partitions_malloc_time.as_nanos() as f64),
    })
}
