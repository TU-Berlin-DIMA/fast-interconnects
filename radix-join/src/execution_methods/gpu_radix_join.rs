/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2020-2021, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use crate::error::{ErrorKind, Result};
use crate::measurement::harness::RadixJoinPoint;
use data_store::join_data::JoinData;
use numa_gpu::error::Result as NumaGpuResult;
use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemType};
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::cuda_wrapper;
use numa_gpu::runtime::linux_wrapper;
use numa_gpu::runtime::memory::*;
use rustacuda::context::{CacheConfig, CurrentContext, SharedMemoryConfig};
use rustacuda::event::{Event, EventFlags};
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::{CopyDestination, DeviceBuffer, DeviceCopy};
use rustacuda::stream::{Stream, StreamFlags, StreamWaitEventFlags};
use sql_ops::join::{cuda_radix_join, no_partitioning_join, HashingScheme};
use sql_ops::partition::cpu_radix_partition::{
    CpuHistogramAlgorithm, CpuRadixPartitionAlgorithm, CpuRadixPartitionable, CpuRadixPartitioner,
};
use sql_ops::partition::gpu_radix_partition::{
    GpuHistogramAlgorithm, GpuRadixPartitionAlgorithm, GpuRadixPartitionable, GpuRadixPartitioner,
};
use sql_ops::partition::{
    PartitionOffsets, PartitionedRelation, RadixBits, RadixPartitionInputChunkable, RadixPass,
    Tuple,
};
use std::cmp;
use std::convert::TryInto;
use std::iter;
use std::mem;
use std::sync::Arc;
use std::time::Instant;

// Helper struct that stores state of 2nd partitioning passes. Memory
// allocations occur asynchronously in parallel to partitioning.
struct StreamState<T: DeviceCopy> {
    stream: Stream,
    event: Event,
    radix_prnr_2nd: GpuRadixPartitioner,
    radix_join: cuda_radix_join::CudaRadixJoin,
    cached_inner_key: Mem<T>,
    cached_inner_pay: Mem<T>,
    cached_outer_key: Mem<T>,
    cached_outer_pay: Mem<T>,
    inner_rel_partition_offsets_2nd: PartitionOffsets<Tuple<T, T>>,
    outer_rel_partition_offsets_2nd: PartitionOffsets<Tuple<T, T>>,
    inner_rel_partitions_2nd: PartitionedRelation<Tuple<T, T>>,
    outer_rel_partitions_2nd: PartitionedRelation<Tuple<T, T>>,
    join_task_assignments: Mem<u32>,
    join_result_sums: DeviceBuffer<i64>,
}

impl<T: DeviceCopy> MemLock for StreamState<T> {
    fn mlock(&mut self) -> NumaGpuResult<()> {
        self.cached_inner_key.mlock()?;
        self.cached_inner_pay.mlock()?;
        self.cached_outer_key.mlock()?;
        self.cached_outer_pay.mlock()?;
        self.inner_rel_partition_offsets_2nd.mlock()?;
        self.outer_rel_partition_offsets_2nd.mlock()?;
        self.inner_rel_partitions_2nd.mlock()?;
        self.outer_rel_partitions_2nd.mlock()?;
        self.join_task_assignments.mlock()?;

        Ok(())
    }

    fn munlock(&mut self) -> NumaGpuResult<()> {
        unimplemented!();
    }
}

pub fn gpu_radix_join<T>(
    data: &mut JoinData<T>,
    hashing_scheme: HashingScheme,
    histogram_algorithms: [GpuHistogramAlgorithm; 2],
    partition_algorithms: [GpuRadixPartitionAlgorithm; 2],
    radix_bits: &RadixBits,
    dmem_buffer_bytes: usize,
    threads: usize,
    cpu_affinity: CpuAffinity,
    partitions_mem_type: MemType,
    stream_state_mem_type: MemType,
    partition_dim: (&GridSize, &BlockSize),
    join_dim: (&GridSize, &BlockSize),
) -> Result<(i64, RadixJoinPoint)>
where
    T: Default
        + DeviceCopy
        + Sync
        + Send
        + CpuRadixPartitionable
        + GpuRadixPartitionable
        + no_partitioning_join::NullKey
        + no_partitioning_join::CudaHashJoinable
        + no_partitioning_join::CpuHashJoinable
        + cuda_radix_join::CudaRadixJoinable,
{
    const NUM_STREAMS: usize = 2;

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

    let grid_size = &partition_dim.0;
    let stream_grid_size = &join_dim.0;
    let stream_block_size = &join_dim.1;
    let block_size = &partition_dim.1;
    let max_chunks_1st = grid_size.x;
    let max_chunks_2nd = stream_grid_size.x;
    let join_result_sums_len = (stream_grid_size.x * stream_block_size.x) as usize;

    let mut radix_prnr = GpuRadixPartitioner::new(
        histogram_algorithms[0],
        partition_algorithms[0],
        radix_bits.clone(),
        grid_size,
        block_size,
        dmem_buffer_bytes,
    )?;

    // FIXME: Enable GPU memory caching of partitioned relation
    //  - Calculate how much space is left over in device memory for partitioned relations
    //  - Allocate cached keys/payload, partitioned relations, etc
    let mut inner_rel_partitions = PartitionedRelation::new(
        data.build_relation_key.len(),
        histogram_algorithms[0].into(),
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        max_chunks_1st,
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
    );

    let mut outer_rel_partitions = PartitionedRelation::new(
        data.probe_relation_key.len(),
        histogram_algorithms[0].into(),
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        max_chunks_1st,
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
    );

    let mut inner_rel_partition_offsets = PartitionOffsets::new(
        histogram_algorithms[0].into(),
        max_chunks_1st,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
    );

    let mut outer_rel_partition_offsets = PartitionOffsets::new(
        histogram_algorithms[0].into(),
        max_chunks_1st,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
    );

    inner_rel_partitions.mlock()?;
    outer_rel_partitions.mlock()?;
    inner_rel_partition_offsets.mlock()?;
    outer_rel_partition_offsets.mlock()?;

    stream.synchronize()?;

    let partitions_malloc_time = partitions_malloc_timer.elapsed();
    let prefix_sum_timer = Instant::now();

    match histogram_algorithms[0] {
        GpuHistogramAlgorithm::CpuChunked => {
            let inner_key_slice: &[T] = (&data.build_relation_key).try_into().map_err(|_| {
                ErrorKind::RuntimeError("Failed to run CPU prefix sum on device memory".into())
            })?;
            let inner_key_chunks = inner_key_slice.input_chunks::<T>(max_chunks_1st)?;
            let inner_offsets_chunks = inner_rel_partition_offsets.chunks_mut();

            let outer_key_slice: &[T] = (&data.probe_relation_key).try_into().map_err(|_| {
                ErrorKind::RuntimeError("Failed to run CPU prefix sum on device memory".into())
            })?;
            let outer_key_chunks = outer_key_slice.input_chunks::<T>(max_chunks_1st)?;
            let outer_offsets_chunks = outer_rel_partition_offsets.chunks_mut();

            thread_pool.scope(|s| {
                // First outer and then inner, because inner is smaller. Both have equal amount
                // of chunks, thus inner chunks are smaller. Scheduling the smaller chunks last
                // potentially mitigates stragglers.
                for (input, output) in outer_key_chunks
                    .into_iter()
                    .zip(outer_offsets_chunks)
                    .chain(inner_key_chunks.into_iter().zip(inner_offsets_chunks))
                {
                    s.spawn(move |_| {
                        let cpu_id = CpuAffinity::get_cpu().expect("Failed to get CPU ID");
                        let local_node: u16 = linux_wrapper::numa_node_of_cpu(cpu_id)
                            .expect("Failed to map CPU to NUMA node");
                        // FIXME: don't hard-code histogram algorithm
                        let mut radix_prnr = CpuRadixPartitioner::new(
                            CpuHistogramAlgorithm::Chunked,
                            CpuRadixPartitionAlgorithm::NC,
                            radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
                            DerefMemType::NumaMem(local_node, None),
                        );
                        radix_prnr
                            .prefix_sum(input, output)
                            .expect("Failed to run CPU prefix sum");
                    })
                }
            });
        }
        _ => {
            radix_prnr.prefix_sum(
                RadixPass::First,
                data.build_relation_key.as_launchable_slice(),
                &mut inner_rel_partition_offsets,
                &stream,
            )?;
            radix_prnr.prefix_sum(
                RadixPass::First,
                data.probe_relation_key.as_launchable_slice(),
                &mut outer_rel_partition_offsets,
                &stream,
            )?;
        }
    }

    let prefix_sum_event = Event::new(EventFlags::DEFAULT)?;
    prefix_sum_event.record(&stream)?;

    // Partition inner relation
    radix_prnr.partition(
        RadixPass::First,
        data.build_relation_key.as_launchable_slice(),
        data.build_relation_payload.as_launchable_slice(),
        &mut inner_rel_partition_offsets,
        &mut inner_rel_partitions,
        &stream,
    )?;

    // Partition outer relation
    radix_prnr.partition(
        RadixPass::First,
        data.probe_relation_key.as_launchable_slice(),
        data.probe_relation_payload.as_launchable_slice(),
        &mut outer_rel_partition_offsets,
        &mut outer_rel_partitions,
        &stream,
    )?;

    // Wait for prefix sum to finish before computing max partition lengths and
    // stopping the prefix sum timer.
    prefix_sum_event.synchronize()?;

    // Stop the prefix sum timer and start the partitioning timer.
    //
    // The timers are stopped/started after scheduling the partitioning kernels.
    // This enables scheduling the partitioning kernels asynchronously to the
    // running prefix sum kernels.
    //
    // The reason is that some partitioning variants (e.g., HSSWWCv4) allocate
    // memory, and these allocations should occur in parallel to prefix sum
    // computation.
    //
    // We can't do the timing with CUDA events because some prefix sum variants
    // run on the CPU instead of the GPU.
    //
    // Doing the timer stop/start in a CUDA callback on the stream works, but
    // lauching callbacks is relatively slow (~2 ms). Guessing that this is
    // because CUDA lauches a new thread for the callback, but didn't verify.
    let prefix_sum_time = prefix_sum_timer.elapsed();
    let partition_timer = Instant::now();

    let max_inner_partition_len =
        (0..inner_rel_partitions.fanout()).try_fold(0, |max, partition_id| {
            inner_rel_partitions
                .partition_len(partition_id)
                .map(|len| cmp::max(max, len))
        })?;
    let max_outer_partition_len =
        (0..outer_rel_partitions.fanout()).try_fold(0, |max, partition_id| {
            outer_rel_partitions
                .partition_len(partition_id)
                .map(|len| cmp::max(max, len))
        })?;

    // Memory allocations occur asynchronously in parallel to partitioning
    let mut stream_states = iter::repeat_with(|| {
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let mut join_result_sums = unsafe { DeviceBuffer::uninitialized(join_result_sums_len)? };
        cuda_wrapper::memset_async(join_result_sums.as_launchable_mut_slice(), 0, &stream)?;

        Ok(StreamState {
            stream,
            event: Event::new(EventFlags::DEFAULT)?,
            radix_prnr_2nd: GpuRadixPartitioner::new(
                histogram_algorithms[1],
                partition_algorithms[1],
                radix_bits.clone(),
                stream_grid_size,
                stream_block_size,
                dmem_buffer_bytes,
            )?,
            radix_join: cuda_radix_join::CudaRadixJoin::new(
                RadixPass::Second,
                radix_bits.clone(),
                hashing_scheme,
                stream_grid_size,
                stream_block_size,
            )?,
            cached_inner_key: Allocator::alloc_mem(
                stream_state_mem_type.clone(),
                max_inner_partition_len,
            ),
            cached_inner_pay: Allocator::alloc_mem(
                stream_state_mem_type.clone(),
                max_inner_partition_len,
            ),
            cached_outer_key: Allocator::alloc_mem(
                stream_state_mem_type.clone(),
                max_outer_partition_len,
            ),
            cached_outer_pay: Allocator::alloc_mem(
                stream_state_mem_type.clone(),
                max_outer_partition_len,
            ),
            inner_rel_partition_offsets_2nd: PartitionOffsets::new(
                histogram_algorithms[1].into(),
                max_chunks_2nd,
                radix_bits.pass_radix_bits(RadixPass::Second).unwrap(),
                Allocator::mem_alloc_fn(stream_state_mem_type.clone()),
            ),
            outer_rel_partition_offsets_2nd: PartitionOffsets::new(
                histogram_algorithms[1].into(),
                max_chunks_2nd,
                radix_bits.pass_radix_bits(RadixPass::Second).unwrap(),
                Allocator::mem_alloc_fn(stream_state_mem_type.clone()),
            ),
            inner_rel_partitions_2nd: PartitionedRelation::new(
                max_inner_partition_len,
                histogram_algorithms[1].into(),
                radix_bits.pass_radix_bits(RadixPass::Second).unwrap(),
                max_chunks_2nd,
                Allocator::mem_alloc_fn(stream_state_mem_type.clone()),
                Allocator::mem_alloc_fn(stream_state_mem_type.clone()),
            ),
            outer_rel_partitions_2nd: PartitionedRelation::new(
                max_outer_partition_len,
                histogram_algorithms[1].into(),
                radix_bits.pass_radix_bits(RadixPass::Second).unwrap(),
                max_chunks_2nd,
                Allocator::mem_alloc_fn(stream_state_mem_type.clone()),
                Allocator::mem_alloc_fn(stream_state_mem_type.clone()),
            ),
            join_task_assignments: Allocator::alloc_mem(
                stream_state_mem_type.clone(),
                join_dim.0.x as usize + 1,
            ),
            join_result_sums,
        })
    })
    .take(NUM_STREAMS)
    .collect::<Result<Vec<_>>>()?;

    stream.synchronize()?;
    Stream::drop(stream).map_err(|(e, _)| e)?;

    let partition_time = partition_timer.elapsed();
    let state_malloc_timer = Instant::now();

    // Populate the pages with mlock(); ideally this would be taken care of by a
    // NUMA-aware malloc implementation
    stream_states
        .iter_mut()
        .try_for_each(|state| state.mlock())?;

    let state_malloc_time = state_malloc_timer.elapsed();
    let join_timer = Instant::now();

    for (partition_id, stream_id) in
        (0..radix_bits.pass_fanout(RadixPass::First).unwrap()).zip((0..stream_states.len()).cycle())
    {
        let StreamState {
            stream,
            event,
            radix_prnr_2nd,
            radix_join,
            cached_inner_key,
            cached_inner_pay,
            cached_outer_key,
            cached_outer_pay,
            inner_rel_partition_offsets_2nd,
            outer_rel_partition_offsets_2nd,
            inner_rel_partitions_2nd,
            outer_rel_partitions_2nd,
            join_task_assignments,
            join_result_sums,
        } = stream_states
            .get_mut(stream_id)
            .ok_or_else(|| ErrorKind::RuntimeError("Failed to get stream state".into()))?;

        let old_event = mem::replace(event, Event::new(EventFlags::DEFAULT)?);
        stream.wait_event(old_event, StreamWaitEventFlags::DEFAULT)?;

        let inner_partition_len = inner_rel_partitions.partition_len(partition_id)?;
        let outer_partition_len = outer_rel_partitions.partition_len(partition_id)?;

        inner_rel_partitions_2nd.resize(inner_partition_len)?;
        outer_rel_partitions_2nd.resize(outer_partition_len)?;

        let cached_inner_key_slice = unsafe {
            &mut cached_inner_key.as_launchable_mut_slice().as_mut_slice()[0..inner_partition_len]
        };
        let cached_inner_pay_slice = unsafe {
            &mut cached_inner_pay.as_launchable_mut_slice().as_mut_slice()[0..inner_partition_len]
        };
        let cached_outer_key_slice = unsafe {
            &mut cached_outer_key.as_launchable_mut_slice().as_mut_slice()[0..outer_partition_len]
        };
        let cached_outer_pay_slice = unsafe {
            &mut cached_outer_pay.as_launchable_mut_slice().as_mut_slice()[0..outer_partition_len]
        };

        radix_prnr_2nd.prefix_sum_and_transform(
            RadixPass::Second,
            partition_id,
            &inner_rel_partitions,
            cached_inner_key_slice.as_launchable_mut_slice(),
            cached_inner_pay_slice.as_launchable_mut_slice(),
            inner_rel_partition_offsets_2nd,
            stream,
        )?;

        radix_prnr_2nd.prefix_sum_and_transform(
            RadixPass::Second,
            partition_id,
            &outer_rel_partitions,
            cached_outer_key_slice.as_launchable_mut_slice(),
            cached_outer_pay_slice.as_launchable_mut_slice(),
            outer_rel_partition_offsets_2nd,
            stream,
        )?;

        radix_prnr_2nd.partition(
            RadixPass::Second,
            cached_inner_key_slice.as_launchable_slice(),
            cached_inner_pay_slice.as_launchable_slice(),
            inner_rel_partition_offsets_2nd,
            inner_rel_partitions_2nd,
            stream,
        )?;

        radix_prnr_2nd.partition(
            RadixPass::Second,
            cached_outer_key_slice.as_launchable_slice(),
            cached_outer_pay_slice.as_launchable_slice(),
            outer_rel_partition_offsets_2nd,
            outer_rel_partitions_2nd,
            stream,
        )?;

        radix_join.join(
            &inner_rel_partitions_2nd,
            &outer_rel_partitions_2nd,
            &mut join_result_sums.as_launchable_mut_slice(),
            &mut join_task_assignments.as_launchable_mut_slice(),
            stream,
        )?;

        event.record(stream)?;
    }

    let mut result_sums_host = vec![0; join_result_sums_len * NUM_STREAMS];

    stream_states
        .into_iter()
        .zip(result_sums_host.chunks_mut(join_result_sums_len))
        .map(
            |(
                StreamState {
                    stream,
                    join_result_sums,
                    ..
                },
                host_sums,
            )| {
                stream.synchronize()?;
                join_result_sums.copy_to(host_sums)?;
                Ok(())
            },
        )
        .collect::<Result<()>>()?;

    let join_time = join_timer.elapsed();

    let sum = result_sums_host.iter().sum();

    let data_point = RadixJoinPoint {
        prefix_sum_ns: Some(prefix_sum_time.as_nanos() as f64),
        partition_ns: Some(partition_time.as_nanos() as f64),
        join_ns: Some(join_time.as_nanos() as f64),
        partitions_malloc_ns: Some(partitions_malloc_time.as_nanos() as f64),
        state_malloc_ns: Some(state_malloc_time.as_nanos() as f64),
    };

    Ok((sum, data_point))
}
