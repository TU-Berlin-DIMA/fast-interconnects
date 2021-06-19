/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use crate::error::{ErrorKind, Result};
use crate::measurement::harness::RadixJoinPoint;
use data_store::join_data::JoinData;
use datagen::relation::KeyAttribute;
use itertools::izip;
use numa_gpu::error::Result as NumaGpuResult;
use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemType};
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::cuda_wrapper;
use numa_gpu::runtime::hw_info::NvidiaDriverInfo;
use numa_gpu::runtime::linux_wrapper;
use numa_gpu::runtime::memory::*;
use numa_gpu::runtime::numa::PageType;
use numa_gpu::utils::DeviceType;
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

/// GPU memory to leave free when allocating the Triton join's cache
///
/// Getting the amount of free GPU memory and allocating the memory is
/// *not atomic*. Sometimes the allocation spuriously fails.
///
/// Instead of allocating every last byte of GPU memory, leave some slack space.
const GPU_MEM_SLACK_BYTES: usize = 32 * 1024 * 1024;

/// GPU memory to leave free between the partitioned relations
///
/// The cached partitioned relations are allocated individually. The CUDA seems
/// to have some external fragementation, presumably due to page alignment of
/// allocations.
///
/// The overhead needs to be taken into account when calculating the maximum
/// number of partitioned relations that can be cached in GPU memory.
///
/// Note that this is a quick-fix. A better implementation would allocate the
/// 2nd pass partitioned relations in one or more arrays instead of one-by-one.
/// However, a single array wouldn't be sufficient, as not all relations can be
/// cached.
const GPU_MEM_RELATION_SLACK_BYTES: usize = 2 * 1024 * 1024;

// Helper struct that stores state of 2nd partitioning passes. Memory
// allocations occur asynchronously in parallel to partitioning.
//
// FIXME: refactor the StreamState into an seperate file, and deduplicate
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

pub fn cpu_partitioned_radix_join<T>(
    data: &mut JoinData<T>,
    hashing_scheme: HashingScheme,
    histogram_algorithm_fst: DeviceType<CpuHistogramAlgorithm, GpuHistogramAlgorithm>,
    histogram_algorithm_snd: DeviceType<CpuHistogramAlgorithm, GpuHistogramAlgorithm>,
    partition_algorithm_fst: DeviceType<CpuRadixPartitionAlgorithm, GpuRadixPartitionAlgorithm>,
    partition_algorithm_snd: DeviceType<CpuRadixPartitionAlgorithm, GpuRadixPartitionAlgorithm>,
    radix_bits: &RadixBits,
    dmem_buffer_bytes: usize,
    max_partitions_cache_bytes: Option<usize>,
    threads: usize,
    cpu_affinity: CpuAffinity,
    partitions_mem_type: MemType,
    stream_state_mem_type: MemType,
    _page_type: PageType,
    _partition_dim: (&GridSize, &BlockSize),
    join_dim: (&GridSize, &BlockSize),
) -> Result<(i64, RadixJoinPoint)>
where
    T: Default
        + Clone
        + DeviceCopy
        + Sync
        + Send
        + CpuRadixPartitionable
        + GpuRadixPartitionable
        + KeyAttribute
        + no_partitioning_join::CudaHashJoinable
        + no_partitioning_join::CpuHashJoinable
        + cuda_radix_join::CudaRadixJoinable,
{
    const NUM_STREAMS: usize = 2;

    // Precondition checks
    let histogram_algorithm_snd = histogram_algorithm_snd.gpu().ok_or_else(|| {
        ErrorKind::InvalidArgument("Only GPU prefix sum is supported in 2nd pass".to_string())
    })?;
    let partition_algorithm_fst = partition_algorithm_fst.cpu().ok_or_else(|| {
        ErrorKind::InvalidArgument("Only CPU partitioning is supported in 1st pass".to_string())
    })?;
    let partition_algorithm_snd = partition_algorithm_snd.gpu().ok_or_else(|| {
        ErrorKind::InvalidArgument("Only GPU partitioning is supported in 2nd pass".to_string())
    })?;

    CurrentContext::set_cache_config(CacheConfig::PreferShared)?;
    CurrentContext::set_shared_memory_config(SharedMemoryConfig::FourByteBankSize)?;

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

    let stream_grid_size = &join_dim.0;
    let stream_block_size = &join_dim.1;
    let max_chunks_1st = threads as u32;
    let max_chunks_2nd = stream_grid_size.x;
    let join_result_sums_len = (stream_grid_size.x * stream_block_size.x) as usize;

    let mut radix_prnrs: Vec<_> = (0..threads)
        .map(|_| {
            Ok(CpuRadixPartitioner::new(
                histogram_algorithm_fst.cpu().ok_or_else(|| {
                    ErrorKind::InvalidArgument("Only CPU prefix sum is supported".to_string())
                })?,
                partition_algorithm_fst,
                radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
                DerefMemType::AlignedSysMem {
                    align_bytes: sql_ops::CPU_CACHE_LINE_SIZE as usize,
                },
            ))
        })
        .collect::<Result<_>>()?;

    let mut inner_rel_partitions = PartitionedRelation::new(
        data.build_relation_key.len(),
        histogram_algorithm_fst.either(|cpu| cpu.into(), |gpu| gpu.into()),
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        max_chunks_1st,
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
    );

    let mut outer_rel_partitions = PartitionedRelation::new(
        data.probe_relation_key.len(),
        histogram_algorithm_fst.either(|cpu| cpu.into(), |gpu| gpu.into()),
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        max_chunks_1st,
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
    );

    let mut inner_rel_partition_offsets = PartitionOffsets::new(
        histogram_algorithm_fst.either(|cpu| cpu.into(), |gpu| gpu.into()),
        max_chunks_1st,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
    );

    let mut outer_rel_partition_offsets = PartitionOffsets::new(
        histogram_algorithm_fst.either(|cpu| cpu.into(), |gpu| gpu.into()),
        max_chunks_1st,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        Allocator::mem_alloc_fn(partitions_mem_type.clone()),
    );

    inner_rel_partitions.mlock()?;
    outer_rel_partitions.mlock()?;
    inner_rel_partition_offsets.mlock()?;
    outer_rel_partition_offsets.mlock()?;

    let partitions_malloc_time = partitions_malloc_timer.elapsed();
    let prefix_sum_timer = Instant::now();

    let inner_key_slice: &[T] = (&data.build_relation_key).try_into().map_err(|_| {
        ErrorKind::RuntimeError("Failed to run CPU prefix sum on device memory".into())
    })?;
    let inner_pay_slice: &[T] = (&data.build_relation_payload).try_into().map_err(|_| {
        ErrorKind::RuntimeError("Failed to run CPU prefix sum on device memory".into())
    })?;
    let inner_key_chunks = inner_key_slice.input_chunks::<T>(max_chunks_1st)?;
    let inner_pay_chunks = inner_pay_slice.input_chunks::<T>(max_chunks_1st)?;

    let outer_key_slice: &[T] = (&data.probe_relation_key).try_into().map_err(|_| {
        ErrorKind::RuntimeError("Failed to run CPU prefix sum on device memory".into())
    })?;
    let outer_pay_slice: &[T] = (&data.probe_relation_payload).try_into().map_err(|_| {
        ErrorKind::RuntimeError("Failed to run CPU prefix sum on device memory".into())
    })?;
    let outer_key_chunks = outer_key_slice.input_chunks::<T>(max_chunks_1st)?;
    let outer_pay_chunks = outer_pay_slice.input_chunks::<T>(max_chunks_1st)?;

    // Prefix sum on inner relation
    thread_pool.scope(|s| {
        for (radix_prnr, key_chunk, offsets_chunk) in izip!(
            radix_prnrs.iter_mut(),
            inner_key_chunks.into_iter(),
            inner_rel_partition_offsets.chunks_mut()
        ) {
            s.spawn(move |_| {
                radix_prnr
                    .prefix_sum(key_chunk, offsets_chunk)
                    .expect("Failed to run prefix sum");
            })
        }
    });

    // Prefix sum on outer relation
    thread_pool.scope(|s| {
        for (radix_prnr, key_chunk, offsets_chunk) in izip!(
            radix_prnrs.iter_mut(),
            outer_key_chunks.into_iter(),
            outer_rel_partition_offsets.chunks_mut()
        ) {
            s.spawn(move |_| {
                radix_prnr
                    .prefix_sum(key_chunk, offsets_chunk)
                    .expect("Failed to run prefix sum");
            })
        }
    });

    let prefix_sum_time = prefix_sum_timer.elapsed().as_nanos() as f64;
    let state_malloc_timer = Instant::now();

    let max_inner_partition_len =
        (0..inner_rel_partitions.fanout()).try_fold(0, |max, partition_id| {
            inner_rel_partition_offsets
                .partition_len(partition_id)
                .map(|len| cmp::max(max, len))
        })?;
    let max_outer_partition_len =
        (0..outer_rel_partitions.fanout()).try_fold(0, |max, partition_id| {
            outer_rel_partition_offsets
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
                histogram_algorithm_snd,
                partition_algorithm_snd,
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
                histogram_algorithm_snd.into(),
                max_chunks_2nd,
                radix_bits.pass_radix_bits(RadixPass::Second).unwrap(),
                Allocator::mem_alloc_fn(stream_state_mem_type.clone()),
            ),
            outer_rel_partition_offsets_2nd: PartitionOffsets::new(
                histogram_algorithm_snd.into(),
                max_chunks_2nd,
                radix_bits.pass_radix_bits(RadixPass::Second).unwrap(),
                Allocator::mem_alloc_fn(stream_state_mem_type.clone()),
            ),
            inner_rel_partitions_2nd: PartitionedRelation::new(
                max_inner_partition_len,
                histogram_algorithm_snd.into(),
                radix_bits.pass_radix_bits(RadixPass::Second).unwrap(),
                max_chunks_2nd,
                Allocator::mem_alloc_fn(stream_state_mem_type.clone()),
                Allocator::mem_alloc_fn(stream_state_mem_type.clone()),
            ),
            outer_rel_partitions_2nd: PartitionedRelation::new(
                max_outer_partition_len,
                histogram_algorithm_snd.into(),
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

    // Populate the pages with mlock(); ideally this would be taken care of by a
    // NUMA-aware malloc implementation
    stream_states
        .iter_mut()
        .try_for_each(|state| state.mlock())?;

    // Get the amount of free GPU memory. The free memory is then used to cache
    // the partitioned relations in GPU memory.
    //
    // In this case, we are trying to capture:
    //  - CUDA context uses several hundred MB memory
    //  - radix_prnr state (e.g., HSSWWC and prefix sum)
    //  - StreamState for all streams
    //
    // Note that CudaMemInfo over-reports how much memory is free. The Linux
    // kernel seems to give a more accurate report (on IBM AC922 with CUDA 10.2).
    let cache_node = CurrentContext::get_device()?.numa_node().map_err(|_| {
        ErrorKind::RuntimeError(
            "Failed to get the GPU NUMA node. Maybe it isn't a cache-coherent GPU?".to_string(),
        )
    })?;
    let linux_wrapper::NumaMemInfo { free, .. } = linux_wrapper::numa_mem_info(cache_node)?;
    let free = free - GPU_MEM_SLACK_BYTES;
    let cache_bytes = max_partitions_cache_bytes.map_or(free, |bytes| {
        if bytes > free {
            eprintln!(
                "Warning: Partitions cache size too large, reducing to maximum available memory"
            );
        }
        cmp::min(bytes, free)
    });

    // Allocate 2nd pass inner relation partitions for overlapping CPU
    // partitioning with transfer and GPU partitioning. The partitions result
    // from the 2nd pass on the GPU.
    let num_cached_inner_partitions = cmp::min(
        radix_bits.pass_fanout(RadixPass::First).unwrap(),
        // FIXME: use the exact length instead of the maximum essentially compute the inclusive
        // prefix sum, and take the maximum that fits into the cache. Or even better: make an
        // actual schedule using the knapsack algorithm
        (cache_bytes
            / (stream_states[0].inner_rel_partitions_2nd.bytes() + GPU_MEM_RELATION_SLACK_BYTES))
            as u32,
    );
    let mut cached_inner_partitions = (0..num_cached_inner_partitions)
        .map(|_| {
            let mut p = PartitionedRelation::<Tuple<T, T>>::new(
                max_inner_partition_len,
                histogram_algorithm_snd.into(),
                radix_bits.pass_radix_bits(RadixPass::Second).unwrap(),
                max_chunks_2nd,
                Allocator::mem_alloc_fn(stream_state_mem_type.clone()),
                Allocator::mem_alloc_fn(stream_state_mem_type.clone()),
            );
            p.mlock()?;
            Ok(p)
        })
        .collect::<Result<Vec<_>>>()?;
    let cached_build_tuples = cached_inner_partitions.iter().map(|p| p.padded_len()).sum();

    let state_malloc_time = state_malloc_timer.elapsed();
    let partition_timer = Instant::now();

    // Partition inner relation
    let inner_key_chunks = inner_key_slice.input_chunks::<T>(max_chunks_1st)?;
    thread_pool.scope(|s| {
        for (radix_prnr, key_chunk, pay_chunk, offsets_chunk, partitioned_chunk) in izip!(
            radix_prnrs.iter_mut(),
            inner_key_chunks.into_iter(),
            inner_pay_chunks.into_iter(),
            inner_rel_partition_offsets.chunks_mut(),
            inner_rel_partitions.chunks_mut()
        ) {
            s.spawn(move |_| {
                radix_prnr
                    .partition(key_chunk, pay_chunk, offsets_chunk, partitioned_chunk)
                    .expect("Failed to partition the data");
            })
        }
    });

    // Enqueue GPU kernels that copy and 2nd pass partition the inner relation
    // to the cache
    for (partition_id, stream_id, inner_rel_partitions_2nd) in izip!(
        0..num_cached_inner_partitions,
        (0..stream_states.len()).cycle(),
        cached_inner_partitions.iter_mut()
    ) {
        let StreamState {
            stream,
            event,
            radix_prnr_2nd,
            cached_inner_key,
            cached_inner_pay,
            inner_rel_partition_offsets_2nd,
            ..
        } = stream_states
            .get_mut(stream_id)
            .ok_or_else(|| ErrorKind::RuntimeError("Failed to get stream state".into()))?;

        let old_event = mem::replace(event, Event::new(EventFlags::DEFAULT)?);
        stream.wait_event(old_event, StreamWaitEventFlags::DEFAULT)?;

        let inner_partition_len = inner_rel_partitions.partition_len(partition_id)?;
        inner_rel_partitions_2nd.resize(inner_partition_len)?;

        let cached_inner_key_slice = unsafe {
            &mut cached_inner_key.as_launchable_mut_slice().as_mut_slice()[0..inner_partition_len]
        };
        let cached_inner_pay_slice = unsafe {
            &mut cached_inner_pay.as_launchable_mut_slice().as_mut_slice()[0..inner_partition_len]
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

        radix_prnr_2nd.partition(
            RadixPass::Second,
            cached_inner_key_slice.as_launchable_slice(),
            cached_inner_pay_slice.as_launchable_slice(),
            inner_rel_partition_offsets_2nd,
            inner_rel_partitions_2nd,
            stream,
        )?;

        event.record(stream)?;
    }

    // Partition outer relation on the CPU in parallel to the transfer and 2nd
    // pass of the inner relation on the GPU.
    let outer_key_chunks = outer_key_slice.input_chunks::<T>(max_chunks_1st).unwrap();
    thread_pool.scope(|s| {
        for (radix_prnr, key_chunk, pay_chunk, offsets_chunk, partitioned_chunk) in izip!(
            radix_prnrs.iter_mut(),
            outer_key_chunks.into_iter(),
            outer_pay_chunks.into_iter(),
            outer_rel_partition_offsets.chunks_mut(),
            outer_rel_partitions.chunks_mut()
        ) {
            s.spawn(move |_| {
                radix_prnr
                    .partition(key_chunk, pay_chunk, offsets_chunk, partitioned_chunk)
                    .expect("Failed to partition the data");
            })
        }
    });

    let partition_time = partition_timer.elapsed();
    let join_timer = Instant::now();

    let join_start_event = Event::new(EventFlags::DEFAULT)?;
    stream_states
        .iter()
        .take(1)
        .try_for_each(|StreamState { stream, .. }| join_start_event.record(stream))?;

    // Note: Outer relation partitions must be enqueued on the same stream as
    // their respective inner relation partition. Thus, need to match up the
    // stream_state cycle.
    for (partition_id, stream_id, inner_rel_partitions_2nd) in izip!(
        0..num_cached_inner_partitions,
        (0..stream_states.len()).cycle(),
        cached_inner_partitions.iter_mut()
    ) {
        let StreamState {
            stream,
            event,
            radix_prnr_2nd,
            radix_join,
            cached_outer_key,
            cached_outer_pay,
            outer_rel_partition_offsets_2nd,
            outer_rel_partitions_2nd,
            join_task_assignments,
            join_result_sums,
            ..
        } = stream_states
            .get_mut(stream_id)
            .ok_or_else(|| ErrorKind::RuntimeError("Failed to get stream state".into()))?;

        let old_event = mem::replace(event, Event::new(EventFlags::DEFAULT)?);
        stream.wait_event(old_event, StreamWaitEventFlags::DEFAULT)?;

        let inner_partition_len = inner_rel_partitions.partition_len(partition_id)?;
        let outer_partition_len = outer_rel_partitions.partition_len(partition_id)?;

        inner_rel_partitions_2nd.resize(inner_partition_len)?;
        outer_rel_partitions_2nd.resize(outer_partition_len)?;

        let cached_outer_key_slice = unsafe {
            &mut cached_outer_key.as_launchable_mut_slice().as_mut_slice()[0..outer_partition_len]
        };
        let cached_outer_pay_slice = unsafe {
            &mut cached_outer_pay.as_launchable_mut_slice().as_mut_slice()[0..outer_partition_len]
        };

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

    for (partition_id, stream_id) in izip!(
        (0..radix_bits.pass_fanout(RadixPass::First).unwrap())
            .skip(num_cached_inner_partitions as usize),
        // Skip the first N stream_states to balance the queues and avoid stragglers
        (0..stream_states.len())
            .cycle()
            .skip(num_cached_inner_partitions as usize)
    ) {
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

    stream_states
        .iter()
        .try_for_each::<_, Result<_>>(|StreamState { stream, .. }| {
            stream.synchronize()?;
            Ok(())
        })?;

    let join_time = join_timer.elapsed();

    let mut result_sums_host = vec![0; join_result_sums_len * NUM_STREAMS];
    stream_states
        .into_iter()
        .zip(result_sums_host.chunks_mut(join_result_sums_len))
        .map(
            |(
                StreamState {
                    join_result_sums, ..
                },
                host_sums,
            )| {
                join_result_sums.copy_to(host_sums)?;
                Ok(())
            },
        )
        .collect::<Result<()>>()?;

    let sum = result_sums_host.iter().sum();

    let data_point = RadixJoinPoint {
        prefix_sum_ns: Some(prefix_sum_time),
        partition_ns: Some(partition_time.as_nanos() as f64),
        join_ns: Some(join_time.as_nanos() as f64),
        partitions_malloc_ns: Some(partitions_malloc_time.as_nanos() as f64),
        state_malloc_ns: Some(state_malloc_time.as_nanos() as f64),
        cached_build_tuples: Some(cached_build_tuples),
        cached_probe_tuples: None,
    };

    Ok((sum, data_point))
}
