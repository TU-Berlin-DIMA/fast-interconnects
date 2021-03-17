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
use numa_gpu::error::Result as NumaGpuResult;
use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemAllocFn, MemType};
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::cuda_wrapper;
use numa_gpu::runtime::hw_info::ProcessorCache;
use numa_gpu::runtime::linux_wrapper;
use numa_gpu::runtime::memory::*;
use numa_gpu::runtime::numa::{NodeLen, PageType};
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
use std::cell::RefCell;
use std::cmp;
use std::convert::TryInto;
use std::iter;
use std::mem;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

/// GPU memory to leave free when allocating the Triton join's cache
///
/// Getting the amount of free GPU memory and allocating the memory is
/// *not atomic*. Sometimes the allocation spuriously fails.
///
/// Instead of allocating every last byte of GPU memory, leave some slack space.
const GPU_MEM_SLACK_BYTES: usize = 32 * 1024 * 1024;

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

pub fn gpu_triton_join<T>(
    data: &mut JoinData<T>,
    hashing_scheme: HashingScheme,
    histogram_algorithm_fst: DeviceType<CpuHistogramAlgorithm, GpuHistogramAlgorithm>,
    histogram_algorithm_snd: GpuHistogramAlgorithm,
    partition_algorithm_fst: GpuRadixPartitionAlgorithm,
    partition_algorithm_snd: GpuRadixPartitionAlgorithm,
    radix_bits: &RadixBits,
    dmem_buffer_bytes: usize,
    threads: usize,
    cpu_affinity: CpuAffinity,
    partitions_mem_type: MemType,
    stream_state_mem_type: MemType,
    page_type: PageType,
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

    let (cache_node, overflow_node) =
        if let MemType::DistributedNumaMem { nodes, .. } = partitions_mem_type {
            if let [cache_node, overflow_node] = *nodes {
                Ok((cache_node.node, overflow_node.node))
            } else {
                Err(ErrorKind::InvalidArgument(
                    "Partitioned memory type define exactly two NUMA nodes".to_string(),
                ))
            }
        } else {
            Err(ErrorKind::InvalidArgument(
                "Partitioned memory type must be DistributedNumaMem".to_string(),
            ))
        }?;

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

    let offsets_mem_type = MemType::NumaMem {
        node: overflow_node,
        page_type,
    };

    let mut radix_prnr = GpuRadixPartitioner::new(
        histogram_algorithm_fst.gpu_or_else(|cpu_algo| cpu_algo.into()),
        partition_algorithm_fst,
        radix_bits.clone(),
        grid_size,
        block_size,
        dmem_buffer_bytes,
    )?;
    radix_prnr.preallocate_partition_state::<T>(RadixPass::First)?;

    let mut inner_rel_partition_offsets = PartitionOffsets::new(
        histogram_algorithm_fst.either(|cpu| cpu.into(), |gpu| gpu.into()),
        max_chunks_1st,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        Allocator::mem_alloc_fn(offsets_mem_type.clone()),
    );

    let mut outer_rel_partition_offsets = PartitionOffsets::new(
        histogram_algorithm_fst.either(|cpu| cpu.into(), |gpu| gpu.into()),
        max_chunks_1st,
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        Allocator::mem_alloc_fn(offsets_mem_type.clone()),
    );

    inner_rel_partition_offsets.mlock()?;
    outer_rel_partition_offsets.mlock()?;

    stream.synchronize()?;

    let partitions_malloc_time = partitions_malloc_timer.elapsed();

    let prefix_sum_time = match histogram_algorithm_fst {
        DeviceType::Cpu(histogram_algorithm) => {
            let prefix_sum_timer = Instant::now();

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
                        let mut radix_prnr = CpuRadixPartitioner::new(
                            histogram_algorithm,
                            CpuRadixPartitionAlgorithm::NC,
                            radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
                            DerefMemType::AlignedSysMem {
                                align_bytes: sql_ops::CPU_CACHE_LINE_SIZE as usize,
                            },
                        );
                        radix_prnr
                            .prefix_sum(input, output)
                            .expect("Failed to run CPU prefix sum");
                    })
                }
            });

            prefix_sum_timer.elapsed().as_nanos() as f64
        }
        DeviceType::Gpu(_) => {
            let prefix_sum_start_event = Event::new(EventFlags::DEFAULT)?;
            let prefix_sum_stop_event = Event::new(EventFlags::DEFAULT)?;
            prefix_sum_start_event.record(&stream)?;

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

            prefix_sum_stop_event.record(&stream)?;
            stream.synchronize()?;
            prefix_sum_stop_event.elapsed_time_f32(&prefix_sum_start_event)? as f64
                * 10_f64.powf(6.0)
        }
    };

    let state_malloc_timer = Instant::now();

    let max_inner_partition_len =
        (0..inner_rel_partition_offsets.fanout()).try_fold(0, |max, partition_id| {
            inner_rel_partition_offsets
                .partition_len(partition_id)
                .map(|len| cmp::max(max, len))
        })?;
    let max_outer_partition_len =
        (0..outer_rel_partition_offsets.fanout()).try_fold(0, |max, partition_id| {
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

    let state_malloc_time = state_malloc_timer.elapsed();
    let partitions_malloc_timer = Instant::now();

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
    let linux_wrapper::NumaMemInfo { free, .. } = linux_wrapper::numa_mem_info(cache_node)?;
    let free = free - GPU_MEM_SLACK_BYTES;

    // Use one half of space for inner relation, and the other half for outer
    // relation
    let cached_len = free / 2 / mem::size_of::<Tuple<T, T>>();

    let (inner_rel_alloc, cached_build_tuples) =
        build_distributed_partitions_allocator(cached_len, cache_node, overflow_node, page_type)?;
    let mut inner_rel_partitions = PartitionedRelation::new(
        data.build_relation_key.len(),
        histogram_algorithm_fst.either(|cpu| cpu.into(), |gpu| gpu.into()),
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        max_chunks_1st,
        inner_rel_alloc,
        Allocator::mem_alloc_fn(offsets_mem_type.clone()),
    );

    let (outer_rel_alloc, cached_probe_tuples) =
        build_distributed_partitions_allocator(cached_len, cache_node, overflow_node, page_type)?;
    let mut outer_rel_partitions = PartitionedRelation::new(
        data.probe_relation_key.len(),
        histogram_algorithm_fst.either(|cpu| cpu.into(), |gpu| gpu.into()),
        radix_bits.pass_radix_bits(RadixPass::First).unwrap(),
        max_chunks_1st,
        outer_rel_alloc,
        Allocator::mem_alloc_fn(offsets_mem_type.clone()),
    );

    inner_rel_partitions.mlock()?;
    outer_rel_partitions.mlock()?;

    let partitions_malloc_time = partitions_malloc_time + partitions_malloc_timer.elapsed();

    let partition_start_event = Event::new(EventFlags::DEFAULT)?;
    let partition_stop_event = Event::new(EventFlags::DEFAULT)?;
    partition_start_event.record(&stream)?;

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

    partition_stop_event.record(&stream)?;

    stream.synchronize()?;
    let partition_time =
        partition_stop_event.elapsed_time_f32(&partition_start_event)? as f64 * 10_f64.powf(6.0);

    Stream::drop(stream).map_err(|(e, _)| e)?;

    let join_start_event = Event::new(EventFlags::DEFAULT)?;
    stream_states
        .iter()
        .take(1)
        .try_for_each(|StreamState { stream, .. }| join_start_event.record(stream))?;

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

    let join_stop_events = stream_states
        .iter()
        .map(|StreamState { stream, .. }| {
            let event = Event::new(EventFlags::DEFAULT)?;
            event.record(stream)?;
            Ok(event)
        })
        .collect::<Result<Vec<Event>>>()?;

    let join_time = stream_states
        .iter()
        .zip(join_stop_events.iter())
        .try_fold::<_, _, Result<_>>(0_f64, |time, (StreamState { stream, .. }, stop_event)| {
            stream.synchronize()?;
            let new_time =
                stop_event.elapsed_time_f32(&join_start_event)? as f64 * 10_f64.powf(6.0);
            Ok(time.max(new_time))
        })?;

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
        partition_ns: Some(partition_time),
        join_ns: Some(join_time),
        partitions_malloc_ns: Some(partitions_malloc_time.as_nanos() as f64),
        state_malloc_ns: Some(state_malloc_time.as_nanos() as f64),
        cached_build_tuples: *cached_build_tuples.borrow(),
        cached_probe_tuples: *cached_probe_tuples.borrow(),
    };

    Ok((sum, data_point))
}

/// Builds an allocator that caches partitions in GPU memory
///
/// The allocator uses GPU memory until `cache_max_len` is reached. Then, the
/// allocator overflows the remainder to CPU memory.
///
/// The functional programming approach used here solves the leaky abstraction
/// problem. In the execution method, we know how much cache space is available,
/// but not the required space for the partitioned relation (due to padding).
/// We embed this knowledge into a closure, instead of leaking the internal
/// length of `PartitionedRelation`.
///
/// Returns a "future" that is set to the cached length when the allocator is
/// invoked, for logging purposes.
fn build_distributed_partitions_allocator<T>(
    cache_max_len: usize,
    gpu_node: u16,
    cpu_node: u16,
    page_type: PageType,
) -> Result<(MemAllocFn<Tuple<T, T>>, Rc<RefCell<Option<usize>>>)>
where
    T: Clone + Default + DeviceCopy,
{
    // Round down to page size
    let page_size = ProcessorCache::huge_page_size()?;
    let cache_max_len = (cache_max_len / page_size) * page_size;

    let cached_len_future = Rc::new(RefCell::new(None));
    let cached_len_setter = cached_len_future.clone();

    let alloc = move |len| {
        let (cached_len, overflowed_len) = if len <= cache_max_len {
            (len, 0)
        } else {
            (cache_max_len, len - cache_max_len)
        };

        cached_len_setter.replace(Some(cached_len));

        // FIXME; make DistributedNumaMem use huge pages
        let mem_type = MemType::DistributedNumaMemWithLen {
            nodes: Box::new([
                NodeLen {
                    node: gpu_node,
                    len: cached_len,
                },
                NodeLen {
                    node: cpu_node,
                    len: overflowed_len,
                },
            ]),
            page_type,
        };

        Allocator::alloc_mem(mem_type, len)
    };

    Ok((Box::new(alloc), cached_len_future))
}