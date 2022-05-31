// Copyright 2019-2022 Clemens Lutz
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

use crate::error::{ErrorKind, Result};
use data_store::join_data::JoinData;
use datagen::relation::KeyAttribute;
use num_traits::cast::AsPrimitive;
use numa_gpu::runtime::allocator;
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::cuda::{
    CudaTransferStrategy, IntoCudaIterator, IntoCudaIteratorWithStrategy,
};
use numa_gpu::runtime::dispatcher::{
    HetMorselExecutorBuilder, IntoHetMorselIterator, MorselSpec, WorkerCpuAffinity,
};
use numa_gpu::runtime::memory::*;
use numa_gpu::runtime::{cuda_wrapper, linux_wrapper};
use numa_gpu::utils::CachePadded;
use rustacuda::event::{Event, EventFlags};
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::{AsyncCopyDestination, DeviceBuffer, DeviceCopy};
use rustacuda::stream::{Stream, StreamFlags};
use sql_ops::join::{no_partitioning_join, HashingScheme, HtEntry};
use std::cell::RefCell;
use std::convert::TryInto;
use std::os::raw::c_uint;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;
use std::{cmp, mem};

/// GPU memory to leave free when allocating a hybrid hash table
///
/// Getting the amount of free GPU memory and allocating the memory is
/// *not atomic*. Sometimes the allocation spuriously fails.
///
/// Instead of allocating every last byte of GPU memory, leave some slack space.
const GPU_MEM_SLACK_BYTES: usize = 32 * 1024 * 1024;

pub struct HashJoinBench<T> {
    pub hashing_scheme: HashingScheme,
    pub is_selective: bool,
    pub hash_table_len: usize,
    _phantom_data: std::marker::PhantomData<T>,
}

pub struct HashJoinBenchBuilder {
    hash_table_load_factor: usize,
    hashing_scheme: HashingScheme,
    is_selective: bool,
}

#[derive(Debug, Default)]
pub struct HashJoinPoint {
    pub hash_table_malloc_ns: Option<f64>,
    pub build_ns: Option<f64>,
    pub probe_ns: Option<f64>,
    pub build_warm_up_ns: Option<f64>,
    pub probe_warm_up_ns: Option<f64>,
    pub build_copy_ns: Option<f64>,
    pub probe_copy_ns: Option<f64>,
    pub build_compute_ns: Option<f64>,
    pub probe_compute_ns: Option<f64>,
    pub build_cool_down_ns: Option<f64>,
    pub probe_cool_down_ns: Option<f64>,
    pub cached_hash_table_tuples: Option<usize>,
}

impl Default for HashJoinBenchBuilder {
    fn default() -> HashJoinBenchBuilder {
        HashJoinBenchBuilder {
            hash_table_load_factor: 2,
            hashing_scheme: HashingScheme::LinearProbing,
            is_selective: false,
        }
    }
}

impl HashJoinBenchBuilder {
    pub fn hash_table_load_factor(&mut self, hash_table_load_factor: usize) -> &mut Self {
        self.hash_table_load_factor = hash_table_load_factor;
        self
    }

    pub fn hashing_scheme(&mut self, hashing_scheme: HashingScheme) -> &mut Self {
        self.hashing_scheme = hashing_scheme;
        self
    }

    pub fn is_selective(&mut self, is_selective: bool) -> &mut Self {
        self.is_selective = is_selective;
        self
    }

    fn get_hash_table_len(&self, inner_relation_len: usize) -> Result<usize> {
        let hash_table_len = match self.hashing_scheme {
            HashingScheme::LinearProbing => inner_relation_len
                .checked_next_power_of_two()
                .and_then(|x| x.checked_mul(self.hash_table_load_factor))
                .ok_or_else(|| {
                    ErrorKind::IntegerOverflow("Failed to compute hash table length".to_string())
                })?,
            HashingScheme::Perfect => inner_relation_len,
            HashingScheme::BucketChaining => unimplemented!(),
        };

        Ok(hash_table_len)
    }

    pub fn build<T>(&mut self, inner_relation_len: usize) -> Result<HashJoinBench<T>> {
        Ok(HashJoinBench {
            hashing_scheme: self.hashing_scheme,
            is_selective: self.is_selective,
            hash_table_len: self.get_hash_table_len(inner_relation_len)?,
            _phantom_data: std::marker::PhantomData::<T>,
        })
    }
}

impl<T> HashJoinBench<T>
where
    T: Default
        + AsPrimitive<c_uint>
        + DeviceCopy
        + Sync
        + Send
        + KeyAttribute
        + no_partitioning_join::CudaHashJoinable
        + no_partitioning_join::CpuHashJoinable,
{
    pub fn cuda_hash_join(
        &self,
        data: &mut JoinData<T>,
        hash_table_alloc: allocator::MemSpillAllocFn<HtEntry<T, T>>,
        cache_node: u16,
        max_hash_table_cache_bytes: Option<usize>,
        cached_hash_table_tuples: Rc<RefCell<Option<usize>>>,
        build_dim: (GridSize, BlockSize),
        probe_dim: (GridSize, BlockSize),
    ) -> Result<HashJoinPoint> {
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // FIXME: specify load factor as argument
        let ht_malloc_timer = Instant::now();

        // Builder loads the CUDA module. The module requires some GPU memory.
        // Thus, module needs to be loaded before calculating the available GPU
        // memory.
        let hj_op_builder = no_partitioning_join::CudaHashJoinBuilder::<T>::default();

        let linux_wrapper::NumaMemInfo { free, .. } = linux_wrapper::numa_mem_info(cache_node)?;
        let free = free - GPU_MEM_SLACK_BYTES;
        let cache_bytes = max_hash_table_cache_bytes.map_or(free, |bytes| {
            if bytes > free {
                eprintln!(
                    "Warning: Hash table cache size too large, reducing to maximum available memory"
                    );
            }
            cmp::min(bytes, free)
        });
        let cache_max_len = cache_bytes / mem::size_of::<HtEntry<T, T>>();
        let ht_alloc = hash_table_alloc(cache_max_len);

        let mut hash_table_mem = ht_alloc(self.hash_table_len);
        if let CudaUniMem(ref mut _mem) = hash_table_mem {
            // mem_advise(
            //     mem.as_unified_ptr(),
            //     mem.len(),
            //     MemAdviseFlags::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
            //     CPU_DEVICE_ID,
            // )?;
            // prefetch_async(mem.as_unified_ptr(), mem.len(), CPU_DEVICE_ID, &stream)?;
        }
        let mut hash_table =
            no_partitioning_join::HashTable::new_on_gpu(hash_table_mem, self.hash_table_len)?;
        hash_table.mlock()?;
        let hash_table = hash_table;
        let ht_malloc_time = ht_malloc_timer.elapsed();

        let mut result_sums = {
            let mut mem =
                unsafe { DeviceBuffer::uninitialized((probe_dim.0.x * probe_dim.1.x) as usize)? };
            cuda_wrapper::memset_async(mem.as_launchable_mut_slice(), 0, &stream)?;
            Mem::CudaDevMem(mem)
        };

        stream.synchronize()?;

        let hj_op = hj_op_builder
            .hashing_scheme(self.hashing_scheme)
            .is_selective(self.is_selective)
            .build_dim(build_dim.0.clone(), build_dim.1.clone())
            .probe_dim(probe_dim.0.clone(), probe_dim.1.clone())
            .hash_table(Arc::new(hash_table))
            .build()?;

        let start_event = Event::new(EventFlags::DEFAULT)?;
        let stop_event = Event::new(EventFlags::DEFAULT)?;

        start_event.record(&stream)?;
        hj_op.build(
            data.build_relation_key.as_launchable_slice(),
            data.build_relation_payload.as_launchable_slice(),
            &stream,
        )?;

        stop_event.record(&stream)?;
        stop_event.synchronize()?;
        let build_millis = stop_event.elapsed_time_f32(&start_event)?;

        start_event.record(&stream)?;
        hj_op.probe_sum(
            data.probe_relation_key.as_launchable_slice(),
            data.probe_relation_payload.as_launchable_slice(),
            &mut result_sums,
            &stream,
        )?;

        stop_event.record(&stream)?;
        stop_event.synchronize()?;
        let probe_millis = stop_event.elapsed_time_f32(&start_event)?;

        let mut result_sums_host = vec![0; result_sums.len()];
        if let Mem::CudaDevMem(results) = result_sums {
            unsafe { results.async_copy_to(&mut result_sums_host, &stream)? };
        }

        stream.synchronize()?;
        let _sum: u64 = result_sums_host.iter().sum();

        Ok(HashJoinPoint {
            build_ns: Some(build_millis as f64 * 10_f64.powf(6.0)),
            probe_ns: Some(probe_millis as f64 * 10_f64.powf(6.0)),
            hash_table_malloc_ns: Some(ht_malloc_time.as_nanos() as f64),
            cached_hash_table_tuples: *cached_hash_table_tuples.borrow(),
            ..Default::default()
        })
    }

    pub fn cuda_streaming_hash_join(
        &self,
        data: &mut JoinData<T>,
        hash_table_alloc: allocator::MemAllocFn<HtEntry<T, T>>,
        build_dim: (GridSize, BlockSize),
        probe_dim: (GridSize, BlockSize),
        transfer_strategy: CudaTransferStrategy,
        gpu_morsel_bytes: usize,
        cpu_memcpy_threads: usize,
        cpu_affinity: &CpuAffinity,
    ) -> Result<HashJoinPoint> {
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let ht_malloc_timer = Instant::now();
        let mut hash_table_mem = hash_table_alloc(self.hash_table_len);
        if let CudaUniMem(ref mut _mem) = hash_table_mem {
            // mem_advise(
            //     mem.as_unified_ptr(),
            //     mem.len(),
            //     MemAdviseFlags::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
            //     CPU_DEVICE_ID,
            // )?;
            //
            // let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
            // prefetch_async(mem.as_unified_ptr(), mem.len(), CPU_DEVICE_ID, &stream)?;
            // stream.synchronize()?;
        }
        let mut hash_table =
            no_partitioning_join::HashTable::new_on_gpu(hash_table_mem, self.hash_table_len)?;
        hash_table.mlock()?;
        let hash_table = hash_table;
        let ht_malloc_time = ht_malloc_timer.elapsed();

        let result_sums = {
            let mut mem =
                unsafe { DeviceBuffer::uninitialized((probe_dim.0.x * probe_dim.1.x) as usize)? };
            cuda_wrapper::memset_async(mem.as_launchable_mut_slice(), 0, &stream)?;
            Mem::CudaDevMem(mem)
        };

        stream.synchronize()?;

        let build_rel_key: &mut [T] = (&mut data.build_relation_key)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let build_rel_pay: &mut [T] = (&mut data.build_relation_payload)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let probe_rel_key: &mut [T] = (&mut data.probe_relation_key)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let probe_rel_pay: &mut [T] = (&mut data.probe_relation_payload)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");

        let hj_op = no_partitioning_join::CudaHashJoinBuilder::<T>::default()
            .hashing_scheme(self.hashing_scheme)
            .is_selective(self.is_selective)
            .build_dim(build_dim.0.clone(), build_dim.1.clone())
            .probe_dim(probe_dim.0.clone(), probe_dim.1.clone())
            .hash_table(Arc::new(hash_table))
            .build()?;

        let mut build_relation = (build_rel_key, build_rel_pay);
        let mut probe_relation = (probe_rel_key, probe_rel_pay);
        let mut build_iter = build_relation.into_cuda_iter_with_strategy(
            transfer_strategy,
            gpu_morsel_bytes,
            cpu_memcpy_threads,
            cpu_affinity,
        )?;
        let mut probe_iter = probe_relation.into_cuda_iter_with_strategy(
            transfer_strategy,
            gpu_morsel_bytes,
            cpu_memcpy_threads,
            cpu_affinity,
        )?;

        let build_timer = Instant::now();
        let build_mnts = build_iter.fold(|(key, val), stream| {
            hj_op
                .build(key, val, stream)
                .expect("Failed to run hash join build");
            Ok(())
        })?;
        let build_time = build_timer.elapsed();

        let probe_timer = Instant::now();
        let probe_mnts = probe_iter.fold(|(key, val), stream| {
            hj_op
                .probe_sum(key, val, &result_sums, stream)
                .expect("Failed to run hash join probe");
            Ok(())
        })?;
        let probe_time = probe_timer.elapsed();

        let mut result_sums_host = vec![0; result_sums.len()];
        if let Mem::CudaDevMem(results) = result_sums {
            unsafe { results.async_copy_to(&mut result_sums_host, &stream)? };
        }

        stream.synchronize()?;
        let _sum: u64 = result_sums_host.iter().sum();

        Ok(HashJoinPoint {
            build_ns: Some(build_time.as_nanos() as f64),
            probe_ns: Some(probe_time.as_nanos() as f64),
            hash_table_malloc_ns: Some(ht_malloc_time.as_nanos() as f64),
            build_warm_up_ns: build_mnts.warm_up_ns,
            probe_warm_up_ns: probe_mnts.warm_up_ns,
            build_copy_ns: build_mnts.copy_ns,
            probe_copy_ns: probe_mnts.copy_ns,
            build_compute_ns: build_mnts.compute_ns,
            probe_compute_ns: probe_mnts.compute_ns,
            build_cool_down_ns: build_mnts.cool_down_ns,
            probe_cool_down_ns: probe_mnts.cool_down_ns,
            cached_hash_table_tuples: None,
        })
    }

    pub fn cuda_streaming_unified_hash_join(
        &self,
        data: &mut JoinData<T>,
        hash_table_alloc: allocator::MemAllocFn<HtEntry<T, T>>,
        build_dim: (GridSize, BlockSize),
        probe_dim: (GridSize, BlockSize),
        gpu_morsel_bytes: usize,
    ) -> Result<HashJoinPoint> {
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let ht_malloc_timer = Instant::now();
        let hash_table_mem = hash_table_alloc(self.hash_table_len);
        let mut hash_table =
            no_partitioning_join::HashTable::new_on_gpu(hash_table_mem, self.hash_table_len)?;
        hash_table.mlock()?;
        let hash_table = hash_table;
        let ht_malloc_time = ht_malloc_timer.elapsed();

        let result_sums = {
            let mut mem =
                unsafe { DeviceBuffer::uninitialized((probe_dim.0.x * probe_dim.1.x) as usize)? };
            cuda_wrapper::memset_async(mem.as_launchable_mut_slice(), 0, &stream)?;
            Mem::CudaDevMem(mem)
        };

        stream.synchronize()?;

        let build_rel_key = match data.build_relation_key {
            Mem::CudaUniMem(ref mut m) => m,
            _ => unreachable!(),
        };
        let build_rel_pay = match data.build_relation_payload {
            Mem::CudaUniMem(ref mut m) => m,
            _ => unreachable!(),
        };
        let probe_rel_key = match data.probe_relation_key {
            Mem::CudaUniMem(ref mut m) => m,
            _ => unreachable!(),
        };
        let probe_rel_pay = match data.probe_relation_payload {
            Mem::CudaUniMem(ref mut m) => m,
            _ => unreachable!(),
        };

        let hj_op = no_partitioning_join::CudaHashJoinBuilder::<T>::default()
            .hashing_scheme(self.hashing_scheme)
            .is_selective(self.is_selective)
            .build_dim(build_dim.0.clone(), build_dim.1.clone())
            .probe_dim(probe_dim.0.clone(), probe_dim.1.clone())
            .hash_table(Arc::new(hash_table))
            .build()?;

        let mut build_relation = (build_rel_key, build_rel_pay);
        let mut probe_relation = (probe_rel_key, probe_rel_pay);

        let build_timer = Instant::now();
        let build_mnts =
            build_relation
                .into_cuda_iter(gpu_morsel_bytes)?
                .fold(|(key, val), stream| {
                    hj_op
                        .build(key, val, stream)
                        .expect("Failed to run hash join build");
                    Ok(())
                })?;
        let build_time = build_timer.elapsed();

        let probe_timer = Instant::now();
        let probe_mnts =
            probe_relation
                .into_cuda_iter(gpu_morsel_bytes)?
                .fold(|(key, val), stream| {
                    hj_op
                        .probe_sum(key, val, &result_sums, stream)
                        .expect("Failed to run hash join probe");
                    Ok(())
                })?;
        let probe_time = probe_timer.elapsed();

        let mut result_sums_host = vec![0; result_sums.len()];
        if let Mem::CudaDevMem(results) = result_sums {
            unsafe { results.async_copy_to(&mut result_sums_host, &stream)? };
        }

        stream.synchronize()?;
        let _sum: u64 = result_sums_host.iter().sum();

        Ok(HashJoinPoint {
            build_ns: Some(build_time.as_nanos() as f64),
            probe_ns: Some(probe_time.as_nanos() as f64),
            hash_table_malloc_ns: Some(ht_malloc_time.as_nanos() as f64),
            build_warm_up_ns: build_mnts.warm_up_ns,
            probe_warm_up_ns: probe_mnts.warm_up_ns,
            build_copy_ns: build_mnts.copy_ns,
            probe_copy_ns: probe_mnts.copy_ns,
            build_compute_ns: build_mnts.compute_ns,
            probe_compute_ns: probe_mnts.compute_ns,
            build_cool_down_ns: build_mnts.cool_down_ns,
            probe_cool_down_ns: probe_mnts.cool_down_ns,
            cached_hash_table_tuples: None,
        })
    }

    pub fn cpu_hash_join(
        &self,
        data: &mut JoinData<T>,
        threads: usize,
        cpu_affinity: &CpuAffinity,
        hash_table_alloc: allocator::DerefMemAllocFn<HtEntry<T, T>>,
    ) -> Result<HashJoinPoint> {
        let ht_malloc_timer = Instant::now();
        let hash_table_mem = hash_table_alloc(self.hash_table_len);
        let mut hash_table =
            no_partitioning_join::HashTable::new_on_cpu(hash_table_mem, self.hash_table_len)?;
        hash_table.mlock()?;
        let hash_table = hash_table;
        let ht_malloc_time = ht_malloc_timer.elapsed();

        let mut result_sums = vec![CachePadded { value: 0 }; threads];

        let boxed_cpu_affinity = Arc::new(cpu_affinity.clone());
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .start_handler(move |tid| {
                boxed_cpu_affinity
                    .clone()
                    .set_affinity(tid as u16)
                    .expect("Couldn't set CPU core affinity");
                likwid::thread_init();
            })
            .build()
            .map_err(|_| ErrorKind::RuntimeError("Failed to create thread pool".to_string()))?;
        let build_chunk_size = (data.build_relation_key.len() + threads - 1) / threads;
        let probe_chunk_size = (data.probe_relation_key.len() + threads - 1) / threads;

        let build_rel_key: &[T] = (&data.build_relation_key)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let build_rel_chunks: Vec<_> = build_rel_key.chunks(build_chunk_size).collect();

        let build_rel_pay: &[T] = (&data.build_relation_payload)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let build_pay_chunks: Vec<_> = build_rel_pay.chunks(build_chunk_size).collect();

        let probe_rel_key: &[T] = (&data.probe_relation_key)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let probe_rel_chunks: Vec<_> = probe_rel_key.chunks(probe_chunk_size).collect();

        let probe_rel_pay: &[T] = (&data.probe_relation_payload)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let probe_pay_chunks: Vec<_> = probe_rel_pay.chunks(probe_chunk_size).collect();

        let hj_builder = no_partitioning_join::CpuHashJoinBuilder::default()
            .hashing_scheme(self.hashing_scheme)
            .is_selective(self.is_selective)
            .hash_table(Arc::new(hash_table));

        let build_timer = Instant::now();
        thread_pool.scope(|s| {
            for ((_tid, rel), pay) in (0..threads).zip(build_rel_chunks).zip(build_pay_chunks) {
                let mut hj_op = hj_builder.build();
                s.spawn(move |_| {
                    hj_op.build(rel, pay).expect("Couldn't build hash table");
                });
            }
        });
        let build_time = build_timer.elapsed();

        let probe_timer = Instant::now();
        thread_pool.scope(|s| {
            for (((_tid, rel), pay), res) in (0..threads)
                .zip(probe_rel_chunks)
                .zip(probe_pay_chunks)
                .zip(result_sums.iter_mut())
            {
                let mut hj_op = hj_builder.build();
                s.spawn(move |_| {
                    hj_op
                        .probe_sum(rel, pay, &mut res.value)
                        .expect("Couldn't execute hash table probe");
                });
            }
        });
        let probe_time = probe_timer.elapsed();

        Ok(HashJoinPoint {
            build_ns: Some(build_time.as_nanos() as f64),
            probe_ns: Some(probe_time.as_nanos() as f64),
            hash_table_malloc_ns: Some(ht_malloc_time.as_nanos() as f64),
            ..Default::default()
        })
    }

    pub fn hetrogeneous_hash_join(
        &self,
        data: &mut JoinData<T>,
        hash_table_alloc: allocator::MemAllocFn<HtEntry<T, T>>,
        cpu_threads: usize,
        worker_cpu_affinity: &WorkerCpuAffinity,
        gpu_ids: Vec<u16>,
        build_dim: (GridSize, BlockSize),
        probe_dim: (GridSize, BlockSize),
        morsel_spec: &MorselSpec,
    ) -> Result<HashJoinPoint> {
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // FIXME: specify load factor as argument
        let ht_malloc_timer = Instant::now();
        let hash_table_mem = hash_table_alloc(self.hash_table_len);
        let mut hash_table =
            no_partitioning_join::HashTable::new_on_gpu(hash_table_mem, self.hash_table_len)?;
        hash_table.mlock()?;
        let hash_table = Arc::new(hash_table);
        let ht_malloc_time = ht_malloc_timer.elapsed();

        let result_sums = {
            let mut mem =
                unsafe { DeviceBuffer::uninitialized((probe_dim.0.x * probe_dim.1.x) as usize)? };
            cuda_wrapper::memset_async(mem.as_launchable_mut_slice(), 0, &stream)?;
            Mem::CudaDevMem(mem)
        };

        stream.synchronize()?;

        // Convert Mem<T> into &mut [T]
        let build_rel_key: &mut [T] = (&mut data.build_relation_key)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let build_rel_pay: &mut [T] = (&mut data.build_relation_payload)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let probe_rel_key: &mut [T] = (&mut data.probe_relation_key)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let probe_rel_pay: &mut [T] = (&mut data.probe_relation_payload)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");

        let cpu_hj_builder = no_partitioning_join::CpuHashJoinBuilder::default()
            .hashing_scheme(self.hashing_scheme)
            .is_selective(self.is_selective)
            .hash_table(hash_table.clone());

        let gpu_hj_builder = no_partitioning_join::CudaHashJoinBuilder::<T>::default()
            .hashing_scheme(self.hashing_scheme)
            .is_selective(self.is_selective)
            .build_dim(build_dim.0.clone(), build_dim.1.clone())
            .probe_dim(probe_dim.0.clone(), probe_dim.1.clone())
            .hash_table(hash_table.clone());

        let mut executor = HetMorselExecutorBuilder::new()
            .cpu_threads(cpu_threads)
            .worker_cpu_affinity(worker_cpu_affinity.clone())
            .gpu_ids(gpu_ids)
            .morsel_spec(morsel_spec.clone())
            .build()?;

        let build_timer = Instant::now();
        (build_rel_key, build_rel_pay)
            .into_het_morsel_iter(&mut executor)
            .fold(
                |(rel, pay)| {
                    let mut hj_op = cpu_hj_builder.build();
                    hj_op.build(rel, pay).expect("Couldn't build hash table");
                    Ok(())
                },
                |(rel, pay), stream| {
                    let hj_op = gpu_hj_builder
                        .build()
                        .expect("Failed to build GPU hash join");
                    hj_op
                        .build(rel, pay, stream)
                        .expect("Failed to run GPU hash join build");
                    Ok(())
                },
            )?;
        let build_time = build_timer.elapsed();

        let probe_timer = Instant::now();
        (probe_rel_key, probe_rel_pay)
            .into_het_morsel_iter(&mut executor)
            .fold(
                |(rel, pay)| {
                    let mut hj_op = cpu_hj_builder.build();

                    // FIXME: retrieve sums of all threads, e.g., by implementing fold instead of map
                    let mut result_sum = CachePadded { value: 0 };
                    hj_op
                        .probe_sum(rel, pay, &mut result_sum.value)
                        .expect("Failed to run CPU hash join probe");

                    Ok(())
                },
                |(rel, pay), stream| {
                    let hj_op = gpu_hj_builder
                        .build()
                        .expect("Failed to build GPU hash join");

                    hj_op
                        .probe_sum(rel, pay, &result_sums, stream)
                        .expect("Failed to run GPU hash join probe");

                    Ok(())
                },
            )?;

        let probe_time = probe_timer.elapsed();

        let mut result_sums_host = vec![0; result_sums.len()];
        if let Mem::CudaDevMem(results) = result_sums {
            unsafe { results.async_copy_to(&mut result_sums_host, &stream)? };
        }

        stream.synchronize()?;
        let _sum: u64 = result_sums_host.iter().sum();

        Ok(HashJoinPoint {
            build_ns: Some(build_time.as_nanos() as f64),
            probe_ns: Some(probe_time.as_nanos() as f64),
            hash_table_malloc_ns: Some(ht_malloc_time.as_nanos() as f64),
            ..Default::default()
        })
    }

    #[allow(dead_code)]
    pub fn gpu_build_heterogeneous_probe(
        &self,
        data: &mut JoinData<T>,
        cpu_hash_table_alloc: allocator::MemAllocFn<HtEntry<T, T>>,
        gpu_hash_table_alloc: allocator::MemAllocFn<HtEntry<T, T>>,
        cpu_threads: usize,
        worker_cpu_affinity: &WorkerCpuAffinity,
        gpu_ids: Vec<u16>,
        build_dim: (GridSize, BlockSize),
        probe_dim: (GridSize, BlockSize),
        morsel_spec: &MorselSpec,
    ) -> Result<HashJoinPoint> {
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let ht_malloc_timer = Instant::now();

        let gpu_hash_table_mem = gpu_hash_table_alloc(self.hash_table_len);
        let mut gpu_hash_table =
            no_partitioning_join::HashTable::new_on_gpu(gpu_hash_table_mem, self.hash_table_len)?;
        gpu_hash_table.mlock()?;
        let gpu_hash_table = Arc::new(gpu_hash_table);

        let mut cpu_hash_table_mem = cpu_hash_table_alloc(self.hash_table_len);
        cpu_hash_table_mem.mlock()?;
        let cpu_hash_table_mem = cpu_hash_table_mem;

        let ht_malloc_time = ht_malloc_timer.elapsed();

        let result_sums = {
            let mut mem =
                unsafe { DeviceBuffer::uninitialized((probe_dim.0.x * probe_dim.1.x) as usize)? };
            cuda_wrapper::memset_async(mem.as_launchable_mut_slice(), 0, &stream)?;
            Mem::CudaDevMem(mem)
        };

        stream.synchronize()?;

        // Convert Mem<T> into &mut [T]
        let probe_rel_key: &mut [T] = (&mut data.probe_relation_key)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let probe_rel_pay: &mut [T] = (&mut data.probe_relation_payload)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");

        let gpu_hj_builder = no_partitioning_join::CudaHashJoinBuilder::<T>::default()
            .hashing_scheme(self.hashing_scheme)
            .is_selective(self.is_selective)
            .build_dim(build_dim.0.clone(), build_dim.1.clone())
            .probe_dim(probe_dim.0.clone(), probe_dim.1.clone())
            .hash_table(gpu_hash_table.clone());

        let mut executor = HetMorselExecutorBuilder::new()
            .cpu_threads(cpu_threads)
            .worker_cpu_affinity(worker_cpu_affinity.clone())
            .gpu_ids(gpu_ids)
            .morsel_spec(morsel_spec.clone())
            .build()?;

        let build_timer = Instant::now();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let gpu_hj_op = gpu_hj_builder.build()?;
        gpu_hj_op.build(
            data.build_relation_key.as_launchable_slice(),
            data.build_relation_payload.as_launchable_slice(),
            &stream,
        )?;
        stream.synchronize()?;

        let cpu_hash_table = Arc::new(no_partitioning_join::HashTable::new_from_hash_table(
            cpu_hash_table_mem,
            &gpu_hash_table,
        )?);
        let build_time = build_timer.elapsed();

        let probe_timer = Instant::now();
        let cpu_hj_builder = no_partitioning_join::CpuHashJoinBuilder::default()
            .hashing_scheme(self.hashing_scheme)
            .is_selective(self.is_selective)
            .hash_table(cpu_hash_table.clone());
        (probe_rel_key, probe_rel_pay)
            .into_het_morsel_iter(&mut executor)
            .fold(
                |(rel, pay)| {
                    let mut hj_op = cpu_hj_builder.build();

                    // FIXME: retrieve sums of all threads, e.g., by implementing fold instead of map
                    let mut result_sum = CachePadded { value: 0 };
                    hj_op
                        .probe_sum(rel, pay, &mut result_sum.value)
                        .expect("Failed to run CPU hash join probe");

                    Ok(())
                },
                |(rel, pay), stream| {
                    let hj_op = gpu_hj_builder
                        .build()
                        .expect("Failed to run GPU hash join build");

                    hj_op
                        .probe_sum(rel, pay, &result_sums, stream)
                        .expect("Failed to run GPU hash join probe");

                    Ok(())
                },
            )?;
        let probe_time = probe_timer.elapsed();

        let mut result_sums_host = vec![0; result_sums.len()];
        if let Mem::CudaDevMem(results) = result_sums {
            unsafe { results.async_copy_to(&mut result_sums_host, &stream)? };
        }

        stream.synchronize()?;
        let _sum: u64 = result_sums_host.iter().sum();

        Ok(HashJoinPoint {
            build_ns: Some(build_time.as_nanos() as f64),
            probe_ns: Some(probe_time.as_nanos() as f64),
            hash_table_malloc_ns: Some(ht_malloc_time.as_nanos() as f64),
            ..Default::default()
        })
    }
}
