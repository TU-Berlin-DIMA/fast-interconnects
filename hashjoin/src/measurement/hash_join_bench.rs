/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use crate::operators::hash_join;
use crate::types::*;
use crate::DataGenFn;
use numa_gpu::error::{ErrorKind, Result};
use numa_gpu::runtime::allocator;
use numa_gpu::runtime::cuda::{
    CudaTransferStrategy, IntoCudaIterator, IntoCudaIteratorWithStrategy,
};
use numa_gpu::runtime::memory::*;
use numa_gpu::runtime::utils::EnsurePhysicallyBacked;
use rustacuda::event::{Event, EventFlags};
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::DeviceCopy;
use rustacuda::stream::{Stream, StreamFlags};
use std::collections::vec_deque::VecDeque;
use std::convert::TryInto;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct HashJoinBench<T: DeviceCopy> {
    pub hashing_scheme: hash_join::HashingScheme,
    pub hash_table_len: usize,
    pub build_relation_key: Mem<T>,
    pub build_relation_payload: Mem<T>,
    pub probe_relation_key: Mem<T>,
    pub probe_relation_payload: Mem<T>,
}

pub struct HashJoinBenchBuilder {
    hash_table_load_factor: usize,
    hash_table_elems_per_entry: usize,
    inner_len: usize,
    outer_len: usize,
    inner_location: u16,
    outer_location: u16,
    inner_mem_type: ArgMemType,
    outer_mem_type: ArgMemType,
    hashing_scheme: hash_join::HashingScheme,
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
}

impl Default for HashJoinBenchBuilder {
    fn default() -> HashJoinBenchBuilder {
        HashJoinBenchBuilder {
            hash_table_load_factor: 2,
            hash_table_elems_per_entry: 2, // FIXME: replace constant with an HtEntry type
            inner_len: 1,
            outer_len: 1,
            inner_location: 0,
            outer_location: 0,
            inner_mem_type: ArgMemType::System,
            outer_mem_type: ArgMemType::System,
            hashing_scheme: hash_join::HashingScheme::LinearProbing,
        }
    }
}

impl HashJoinBenchBuilder {
    pub fn hash_table_load_factor(&mut self, hash_table_load_factor: usize) -> &mut Self {
        self.hash_table_load_factor = hash_table_load_factor;
        self
    }

    pub fn inner_len(&mut self, inner_len: usize) -> &mut Self {
        self.inner_len = inner_len;
        self
    }

    pub fn outer_len(&mut self, outer_len: usize) -> &mut Self {
        self.outer_len = outer_len;
        self
    }

    pub fn inner_location(&mut self, inner_location: u16) -> &mut Self {
        self.inner_location = inner_location;
        self
    }

    pub fn outer_location(&mut self, outer_location: u16) -> &mut Self {
        self.outer_location = outer_location;
        self
    }

    pub fn inner_mem_type(&mut self, inner_mem_type: ArgMemType) -> &mut Self {
        self.inner_mem_type = inner_mem_type;
        self
    }

    pub fn outer_mem_type(&mut self, outer_mem_type: ArgMemType) -> &mut Self {
        self.outer_mem_type = outer_mem_type;
        self
    }

    pub fn hashing_scheme(&mut self, hashing_scheme: hash_join::HashingScheme) -> &mut Self {
        self.hashing_scheme = hashing_scheme;
        self
    }

    pub fn build_with_data_gen<T>(
        &mut self,
        mut data_gen_fn: DataGenFn<T>,
    ) -> Result<(HashJoinBench<T>, Duration, Duration)>
    where
        T: Copy + Default + DeviceCopy + EnsurePhysicallyBacked,
    {
        // Allocate memory for data sets
        let malloc_timer = Instant::now();
        let mut memory: VecDeque<_> = [
            (self.inner_len, self.inner_mem_type, self.inner_location),
            (self.inner_len, self.inner_mem_type, self.inner_location),
            (self.outer_len, self.outer_mem_type, self.outer_location),
            (self.outer_len, self.outer_mem_type, self.outer_location),
        ]
        .iter()
        .map(|&(len, mem_type, location)| {
            let mut mem = allocator::Allocator::alloc_deref_mem(
                ArgMemTypeHelper { mem_type, location }.into(),
                len,
            );

            // If user selected NumaLazyPinned, then pin the memory.
            // If user selected Unified, then ensure that we measure unified memory transfers from CPU memory
            match (mem_type, &mut mem) {
                (ArgMemType::NumaLazyPinned, DerefMem::NumaMem(lazy_pinned_mem)) => lazy_pinned_mem
                    .page_lock()
                    .expect("Failed to lazily pin memory"),
                // (ArgMemType::Unified, DerefMem::CudaUniMem(mem)) => {
                //     mem_advise(
                //         mem.as_unified_ptr(),
                //         mem.len(),
                //         MemAdviseFlags::CU_MEM_ADVISE_SET_READ_MOSTLY,
                //         0,
                //         MemAdviseFlags::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                //         CPU_DEVICE_ID,
                //     )?;
                // }
                _ => {}
            };

            // Force the OS to physically allocate the memory
            match mem {
                DerefMem::SysMem(ref mut mem) => T::ensure_physically_backed(mem.as_mut_slice()),
                DerefMem::NumaMem(ref mut mem) => T::ensure_physically_backed(mem.as_mut_slice()),
                DerefMem::DistributedNumaMem(ref mut mem) => {
                    T::ensure_physically_backed(mem.as_mut_slice())
                }
                DerefMem::CudaPinnedMem(ref mut mem) => {
                    T::ensure_physically_backed(mem.as_mut_slice())
                }
                DerefMem::CudaUniMem(ref mut mem) => {
                    T::ensure_physically_backed(mem.as_mut_slice())
                }
            };

            Ok(mem)
        })
        .collect::<Result<_>>()?;
        let malloc_time = malloc_timer.elapsed();

        let mut inner_key = memory.pop_front().ok_or_else(|| {
            ErrorKind::LogicError(
                "Failed to get primary key relation. Is it allocated?".to_string(),
            )
        })?;
        let inner_payload = memory.pop_front().ok_or_else(|| {
            ErrorKind::LogicError(
                "Failed to get primary key relation. Is it allocated?".to_string(),
            )
        })?;
        let mut outer_key = memory.pop_front().ok_or_else(|| {
            ErrorKind::LogicError(
                "Failed to get foreign key relation. Is it allocated?".to_string(),
            )
        })?;
        let outer_payload = memory.pop_front().ok_or_else(|| {
            ErrorKind::LogicError(
                "Failed to get foreign key relation. Is it allocated?".to_string(),
            )
        })?;

        // Generate dataset
        let gen_timer = Instant::now();
        data_gen_fn(inner_key.as_mut_slice(), outer_key.as_mut_slice())?;
        let gen_time = gen_timer.elapsed();

        // Calculate hash table length
        let hash_table_len = self
            .inner_len
            .checked_next_power_of_two()
            .and_then(|x| {
                x.checked_mul(self.hash_table_load_factor * self.hash_table_elems_per_entry)
            })
            .ok_or_else(|| {
                ErrorKind::IntegerOverflow("Failed to compute hash table length".to_string())
            })?;

        Ok((
            HashJoinBench {
                hashing_scheme: self.hashing_scheme,
                hash_table_len: hash_table_len,
                build_relation_key: inner_key.into(),
                build_relation_payload: inner_payload.into(),
                probe_relation_key: outer_key.into(),
                probe_relation_payload: outer_payload.into(),
            },
            malloc_time,
            gen_time,
        ))
    }
}

impl<T> HashJoinBench<T>
where
    T: Default
        + DeviceCopy
        + Sync
        + Send
        + hash_join::NullKey
        + hash_join::CudaHashJoinable
        + hash_join::CpuHashJoinable
        + EnsurePhysicallyBacked,
{
    pub fn cuda_hash_join(
        &mut self,
        hash_table_alloc: allocator::MemAllocFn<T>,
        build_dim: (GridSize, BlockSize),
        probe_dim: (GridSize, BlockSize),
    ) -> Result<HashJoinPoint> {
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // FIXME: specify load factor as argument
        let ht_malloc_timer = Instant::now();
        let mut hash_table_mem = hash_table_alloc(self.hash_table_len);
        if let CudaUniMem(ref mut _mem) = hash_table_mem {
            // mem_advise(
            //     mem.as_unified_ptr(),
            //     mem.len(),
            //     MemAdviseFlags::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
            //     CPU_DEVICE_ID,
            // )?;
            // prefetch_async(mem.as_unified_ptr(), mem.len(), CPU_DEVICE_ID, &stream)?;
        }
        let hash_table = hash_join::HashTable::new_on_gpu(hash_table_mem, self.hash_table_len)?;
        let ht_malloc_time = ht_malloc_timer.elapsed();

        let mut result_counts = allocator::Allocator::alloc_mem(
            allocator::MemType::CudaUniMem,
            (probe_dim.0.x * probe_dim.1.x) as usize,
        );

        // Initialize counts
        if let CudaUniMem(ref mut c) = result_counts {
            c.iter_mut().map(|count| *count = 0).for_each(drop);
        }

        stream.synchronize()?;

        let hj_op = hash_join::CudaHashJoinBuilder::<T>::default()
            .hashing_scheme(self.hashing_scheme)
            .build_dim(build_dim.0.clone(), build_dim.1.clone())
            .probe_dim(probe_dim.0.clone(), probe_dim.1.clone())
            .hash_table(hash_table)
            .build()?;

        let start_event = Event::new(EventFlags::DEFAULT)?;
        let stop_event = Event::new(EventFlags::DEFAULT)?;

        start_event.record(&stream)?;
        hj_op.build(
            self.build_relation_key.as_launchable_slice(),
            self.build_relation_payload.as_launchable_slice(),
            &stream,
        )?;

        stop_event.record(&stream)?;
        stop_event.synchronize()?;
        let build_millis = stop_event.elapsed_time_f32(&start_event)?;

        start_event.record(&stream)?;
        hj_op.probe_count(
            self.probe_relation_key.as_launchable_slice(),
            self.probe_relation_payload.as_launchable_slice(),
            &mut result_counts,
            &stream,
        )?;

        stop_event.record(&stream)?;
        stop_event.synchronize()?;
        let probe_millis = stop_event.elapsed_time_f32(&start_event)?;

        stream.synchronize()?;
        Ok(HashJoinPoint {
            build_ns: Some(build_millis as f64 * 10_f64.powf(6.0)),
            probe_ns: Some(probe_millis as f64 * 10_f64.powf(6.0)),
            hash_table_malloc_ns: Some(ht_malloc_time.as_nanos() as f64),
            ..Default::default()
        })
    }

    pub fn cuda_streaming_hash_join(
        &mut self,
        hash_table_alloc: allocator::MemAllocFn<T>,
        build_dim: (GridSize, BlockSize),
        probe_dim: (GridSize, BlockSize),
        transfer_strategy: CudaTransferStrategy,
        chunk_len: usize,
    ) -> Result<HashJoinPoint> {
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
        let hash_table = hash_join::HashTable::new_on_gpu(hash_table_mem, self.hash_table_len)?;
        let ht_malloc_time = ht_malloc_timer.elapsed();

        let mut result_counts = allocator::Allocator::alloc_mem(
            allocator::MemType::CudaUniMem,
            (probe_dim.0.x * probe_dim.1.x) as usize,
        );

        // Initialize counts
        if let CudaUniMem(ref mut c) = result_counts {
            c.iter_mut().map(|count| *count = 0).for_each(drop);
        }

        let build_rel_key: &mut [T] = (&mut self.build_relation_key)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let build_rel_pay: &mut [T] = (&mut self.build_relation_payload)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let probe_rel_key: &mut [T] = (&mut self.probe_relation_key)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let probe_rel_pay: &mut [T] = (&mut self.probe_relation_payload)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");

        let hj_op = hash_join::CudaHashJoinBuilder::<T>::default()
            .hashing_scheme(self.hashing_scheme)
            .build_dim(build_dim.0.clone(), build_dim.1.clone())
            .probe_dim(probe_dim.0.clone(), probe_dim.1.clone())
            .hash_table(hash_table)
            .build()?;

        let mut build_relation = (build_rel_key, build_rel_pay);
        let mut probe_relation = (probe_rel_key, probe_rel_pay);
        let mut build_iter =
            build_relation.into_cuda_iter_with_strategy(transfer_strategy, chunk_len)?;
        let mut probe_iter =
            probe_relation.into_cuda_iter_with_strategy(transfer_strategy, chunk_len)?;

        let build_timer = Instant::now();
        let build_mnts = build_iter.fold(|(key, val), stream| hj_op.build(key, val, stream))?;
        let build_time = build_timer.elapsed();

        let probe_timer = Instant::now();
        let probe_mnts = probe_iter
            .fold(|(key, val), stream| hj_op.probe_count(key, val, &result_counts, stream))?;
        let probe_time = probe_timer.elapsed();

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
        })
    }

    pub fn cuda_streaming_unified_hash_join(
        &mut self,
        hash_table_alloc: allocator::MemAllocFn<T>,
        build_dim: (GridSize, BlockSize),
        probe_dim: (GridSize, BlockSize),
        chunk_len: usize,
    ) -> Result<HashJoinPoint> {
        let ht_malloc_timer = Instant::now();
        let hash_table_mem = hash_table_alloc(self.hash_table_len);
        let hash_table = hash_join::HashTable::new_on_gpu(hash_table_mem, self.hash_table_len)?;
        let ht_malloc_time = ht_malloc_timer.elapsed();

        let mut result_counts = allocator::Allocator::alloc_mem(
            allocator::MemType::CudaUniMem,
            (probe_dim.0.x * probe_dim.1.x) as usize,
        );

        // Initialize counts
        if let CudaUniMem(ref mut c) = result_counts {
            c.iter_mut().map(|count| *count = 0).for_each(drop);
        }

        let build_rel_key = match self.build_relation_key {
            Mem::CudaUniMem(ref mut m) => m,
            _ => unreachable!(),
        };
        let build_rel_pay = match self.build_relation_payload {
            Mem::CudaUniMem(ref mut m) => m,
            _ => unreachable!(),
        };
        let probe_rel_key = match self.probe_relation_key {
            Mem::CudaUniMem(ref mut m) => m,
            _ => unreachable!(),
        };
        let probe_rel_pay = match self.probe_relation_payload {
            Mem::CudaUniMem(ref mut m) => m,
            _ => unreachable!(),
        };

        let hj_op = hash_join::CudaHashJoinBuilder::<T>::default()
            .hashing_scheme(self.hashing_scheme)
            .build_dim(build_dim.0.clone(), build_dim.1.clone())
            .probe_dim(probe_dim.0.clone(), probe_dim.1.clone())
            .hash_table(hash_table)
            .build()?;

        let mut build_relation = (build_rel_key, build_rel_pay);
        let mut probe_relation = (probe_rel_key, probe_rel_pay);

        let build_timer = Instant::now();
        let build_mnts = build_relation
            .into_cuda_iter(chunk_len)?
            .fold(|(key, val), stream| hj_op.build(key, val, stream))?;
        let build_time = build_timer.elapsed();

        let probe_timer = Instant::now();
        let probe_mnts = probe_relation
            .into_cuda_iter(chunk_len)?
            .fold(|(key, val), stream| hj_op.probe_count(key, val, &result_counts, stream))?;
        let probe_time = probe_timer.elapsed();

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
        })
    }

    pub fn cpu_hash_join(
        &self,
        threads: usize,
        hash_table_alloc: allocator::DerefMemAllocFn<T>,
    ) -> Result<HashJoinPoint> {
        let ht_malloc_timer = Instant::now();
        let mut hash_table_mem = hash_table_alloc(self.hash_table_len);
        T::ensure_physically_backed(hash_table_mem.as_mut_slice());
        let hash_table = hash_join::HashTable::new_on_cpu(hash_table_mem, self.hash_table_len)?;
        let ht_malloc_time = ht_malloc_timer.elapsed();

        let mut result_counts = vec![0; threads];

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|_| ErrorKind::RuntimeError("Failed to create thread pool".to_string()))?;
        let build_chunk_size = (self.build_relation_key.len() + threads - 1) / threads;
        let probe_chunk_size = (self.probe_relation_key.len() + threads - 1) / threads;

        let build_rel_key: &[T] = (&self.build_relation_key)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let build_rel_chunks: Vec<_> = build_rel_key.chunks(build_chunk_size).collect();

        let build_rel_pay: &[T] = (&self.build_relation_payload)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let build_pay_chunks: Vec<_> = build_rel_pay.chunks(build_chunk_size).collect();

        let probe_rel_key: &[T] = (&self.probe_relation_key)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let probe_rel_chunks: Vec<_> = probe_rel_key.chunks(probe_chunk_size).collect();

        let probe_rel_pay: &[T] = (&self.probe_relation_payload)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let probe_pay_chunks: Vec<_> = probe_rel_pay.chunks(probe_chunk_size).collect();

        let result_count_chunks: Vec<_> = result_counts.chunks_mut(threads).collect();

        let hj_builder = hash_join::CpuHashJoinBuilder::default()
            .hashing_scheme(self.hashing_scheme)
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
                .zip(result_count_chunks)
            {
                let mut hj_op = hj_builder.build();
                s.spawn(move |_| {
                    hj_op
                        .probe_count(rel, pay, &mut res[0])
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
}
