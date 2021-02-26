/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2020 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use crate::error::{ErrorKind, Result};
use crate::types::*;
use crate::DataGenFn;
use csv::{ByteRecord, ReaderBuilder};
use flate2::read::GzDecoder;
use num_rational::Ratio;
use numa_gpu::runtime::allocator;
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::cuda::{
    CudaTransferStrategy, IntoCudaIterator, IntoCudaIteratorWithStrategy,
};
use numa_gpu::runtime::dispatcher::{
    HetMorselExecutorBuilder, IntoHetMorselIterator, MorselSpec, WorkerCpuAffinity,
};
use numa_gpu::runtime::memory::*;
use numa_gpu::runtime::numa::NodeRatio;
use numa_gpu::runtime::utils::EnsurePhysicallyBacked;
use numa_gpu::utils::CachePadded;
use rustacuda::event::{Event, EventFlags};
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::DeviceCopy;
use rustacuda::stream::{Stream, StreamFlags};
use serde::de::DeserializeOwned;
use sql_ops::join::{no_partitioning_join, HashingScheme};
use std::collections::vec_deque::VecDeque;
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct HashJoinBench<T: DeviceCopy> {
    pub hashing_scheme: HashingScheme,
    pub is_selective: bool,
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
    inner_location: Box<[NodeRatio]>,
    outer_location: Box<[NodeRatio]>,
    inner_mem_type: ArgMemType,
    outer_mem_type: ArgMemType,
    huge_pages: Option<bool>,
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
}

impl Default for HashJoinBenchBuilder {
    fn default() -> HashJoinBenchBuilder {
        HashJoinBenchBuilder {
            hash_table_load_factor: 2,
            hash_table_elems_per_entry: 2, // FIXME: replace constant with an HtEntry type
            inner_len: 1,
            outer_len: 1,
            inner_location: Box::new([NodeRatio {
                node: 0,
                ratio: Ratio::from_integer(1),
            }]),
            outer_location: Box::new([NodeRatio {
                node: 0,
                ratio: Ratio::from_integer(1),
            }]),
            inner_mem_type: ArgMemType::System,
            outer_mem_type: ArgMemType::System,
            huge_pages: None,
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

    pub fn inner_len(&mut self, inner_len: usize) -> &mut Self {
        self.inner_len = inner_len;
        self
    }

    pub fn outer_len(&mut self, outer_len: usize) -> &mut Self {
        self.outer_len = outer_len;
        self
    }

    pub fn inner_location(&mut self, inner_location: Box<[NodeRatio]>) -> &mut Self {
        self.inner_location = inner_location;
        self
    }

    pub fn outer_location(&mut self, outer_location: Box<[NodeRatio]>) -> &mut Self {
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

    pub fn huge_pages(&mut self, huge_pages: Option<bool>) -> &mut Self {
        self.huge_pages = huge_pages;
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

    fn allocate_relations<T>(
        &self,
    ) -> Result<(DerefMem<T>, DerefMem<T>, DerefMem<T>, DerefMem<T>, Duration)>
    where
        T: Clone + Default + DeviceCopy,
    {
        // Allocate memory for data sets
        let malloc_timer = Instant::now();
        let mut memory: VecDeque<_> = [
            (self.inner_len, self.inner_mem_type, &self.inner_location),
            (self.inner_len, self.inner_mem_type, &self.inner_location),
            (self.outer_len, self.outer_mem_type, &self.outer_location),
            (self.outer_len, self.outer_mem_type, &self.outer_location),
        ]
        .iter()
        .map(|&(len, mem_type, node_ratios)| {
            let mut mem = allocator::Allocator::alloc_deref_mem(
                ArgMemTypeHelper {
                    mem_type,
                    node_ratios: node_ratios.clone(),
                    huge_pages: self.huge_pages,
                }
                .into(),
                len,
            );

            // Force the OS to physically allocate the memory
            mem.ensure_physically_backed();

            Ok(mem)
        })
        .collect::<Result<_>>()?;
        let malloc_time = malloc_timer.elapsed();

        let inner_key = memory.pop_front().ok_or_else(|| {
            ErrorKind::LogicError(
                "Failed to get primary key relation. Is it allocated?".to_string(),
            )
        })?;
        let inner_payload = memory.pop_front().ok_or_else(|| {
            ErrorKind::LogicError(
                "Failed to get primary key relation. Is it allocated?".to_string(),
            )
        })?;
        let outer_key = memory.pop_front().ok_or_else(|| {
            ErrorKind::LogicError(
                "Failed to get foreign key relation. Is it allocated?".to_string(),
            )
        })?;
        let outer_payload = memory.pop_front().ok_or_else(|| {
            ErrorKind::LogicError(
                "Failed to get foreign key relation. Is it allocated?".to_string(),
            )
        })?;

        Ok((
            inner_key,
            inner_payload,
            outer_key,
            outer_payload,
            malloc_time,
        ))
    }

    fn get_hash_table_len(&self) -> Result<usize> {
        let hash_table_len = match self.hashing_scheme {
            HashingScheme::LinearProbing => self
                .inner_len
                .checked_next_power_of_two()
                .and_then(|x| {
                    x.checked_mul(self.hash_table_load_factor * self.hash_table_elems_per_entry)
                })
                .ok_or_else(|| {
                    ErrorKind::IntegerOverflow("Failed to compute hash table length".to_string())
                })?,
            HashingScheme::Perfect => self
                .inner_len
                .checked_mul(self.hash_table_elems_per_entry)
                .ok_or_else(|| {
                    ErrorKind::IntegerOverflow("Failed to compute hash table length".to_string())
                })?,
            HashingScheme::BucketChaining => unimplemented!(),
        };

        Ok(hash_table_len)
    }

    pub fn build_with_data_gen<T>(
        &mut self,
        mut data_gen_fn: DataGenFn<T>,
    ) -> Result<(HashJoinBench<T>, Duration, Duration)>
    where
        T: Copy + Default + DeviceCopy,
    {
        let (mut inner_key, inner_payload, mut outer_key, outer_payload, malloc_time) =
            self.allocate_relations()?;

        // Generate dataset
        let gen_timer = Instant::now();
        data_gen_fn(inner_key.as_mut_slice(), outer_key.as_mut_slice())?;
        let gen_time = gen_timer.elapsed();

        Ok((
            HashJoinBench {
                hashing_scheme: self.hashing_scheme,
                is_selective: self.is_selective,
                hash_table_len: self.get_hash_table_len()?,
                build_relation_key: inner_key.into(),
                build_relation_payload: inner_payload.into(),
                probe_relation_key: outer_key.into(),
                probe_relation_payload: outer_payload.into(),
            },
            malloc_time,
            gen_time,
        ))
    }

    pub fn build_with_files<T: DeserializeOwned>(
        &mut self,
        inner_relation_path: &str,
        outer_relation_path: &str,
    ) -> Result<(HashJoinBench<T>, Duration, Duration)>
    where
        T: Copy + Default + DeviceCopy,
    {
        let mut reader_spec = ReaderBuilder::new();
        reader_spec
            .delimiter(b' ')
            .has_headers(true)
            .quoting(false)
            .double_quote(false);

        let mut readers = [&inner_relation_path, &outer_relation_path]
            .iter()
            .map(|path| {
                let file = File::open(path)?;
                let reader: Box<dyn Read> = if path.ends_with("gz") {
                    Box::new(GzDecoder::new(file))
                } else {
                    Box::new(file)
                };
                Ok(reader_spec.from_reader(reader))
            })
            .collect::<Result<VecDeque<_>>>()?;

        let mut inner_reader = readers.pop_front().unwrap();
        let mut outer_reader = readers.pop_front().unwrap();
        let mut record = ByteRecord::new();

        let io_timer = Instant::now();

        // Count the number of tuples
        let mut inner_len = 0;
        while inner_reader.read_byte_record(&mut record)? {
            inner_len += 1;
        }
        self.inner_len = inner_len;

        let mut outer_len = 0;
        while outer_reader.read_byte_record(&mut record)? {
            outer_len += 1;
        }
        self.outer_len = outer_len;

        let io_count_time = io_timer.elapsed();

        let (mut inner_key, mut inner_payload, mut outer_key, mut outer_payload, malloc_time) =
            self.allocate_relations()?;

        let io_timer = Instant::now();

        // Read in the tuples
        let mut readers = [&inner_relation_path, &outer_relation_path]
            .iter()
            .map(|path| {
                let file = File::open(path)?;
                let reader: Box<dyn Read> = if path.ends_with("gz") {
                    Box::new(GzDecoder::new(file))
                } else {
                    Box::new(file)
                };
                Ok(reader_spec.from_reader(reader))
            })
            .collect::<Result<VecDeque<_>>>()?;

        let mut inner_reader = readers.pop_front().unwrap();
        let mut outer_reader = readers.pop_front().unwrap();

        let mut inner_key_iter = inner_key.iter_mut();
        let mut inner_payload_iter = inner_payload.iter_mut();
        while inner_reader.read_byte_record(&mut record)? {
            let (key, value): (T, T) = record.deserialize(None)?;
            *inner_key_iter
                .next()
                .expect("Allocated length is too short") = key;
            *inner_payload_iter
                .next()
                .expect("Allocated length is too short") = value;
        }

        let mut outer_key_iter = outer_key.iter_mut();
        let mut outer_payload_iter = outer_payload.iter_mut();
        while outer_reader.read_byte_record(&mut record)? {
            let (key, value): (T, T) = record.deserialize(None)?;
            *outer_key_iter
                .next()
                .expect("Allocated length is too short") = key;
            *outer_payload_iter
                .next()
                .expect("Allocated length is too short") = value;
        }

        let io_read_time = io_timer.elapsed();

        Ok((
            HashJoinBench {
                hashing_scheme: self.hashing_scheme,
                is_selective: self.is_selective,
                hash_table_len: self.get_hash_table_len()?,
                build_relation_key: inner_key.into(),
                build_relation_payload: inner_payload.into(),
                probe_relation_key: outer_key.into(),
                probe_relation_payload: outer_payload.into(),
            },
            malloc_time,
            io_count_time + io_read_time,
        ))
    }
}

impl<T> HashJoinBench<T>
where
    T: Default
        + DeviceCopy
        + Sync
        + Send
        + no_partitioning_join::NullKey
        + no_partitioning_join::CudaHashJoinable
        + no_partitioning_join::CpuHashJoinable,
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
        let mut hash_table =
            no_partitioning_join::HashTable::new_on_gpu(hash_table_mem, self.hash_table_len)?;
        hash_table.mlock()?;
        let hash_table = hash_table;
        let ht_malloc_time = ht_malloc_timer.elapsed();

        let mut result_sums = allocator::Allocator::alloc_mem(
            allocator::MemType::CudaUniMem,
            (probe_dim.0.x * probe_dim.1.x) as usize,
        );

        // Initialize result
        if let CudaUniMem(ref mut c) = result_sums {
            c.iter_mut().map(|sum| *sum = 0).for_each(drop);
        }

        stream.synchronize()?;

        let hj_op = no_partitioning_join::CudaHashJoinBuilder::<T>::default()
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
            self.build_relation_key.as_launchable_slice(),
            self.build_relation_payload.as_launchable_slice(),
            &stream,
        )?;

        stop_event.record(&stream)?;
        stop_event.synchronize()?;
        let build_millis = stop_event.elapsed_time_f32(&start_event)?;

        start_event.record(&stream)?;
        hj_op.probe_sum(
            self.probe_relation_key.as_launchable_slice(),
            self.probe_relation_payload.as_launchable_slice(),
            &mut result_sums,
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
        gpu_morsel_bytes: usize,
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
        let mut hash_table =
            no_partitioning_join::HashTable::new_on_gpu(hash_table_mem, self.hash_table_len)?;
        hash_table.mlock()?;
        let hash_table = hash_table;
        let ht_malloc_time = ht_malloc_timer.elapsed();

        let mut result_sums = allocator::Allocator::alloc_mem(
            allocator::MemType::CudaUniMem,
            (probe_dim.0.x * probe_dim.1.x) as usize,
        );

        // Initialize sums
        if let CudaUniMem(ref mut c) = result_sums {
            c.iter_mut().map(|sum| *sum = 0).for_each(drop);
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

        let hj_op = no_partitioning_join::CudaHashJoinBuilder::<T>::default()
            .hashing_scheme(self.hashing_scheme)
            .is_selective(self.is_selective)
            .build_dim(build_dim.0.clone(), build_dim.1.clone())
            .probe_dim(probe_dim.0.clone(), probe_dim.1.clone())
            .hash_table(Arc::new(hash_table))
            .build()?;

        let mut build_relation = (build_rel_key, build_rel_pay);
        let mut probe_relation = (probe_rel_key, probe_rel_pay);
        let mut build_iter =
            build_relation.into_cuda_iter_with_strategy(transfer_strategy, gpu_morsel_bytes)?;
        let mut probe_iter =
            probe_relation.into_cuda_iter_with_strategy(transfer_strategy, gpu_morsel_bytes)?;

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
        gpu_morsel_bytes: usize,
    ) -> Result<HashJoinPoint> {
        let ht_malloc_timer = Instant::now();
        let hash_table_mem = hash_table_alloc(self.hash_table_len);
        let mut hash_table =
            no_partitioning_join::HashTable::new_on_gpu(hash_table_mem, self.hash_table_len)?;
        hash_table.mlock()?;
        let hash_table = hash_table;
        let ht_malloc_time = ht_malloc_timer.elapsed();

        let mut result_sums = allocator::Allocator::alloc_mem(
            allocator::MemType::CudaUniMem,
            (probe_dim.0.x * probe_dim.1.x) as usize,
        );

        // Initialize sums
        if let CudaUniMem(ref mut c) = result_sums {
            c.iter_mut().map(|sum| *sum = 0).for_each(drop);
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
        cpu_affinity: &CpuAffinity,
        hash_table_alloc: allocator::DerefMemAllocFn<T>,
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
                    .expect("Couldn't set CPU core affinity")
            })
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
        &mut self,
        hash_table_alloc: allocator::MemAllocFn<T>,
        cpu_threads: usize,
        worker_cpu_affinity: &WorkerCpuAffinity,
        gpu_ids: Vec<u16>,
        build_dim: (GridSize, BlockSize),
        probe_dim: (GridSize, BlockSize),
        morsel_spec: &MorselSpec,
    ) -> Result<HashJoinPoint> {
        // FIXME: specify load factor as argument
        let ht_malloc_timer = Instant::now();
        let hash_table_mem = hash_table_alloc(self.hash_table_len);
        let mut hash_table =
            no_partitioning_join::HashTable::new_on_gpu(hash_table_mem, self.hash_table_len)?;
        hash_table.mlock()?;
        let hash_table = Arc::new(hash_table);
        let ht_malloc_time = ht_malloc_timer.elapsed();

        // Note: CudaDevMem is initialized with zeroes by the allocator
        let result_sums: Mem<u64> = allocator::Allocator::alloc_mem(
            allocator::MemType::CudaDevMem,
            (probe_dim.0.x * probe_dim.1.x) as usize,
        );

        // Convert Mem<T> into &mut [T]
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

        let executor = HetMorselExecutorBuilder::new()
            .cpu_threads(cpu_threads)
            .worker_cpu_affinity(worker_cpu_affinity.clone())
            .gpu_ids(gpu_ids)
            .morsel_spec(morsel_spec.clone())
            .build()?;

        let build_timer = Instant::now();
        (build_rel_key, build_rel_pay)
            .into_het_morsel_iter(&executor)
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
            .into_het_morsel_iter(&executor)
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

                    // FIXME: retrieve sums of all threads, e.g., by implementing fold instead of map
                    hj_op
                        .probe_sum(rel, pay, &result_sums, stream)
                        .expect("Failed to run GPU hash join probe");

                    Ok(())
                },
            )?;

        let probe_time = probe_timer.elapsed();

        Ok(HashJoinPoint {
            build_ns: Some(build_time.as_nanos() as f64),
            probe_ns: Some(probe_time.as_nanos() as f64),
            hash_table_malloc_ns: Some(ht_malloc_time.as_nanos() as f64),
            ..Default::default()
        })
    }

    #[allow(dead_code)]
    pub fn gpu_build_heterogeneous_probe(
        &mut self,
        cpu_hash_table_alloc: allocator::MemAllocFn<T>,
        gpu_hash_table_alloc: allocator::MemAllocFn<T>,
        cpu_threads: usize,
        worker_cpu_affinity: &WorkerCpuAffinity,
        gpu_ids: Vec<u16>,
        build_dim: (GridSize, BlockSize),
        probe_dim: (GridSize, BlockSize),
        morsel_spec: &MorselSpec,
    ) -> Result<HashJoinPoint> {
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

        // Note: CudaDevMem is initialized with zeroes by the allocator
        let result_sums: Mem<u64> = allocator::Allocator::alloc_mem(
            allocator::MemType::CudaDevMem,
            (probe_dim.0.x * probe_dim.1.x) as usize,
        );

        // Convert Mem<T> into &mut [T]
        let probe_rel_key: &mut [T] = (&mut self.probe_relation_key)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");
        let probe_rel_pay: &mut [T] = (&mut self.probe_relation_payload)
            .try_into()
            .map_err(|(err, _)| err)
            .expect("Can't use CUDA device memory on CPU!");

        let gpu_hj_builder = no_partitioning_join::CudaHashJoinBuilder::<T>::default()
            .hashing_scheme(self.hashing_scheme)
            .is_selective(self.is_selective)
            .build_dim(build_dim.0.clone(), build_dim.1.clone())
            .probe_dim(probe_dim.0.clone(), probe_dim.1.clone())
            .hash_table(gpu_hash_table.clone());

        let executor = HetMorselExecutorBuilder::new()
            .cpu_threads(cpu_threads)
            .worker_cpu_affinity(worker_cpu_affinity.clone())
            .gpu_ids(gpu_ids)
            .morsel_spec(morsel_spec.clone())
            .build()?;

        let build_timer = Instant::now();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let gpu_hj_op = gpu_hj_builder.build()?;
        gpu_hj_op.build(
            self.build_relation_key.as_launchable_slice(),
            self.build_relation_payload.as_launchable_slice(),
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
            .into_het_morsel_iter(&executor)
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

                    // FIXME: retrieve sums of all threads, e.g., by implementing fold instead of map
                    hj_op
                        .probe_sum(rel, pay, &result_sums, stream)
                        .expect("Failed to run GPU hash join probe");

                    Ok(())
                },
            )?;
        let probe_time = probe_timer.elapsed();

        Ok(HashJoinPoint {
            build_ns: Some(build_time.as_nanos() as f64),
            probe_ns: Some(probe_time.as_nanos() as f64),
            hash_table_malloc_ns: Some(ht_malloc_time.as_nanos() as f64),
            ..Default::default()
        })
    }
}
