/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019-2020, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use crate::error::{ErrorKind, Result};
use crate::types::*;
use crate::DataGenFn;
use csv::{ByteRecord, ReaderBuilder};
use flate2::read::GzDecoder;
use num_rational::Ratio;
use numa_gpu::runtime::allocator::{self, Allocator};
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::memory::*;
use numa_gpu::runtime::numa::NodeRatio;
use numa_gpu::runtime::utils::EnsurePhysicallyBacked;
use rustacuda::context::{CacheConfig, CurrentContext, SharedMemoryConfig};
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::DeviceCopy;
use rustacuda::stream::{Stream, StreamFlags};
use serde::de::DeserializeOwned;
use sql_ops::join::{no_partitioning_join, HashingScheme};
use sql_ops::partition::gpu_radix_partition::{
    GpuHistogramAlgorithm, GpuRadixPartitionAlgorithm, GpuRadixPartitionable, GpuRadixPartitioner,
    PartitionOffsets, PartitionedRelation, RadixPartitionInputChunkable,
};
use std::collections::vec_deque::VecDeque;
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct RadixJoinBench<T: DeviceCopy> {
    pub hashing_scheme: HashingScheme,
    pub build_relation_key: Mem<T>,
    pub build_relation_payload: Mem<T>,
    pub probe_relation_key: Mem<T>,
    pub probe_relation_payload: Mem<T>,
}

pub struct RadixJoinBenchBuilder {
    hash_table_load_factor: usize,
    inner_len: usize,
    outer_len: usize,
    inner_location: Box<[NodeRatio]>,
    outer_location: Box<[NodeRatio]>,
    inner_mem_type: ArgMemType,
    outer_mem_type: ArgMemType,
    hashing_scheme: HashingScheme,
}

#[derive(Debug, Default)]
pub struct RadixJoinPoint {
    pub partitions_malloc_ns: Option<f64>,
    pub prefix_sum_ns: Option<f64>,
    pub partition_ns: Option<f64>,
    pub join_ns: Option<f64>,
}

impl Default for RadixJoinBenchBuilder {
    fn default() -> RadixJoinBenchBuilder {
        RadixJoinBenchBuilder {
            hash_table_load_factor: 2,
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
            hashing_scheme: HashingScheme::LinearProbing,
        }
    }
}

impl RadixJoinBenchBuilder {
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

    pub fn hashing_scheme(&mut self, hashing_scheme: HashingScheme) -> &mut Self {
        self.hashing_scheme = hashing_scheme;
        self
    }

    fn allocate_relations<T>(
        &self,
    ) -> Result<(DerefMem<T>, DerefMem<T>, DerefMem<T>, DerefMem<T>, Duration)>
    where
        T: Clone + Default + DeviceCopy + EnsurePhysicallyBacked,
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
                }
                .into(),
                len,
            );

            // If user selected NumaLazyPinned, then pin the memory.
            // If user selected Unified, then ensure that we measure unified memory transfers from CPU memory
            if let (ArgMemType::NumaLazyPinned, DerefMem::NumaMem(lazy_pinned_mem)) =
                (mem_type, &mut mem)
            {
                lazy_pinned_mem
                    .page_lock()
                    .expect("Failed to lazily pin memory");
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

    pub fn build_with_data_gen<T>(
        &mut self,
        mut data_gen_fn: DataGenFn<T>,
    ) -> Result<(RadixJoinBench<T>, Duration, Duration)>
    where
        T: Copy + Default + DeviceCopy + EnsurePhysicallyBacked,
    {
        let (mut inner_key, inner_payload, mut outer_key, outer_payload, malloc_time) =
            self.allocate_relations()?;

        // Generate dataset
        let gen_timer = Instant::now();
        data_gen_fn(inner_key.as_mut_slice(), outer_key.as_mut_slice())?;
        let gen_time = gen_timer.elapsed();

        Ok((
            RadixJoinBench {
                hashing_scheme: self.hashing_scheme,
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
    ) -> Result<(RadixJoinBench<T>, Duration, Duration)>
    where
        T: Copy + Default + DeviceCopy + EnsurePhysicallyBacked,
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
            RadixJoinBench {
                hashing_scheme: self.hashing_scheme,
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

impl<T> RadixJoinBench<T>
where
    T: Default
        + DeviceCopy
        + Sync
        + Send
        + GpuRadixPartitionable
        + no_partitioning_join::NullKey
        + no_partitioning_join::CudaHashJoinable
        + no_partitioning_join::CpuHashJoinable
        + EnsurePhysicallyBacked,
{
    pub fn gpu_radix_join(
        &mut self,
        histogram_algorithm: GpuHistogramAlgorithm,
        partition_algorithm: GpuRadixPartitionAlgorithm,
        radix_bits: u32,
        dmem_buffer_bytes: usize,
        threads: usize,
        cpu_affinity: CpuAffinity,
        partitions_mem_type: allocator::MemType,
        partition_dim: (GridSize, BlockSize),
        _build_dim: (GridSize, BlockSize),
        probe_dim: (GridSize, BlockSize),
    ) -> Result<RadixJoinPoint> {
        // FIXME: specify hash table load factor as argument

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
            &partition_dim.0,
            &partition_dim.1,
            dmem_buffer_bytes,
        )?;

        let mut inner_rel_partitions = PartitionedRelation::new(
            self.build_relation_key.len(),
            histogram_algorithm,
            radix_bits,
            partition_dim.0.x,
            Allocator::mem_alloc_fn(partitions_mem_type.clone()),
            Allocator::mem_alloc_fn(partitions_mem_type.clone()),
        );

        let mut outer_rel_partitions = PartitionedRelation::new(
            self.probe_relation_key.len(),
            histogram_algorithm,
            radix_bits,
            partition_dim.0.x,
            Allocator::mem_alloc_fn(partitions_mem_type.clone()),
            Allocator::mem_alloc_fn(partitions_mem_type.clone()),
        );

        let mut inner_rel_partition_offsets = PartitionOffsets::new(
            histogram_algorithm,
            partition_dim.0.x,
            radix_bits,
            Allocator::mem_alloc_fn(allocator::MemType::CudaUniMem),
        );

        let mut outer_rel_partition_offsets = PartitionOffsets::new(
            histogram_algorithm,
            partition_dim.0.x,
            radix_bits,
            Allocator::mem_alloc_fn(allocator::MemType::CudaUniMem),
        );

        let mut result_sums = allocator::Allocator::alloc_mem(
            allocator::MemType::CudaUniMem,
            (probe_dim.0.x * probe_dim.1.x) as usize,
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
                let inner_key_slice: &[T] =
                    (&self.build_relation_key).try_into().map_err(|_| {
                        ErrorKind::RuntimeError(
                            "Failed to run CPU prefix sum on device memory".into(),
                        )
                    })?;
                let inner_key_chunks = inner_key_slice.input_chunks::<T>(&radix_prnr)?;
                let inner_offsets_chunks = inner_rel_partition_offsets.chunks_mut();

                let outer_key_slice: &[T] =
                    (&self.probe_relation_key).try_into().map_err(|_| {
                        ErrorKind::RuntimeError(
                            "Failed to run CPU prefix sum on device memory".into(),
                        )
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
                    self.build_relation_key.as_launchable_slice(),
                    &mut inner_rel_partition_offsets,
                    &stream,
                )?;
                radix_prnr.prefix_sum(
                    self.probe_relation_key.as_launchable_slice(),
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
            self.build_relation_key.as_launchable_slice(),
            self.build_relation_payload.as_launchable_slice(),
            inner_rel_partition_offsets,
            &mut inner_rel_partitions,
            &stream,
        )?;

        // Partition outer relation
        radix_prnr.partition(
            self.probe_relation_key.as_launchable_slice(),
            self.probe_relation_payload.as_launchable_slice(),
            outer_rel_partition_offsets,
            &mut outer_rel_partitions,
            &stream,
        )?;

        stream.synchronize()?;
        let partition_time = partition_timer.elapsed();

        let join_timer = Instant::now();
        let join_time = join_timer.elapsed();

        stream.synchronize()?;
        Ok(RadixJoinPoint {
            prefix_sum_ns: Some(prefix_sum_time.as_nanos() as f64),
            partition_ns: Some(partition_time.as_nanos() as f64),
            join_ns: Some(join_time.as_nanos() as f64),
            partitions_malloc_ns: Some(partitions_malloc_time.as_nanos() as f64),
        })
    }
}
