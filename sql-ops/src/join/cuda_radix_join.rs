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

//! Radix join operators for GPUs.
//!
//! The radix join is specified to join two relations. The current method results in a set of
//! aggregates, i.e., one aggregate per thread. These must be summed up by the caller, e.g., on the
//! CPU.
//!
//! Specifically, the current join implementation performs:
//!
//! ```sql
//! SELECT SUM(s.value)
//! FROM r
//! JOIN s ON r.key = s.key
//! ```
//!
//! Note that `r.value` is inserted into the hash table in order to read all four columns. However,
//! this wouldn't be strictly necessary to correctly answer the query.
//!
//! ## Hashing schemes
//!
//! Perfect hashing (i.e., join key as array index) and bucket chaining are implemented. Linear
//! probing is currently not implemented. Bucket chaining was used because the radix joins by
//! Balkesen et al. and Sioulas et al. also use this scheme.
//!
//! According to our measurements, the hashing scheme doesn't affect the join performance (see the
//! Triton join paper). This might change if a higher fanout is required to fit the hash table into
//! shared memory, e.g., due to the load factor of linear probing.
//!
//! ## Skew handling
//!
//! One call to `CudaRadixJoin::join` processes all partitions. Before the join starts, we
//! calculate how large each partition is and assign partitions evenly among the thread blocks,
//! using a greedy algorithm. Then, each thread block processes the partitions assigned to it in
//! parallel.
//!
//! This assignment method thus handles skew, as long as none of the hash tables exceeds the shared
//! memory capacity.
//!
//! Handling a high degree of skew would require dynamic recursive partitioning. I.e., if a
//! partition exceeds the shared memory capacity, it should be recursively partitioned until all
//! subpartitions fit into shared memory. Alternatively, spilling (parts of) the hash table to GPU
//! memory would be possible as well.

use super::{HashingScheme, HtEntry};
use crate::error::{ErrorKind, Result};
use crate::partition::PartitionedRelation;
use crate::partition::Tuple;
use crate::partition::{RadixBits, RadixPass};
use datagen::relation::KeyAttribute;
use numa_gpu::runtime::memory::{LaunchableMutPtr, LaunchableMutSlice, LaunchablePtr};
use rustacuda::context::CurrentContext;
use rustacuda::device::DeviceAttribute;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::launch;
use rustacuda::memory::DeviceCopy;
use rustacuda::stream::Stream;
use std::ffi;
use std::mem;

/// Arguments to the C/C++ join-aggregate function.
///
/// Note that the struct's layout must be kept in sync with its counterpart in
/// C/C++.
#[repr(C)]
#[derive(Clone, Debug)]
struct JoinAggregateArgs {
    build_rel: LaunchablePtr<ffi::c_void>,
    build_rel_partition_offsets: LaunchablePtr<u64>,
    probe_rel: LaunchablePtr<ffi::c_void>,
    probe_rel_partition_offsets: LaunchablePtr<u64>,
    aggregation_result: LaunchableMutPtr<i64>,
    task_assignments: LaunchableMutPtr<u32>,
    build_rel_len: u32,
    probe_rel_len: u32,
    build_rel_padding_len: u32,
    probe_rel_padding_len: u32,
    radix_bits: u32,
    ignore_bits: u32,
    ht_entries: u32,
}

unsafe impl DeviceCopy for JoinAggregateArgs {}

/// Specifies that the implementing type can be used as a join key in `CudaRadixJoin`.
///
/// CudaRadixJoinable is a trait for which specialized implementations exist for each implementing
/// type (currently i32 and i64). Specialization is necessary because each type requires a
/// different CUDA function to be called.
pub trait CudaRadixJoinable: DeviceCopy + KeyAttribute {
    fn join_impl(
        rj: &CudaRadixJoin,
        build_rel: &PartitionedRelation<Tuple<Self, Self>>,
        probe_rel: &PartitionedRelation<Tuple<Self, Self>>,
        result_set: &mut LaunchableMutSlice<i64>,
        task_assignments: &mut LaunchableMutSlice<u32>,
        stream: &Stream,
    ) -> Result<()>;
}

/// GPU radix join implementation in CUDA.
///
/// See the module documentation for details.
#[derive(Debug)]
pub struct CudaRadixJoin {
    radix_pass: RadixPass,
    radix_bits: RadixBits,
    hashing_scheme: HashingScheme,
    grid_size: GridSize,
    block_size: BlockSize,
}

impl CudaRadixJoin {
    /// Create a new radix join instance.
    pub fn new(
        radix_pass: RadixPass,
        radix_bits: RadixBits,
        hashing_scheme: HashingScheme,
        grid_size: &GridSize,
        block_size: &BlockSize,
    ) -> Result<Self> {
        Ok(Self {
            radix_pass,
            radix_bits,
            hashing_scheme,
            grid_size: grid_size.clone(),
            block_size: block_size.clone(),
        })
    }

    /// Join two relations and output a set of aggregate values.
    pub fn join<T>(
        &self,
        build_rel: &PartitionedRelation<Tuple<T, T>>,
        probe_rel: &PartitionedRelation<Tuple<T, T>>,
        result_set: &mut LaunchableMutSlice<i64>,
        task_assignments: &mut LaunchableMutSlice<u32>,
        stream: &Stream,
    ) -> Result<()>
    where
        T: DeviceCopy + KeyAttribute + CudaRadixJoinable,
    {
        T::join_impl(
            self,
            build_rel,
            probe_rel,
            result_set,
            task_assignments,
            stream,
        )
    }
}

// FIXME: build_rel and probe_rel should be of type PartitionedRelationSlice, i.e., immutable
// FIXME: add i64 implementation
macro_rules! impl_cuda_radix_join_for_type {
    ($Type:ty, $Suffix:expr) => {
        impl CudaRadixJoinable for $Type {
            paste::item! {
                fn join_impl(
                    rj: &CudaRadixJoin,
                    build_rel: &PartitionedRelation<Tuple<Self, Self>>,
                    probe_rel: &PartitionedRelation<Tuple<Self, Self>>,
                    result_set: &mut LaunchableMutSlice<i64>,
                    task_assignments: &mut LaunchableMutSlice<u32>,
                    stream: &Stream,
                    ) -> Result<()> {
                    let grid = &rj.grid_size;
                    let block = &rj.block_size;

                    if build_rel.num_chunks() != 1 {
                        Err(ErrorKind::InvalidArgument(
                                "Chunked build relations are not supported, use a contiguous relation instead"
                                .to_string(),
                                ))?;
                    }
                    if probe_rel.num_chunks() != 1 {
                        Err(ErrorKind::InvalidArgument(
                                "Chunked probe relations are not supported, use a contiguous relation instead"
                                .to_string(),
                                ))?;
                    }
                    if grid.x + 1 != task_assignments.len() as u32 {
                        Err(ErrorKind::InvalidArgument(
                                "Task assignement array must have length: grid size + 1".to_string(),
                                ))?;
                    }

                    let build_rel_len = build_rel.relation.len() as u32;
                    let probe_rel_len = probe_rel.relation.len() as u32;
                    let module = *crate::MODULE;
                    let device = CurrentContext::get_device()?;
                    let max_shared_mem_bytes =
                        device.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlockOptin)? as u32;
                    let radix_bits = rj.radix_bits.pass_radix_bits(rj.radix_pass).unwrap();
                    let ignore_bits = rj.radix_bits.pass_ignore_bits(rj.radix_pass) + radix_bits;

                    let mut args = JoinAggregateArgs {
                        build_rel: build_rel.relation.as_launchable_ptr().as_void(),
                        build_rel_partition_offsets: build_rel.offsets.as_launchable_ptr(),
                        probe_rel: probe_rel.relation.as_launchable_ptr().as_void(),
                        probe_rel_partition_offsets: probe_rel.offsets.as_launchable_ptr(),
                        aggregation_result: result_set.as_launchable_mut_ptr(),
                        task_assignments: task_assignments.as_launchable_mut_ptr(),
                        build_rel_len,
                        probe_rel_len,
                        build_rel_padding_len: build_rel.padding_len(),
                        probe_rel_padding_len: probe_rel.padding_len(),
                        radix_bits,
                        ignore_bits,
                        ht_entries: 0,
                    };

                    unsafe {
                        launch!(
                            module.gpu_radix_join_assign_tasks<<<grid.clone(), 1, 0, stream>>>(
                                args.clone())
                            )?;
                    }

                    match &rj.hashing_scheme {
                        HashingScheme::Perfect => {
                            let ht_entries =
                                max_shared_mem_bytes as usize / mem::size_of::<HtEntry<Self, Self>>() - 1;
                            args.ht_entries = ht_entries as u32;

                            let name =
                                std::ffi::CString::new(stringify!([<gpu_join_aggregate_smem_perfect_ $Suffix _ $Suffix _ $Suffix>]))
                                .unwrap();
                            let mut function = module.get_function(&name)?;
                            function.set_max_dynamic_shared_size_bytes(max_shared_mem_bytes)?;

                            unsafe {
                                launch!(
                                    function<<<grid, block, max_shared_mem_bytes, stream>>>(
                                        args.clone()
                                        )
                                    )?;
                            }
                        }
                        HashingScheme::LinearProbing => unimplemented!(),
                        HashingScheme::BucketChaining => {
                            args.ht_entries = crate::constants::RADIX_JOIN_BUCKET_CHAINING_ENTRIES;

                            let name = std::ffi::CString::new(stringify!([<
                                     gpu_join_aggregate_smem_chaining_ $Suffix _ $Suffix _ $Suffix
                                 >]))
                                .unwrap();
                            let mut function = module.get_function(&name)?;
                            function.set_max_dynamic_shared_size_bytes(max_shared_mem_bytes)?;

                            unsafe {
                                launch!(
                                    function<<<grid, block, max_shared_mem_bytes, stream>>>(
                                        args.clone(),
                                        max_shared_mem_bytes
                                        )
                                    )?;
                            }
                        }
                    };

                    Ok(())
                }
            }
        }
    }
}

impl_cuda_radix_join_for_type!(i32, i32);
impl_cuda_radix_join_for_type!(i64, i64);
