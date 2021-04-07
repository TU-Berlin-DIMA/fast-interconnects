/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2020-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

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

#[derive(Debug)]
pub struct CudaRadixJoin {
    radix_pass: RadixPass,
    radix_bits: RadixBits,
    hashing_scheme: HashingScheme,
    grid_size: GridSize,
    block_size: BlockSize,
}

impl CudaRadixJoin {
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
                        device.get_attribute(DeviceAttribute::MaxSharedMemPerBlockOptin)? as u32;
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
                            // Tuning parameter for the number of hash table buckets in the bucket-chained
                            // hashing scheme. Must be a power of 2, and at least 1. No further constraints.
                            args.ht_entries = 2048;

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
