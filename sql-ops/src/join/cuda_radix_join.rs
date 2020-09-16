/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2020 Clemens Lutz, German Research Center for Artificial Intelligence
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use super::no_partitioning_join::NullKey;
use super::HashingScheme;
use crate::error::{ErrorKind, Result};
use crate::partition::gpu_radix_partition::PartitionedRelation;
use crate::partition::Tuple;
use numa_gpu::runtime::memory::{LaunchableMutPtr, LaunchableMutSlice, LaunchablePtr};
use rustacuda::context::CurrentContext;
use rustacuda::device::DeviceAttribute;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::launch;
use rustacuda::memory::DeviceCopy;
use rustacuda::module::Module;
use rustacuda::stream::Stream;
use std::ffi::{self, CString};
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
    aggregation_result: LaunchableMutPtr<u64>,
    task_assignments: LaunchableMutPtr<u32>,
    build_rel_len: u32,
    probe_rel_len: u32,
    build_rel_padding_len: u32,
    probe_rel_padding_len: u32,
    radix_bits: u32,
    ht_entries: u32,
}

unsafe impl DeviceCopy for JoinAggregateArgs {}

/// A hash table entry in the C/C++ implementation.
///
/// Note that the struct's layout must be kept in sync with its counterpart in
/// C/C++.
#[repr(C)]
#[derive(Clone, Debug)]
struct HtEntry<K, P> {
    key: K,
    payload: P,
}

pub trait CudaRadixJoinable: DeviceCopy + NullKey {
    fn join_impl(
        rj: &CudaRadixJoin,
        radix_bits: u32,
        build_rel: &PartitionedRelation<Tuple<Self, Self>>,
        probe_rel: &PartitionedRelation<Tuple<Self, Self>>,
        result_set: &mut LaunchableMutSlice<u64>,
        task_assignments: &mut LaunchableMutSlice<u32>,
        stream: &Stream,
    ) -> Result<()>;
}

#[derive(Debug)]
pub struct CudaRadixJoin {
    module: Module,
    hashing_scheme: HashingScheme,
    dim: (GridSize, BlockSize),
}

impl CudaRadixJoin {
    pub fn new(hashing_scheme: HashingScheme, dim: (GridSize, BlockSize)) -> Result<Self> {
        let module_path = CString::new(env!("CUDAUTILS_PATH")).map_err(|_| {
            ErrorKind::NulCharError(
                "Failed to load CUDA module, check your CUDAUTILS_PATH".to_string(),
            )
        })?;

        let module = Module::load_from_file(&module_path)?;

        Ok(Self {
            module,
            hashing_scheme,
            dim,
        })
    }

    pub fn join<T>(
        &self,
        radix_bits: u32,
        build_rel: &PartitionedRelation<Tuple<T, T>>,
        probe_rel: &PartitionedRelation<Tuple<T, T>>,
        result_set: &mut LaunchableMutSlice<u64>,
        task_assignments: &mut LaunchableMutSlice<u32>,
        stream: &Stream,
    ) -> Result<()>
    where
        T: DeviceCopy + NullKey + CudaRadixJoinable,
    {
        T::join_impl(
            self,
            radix_bits,
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
impl CudaRadixJoinable for i32 {
    fn join_impl(
        rj: &CudaRadixJoin,
        radix_bits: u32,
        build_rel: &PartitionedRelation<Tuple<Self, Self>>,
        probe_rel: &PartitionedRelation<Tuple<Self, Self>>,
        result_set: &mut LaunchableMutSlice<u64>,
        task_assignments: &mut LaunchableMutSlice<u32>,
        stream: &Stream,
    ) -> Result<()> {
        let (grid, block) = rj.dim.clone();

        if build_rel.chunks() != 1 {
            Err(ErrorKind::InvalidArgument(
                "Chunked build relations are not supported, use a contiguous relation instead"
                    .to_string(),
            ))?;
        }
        if probe_rel.chunks() != 1 {
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
        let module = &rj.module;
        let device = CurrentContext::get_device()?;
        let max_shared_mem_bytes =
            device.get_attribute(DeviceAttribute::MaxSharedMemPerBlockOptin)? as u32;

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
                    std::ffi::CString::new(stringify!(gpu_join_aggregate_smem_perfect_i32_i32_i32))
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
        };

        Ok(())
    }
}
