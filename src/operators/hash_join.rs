/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

extern crate cuda_sys;
extern crate num_traits;
extern crate paste;
extern crate rustacuda;

use cuda_sys::cuda::cuMemsetD32_v2;

use self::num_traits::cast::AsPrimitive;

use self::rustacuda::function::{BlockSize, GridSize};
use self::rustacuda::memory::DeviceCopy;
use self::rustacuda::prelude::*;

use std::ffi::CString;
use std::mem::size_of;
use std::os::raw::{c_uint, c_void};
use std::sync::Arc;

use crate::error::{Error, ErrorKind, Result, ToResult};
use crate::runtime::allocator;
use crate::runtime::memory::*;

extern "C" {
    fn cpu_ht_build_linearprobing_int32(
        hash_table: *mut i32, // FIXME: replace i32 with atomic_i32
        hash_table_entries: u64,
        join_attr_data: *const i32,
        payload_attr_data: *const i32,
        data_length: u64,
    );

    fn cpu_ht_build_linearprobing_int64(
        hash_table: *mut i64, // FIXME: replace i64 with atomic_i64
        hash_table_entries: u64,
        join_attr_data: *const i64,
        payload_attr_data: *const i64,
        data_length: u64,
    );

    fn cpu_ht_probe_aggregate_linearprobing_int32(
        hash_table: *const i32, // FIXME: replace i32 with atomic_i32
        hash_table_entries: u64,
        join_attr_data: *const i32,
        payload_attr_data: *const i32,
        data_length: u64,
        aggregation_result: *mut u64,
    );

    fn cpu_ht_probe_aggregate_linearprobing_int64(
        hash_table: *const i64, // FIXME: replace i64 with atomic_i64
        hash_table_entries: u64,
        join_attr_data: *const i64,
        payload_attr_data: *const i64,
        data_length: u64,
        aggregation_result: *mut u64,
    );

    fn cpu_ht_build_perfect_int32(
        hash_table: *mut i32, // FIXME: replace i32 with atomic_i32
        hash_table_entries: u64,
        join_attr_data: *const i32,
        payload_attr_data: *const i32,
        data_length: u64,
    );

    fn cpu_ht_build_perfect_int64(
        hash_table: *mut i64, // FIXME: replace i64 with atomic_i64
        hash_table_entries: u64,
        join_attr_data: *const i64,
        payload_attr_data: *const i64,
        data_length: u64,
    );

    fn cpu_ht_probe_aggregate_perfect_int32(
        hash_table: *const i32, // FIXME: replace i32 with atomic_i32
        hash_table_entries: u64,
        join_attr_data: *const i32,
        payload_attr_data: *const i32,
        data_length: u64,
        aggregation_result: *mut u64,
    );

    fn cpu_ht_probe_aggregate_perfect_int64(
        hash_table: *const i64, // FIXME: replace i64 with atomic_i64
        hash_table_entries: u64,
        join_attr_data: *const i64,
        payload_attr_data: *const i64,
        data_length: u64,
        aggregation_result: *mut u64,
    );
}

#[derive(Clone, Copy, Debug)]
pub enum HashingScheme {
    Perfect,
    LinearProbing,
}

pub trait NullKey: AsPrimitive<c_uint> {
    fn null_key() -> Self;
}

impl NullKey for i32 {
    fn null_key() -> i32 {
        -1
    }
}

impl NullKey for i64 {
    fn null_key() -> i64 {
        -1
    }
}

pub trait CudaHashJoinable<T: DeviceCopy + NullKey> {
    fn build_impl(
        hj: &mut CudaHashJoin<T>,
        join_attr: &Mem<T>,
        payload_attr: &Mem<T>,
        stream: &Stream,
    ) -> Result<()>;
    fn probe_count_impl(
        hj: &mut CudaHashJoin<T>,
        join_attr: &Mem<T>,
        payload_attr: &Mem<T>,
        result_set: &mut Mem<u64>,
        stream: &Stream,
    ) -> Result<()>;
}

pub trait CpuHashJoinable<T: DeviceCopy + NullKey> {
    fn build_impl(hj: &mut CpuHashJoin<T>, join_attr: &[T], payload_attr: &[T]) -> Result<()>;
    fn probe_count_impl(
        hj: &mut CpuHashJoin<T>,
        join_attr: &[T],
        payload_attr: &[T],
        join_count: &mut u64,
    ) -> Result<()>;
}

#[derive(Debug)]
pub struct CudaHashJoin<T: DeviceCopy + NullKey> {
    ops: Module,
    hashing_scheme: HashingScheme,
    hash_table: HashTable<T>,
    build_dim: (GridSize, BlockSize),
    probe_dim: (GridSize, BlockSize),
}

#[derive(Debug)]
pub struct CpuHashJoin<T: DeviceCopy + NullKey> {
    hashing_scheme: HashingScheme,
    hash_table: Arc<HashTable<T>>,
}

#[derive(Debug)]
pub struct HashTable<T: DeviceCopy + NullKey> {
    mem: Mem<T>,
    size: usize,
}

#[derive(Debug)]
pub struct CudaHashJoinBuilder<T: DeviceCopy + NullKey> {
    hashing_scheme: HashingScheme,
    hash_table_i: Option<HashTable<T>>,
    build_dim_i: (GridSize, BlockSize),
    probe_dim_i: (GridSize, BlockSize),
}

#[derive(Debug)]
pub struct CpuHashJoinBuilder<T: DeviceCopy + NullKey> {
    hashing_scheme: HashingScheme,
    hash_table_i: Option<Arc<HashTable<T>>>,
}

impl<T> CudaHashJoin<T>
where
    T: DeviceCopy + NullKey + CudaHashJoinable<T>,
{
    pub fn build(
        &mut self,
        join_attr: &Mem<T>,
        payload_attr: &Mem<T>,
        stream: &Stream,
    ) -> Result<()> {
        T::build_impl(self, join_attr, payload_attr, stream)
    }
    pub fn probe_count(
        &mut self,
        join_attr: &Mem<T>,
        payload_attr: &Mem<T>,
        result_set: &mut Mem<u64>,
        stream: &Stream,
    ) -> Result<()> {
        T::probe_count_impl(self, join_attr, payload_attr, result_set, stream)
    }
}

impl<T> CpuHashJoin<T>
where
    T: DeviceCopy + NullKey + CpuHashJoinable<T>,
{
    pub fn build(&mut self, join_attr: &[T], payload_attr: &[T]) -> Result<()> {
        T::build_impl(self, join_attr, payload_attr)
    }

    pub fn probe_count(
        &mut self,
        join_attr: &[T],
        payload_attr: &[T],
        join_count: &mut u64,
    ) -> Result<()> {
        T::probe_count_impl(self, join_attr, payload_attr, join_count)
    }
}

macro_rules! impl_cuda_hash_join_for_type {
    ($Type:ty, $Suffix:expr) => {
        impl CudaHashJoinable<$Type> for $Type {
            paste::item!{
                fn build_impl(
                    hj: &mut CudaHashJoin<$Type>,
                    join_attr: &Mem<$Type>,
                    payload_attr: &Mem<$Type>,
                    stream: &Stream,
                    ) -> Result<()> {
                    ensure!(
                        join_attr.len() == payload_attr.len(),
                        "Join and payload attributes have different sizes"
                        );
                    ensure!(
                        join_attr.len() <= hj.hash_table.mem.len(),
                        "Hash table is too small for the build data"
                        );

                    let (grid, block) = hj.build_dim.clone();

                    let join_attr_len = join_attr.len() as u64;
                    let hash_table_size = hj.hash_table.size as u64;
                    let module = &hj.ops;

                    match &hj.hashing_scheme {
                        HashingScheme::Perfect => unsafe{ launch!(
                                module.[<gpu_ht_build_perfect_ $Suffix>]<<<grid, block, 0, stream>>>(
                                    hj.hash_table.mem.as_launchable_mut_ptr(),
                                    hash_table_size,
                                    join_attr.as_launchable_ptr(),
                                    payload_attr.as_launchable_ptr(),
                                    join_attr_len
                                    )
                                )? },
                        HashingScheme::LinearProbing => unsafe { launch!(
                                module.[<gpu_ht_build_linearprobing_ $Suffix>]<<<grid, block, 0, stream>>>(
                                    hj.hash_table.mem.as_launchable_mut_ptr(),
                                    hash_table_size,
                                    join_attr.as_launchable_ptr(),
                                    payload_attr.as_launchable_ptr(),
                                    join_attr_len
                                    )
                                )? },
                    };

                    Ok(())
                }
            }

            paste::item!{
                fn probe_count_impl(
                    hj: &mut CudaHashJoin<$Type>,
                    join_attr: &Mem<$Type>,
                    payload_attr: &Mem<$Type>,
                    result_set: &mut Mem<u64>,
                    stream: &Stream,
                    ) -> Result<()> {
                    let (grid, block) = hj.probe_dim.clone();
                    ensure!(
                        result_set.len() >= (grid.x * block.x) as usize,
                        "Result set size is too small, must be at least grid * block size"
                        );
                    ensure!(
                        join_attr.len() == payload_attr.len(),
                        "Join and payload attributes have different sizes"
                        );

                    let join_attr_len = join_attr.len() as u64;
                    let hash_table_size = hj.hash_table.size as u64;
                    let module = &hj.ops;

                    match &hj.hashing_scheme {
                        HashingScheme::Perfect => unsafe { launch!(
                                module.[<gpu_ht_probe_aggregate_perfect_ $Suffix>]<<<grid, block, 0, stream>>>(
                                    hj.hash_table.mem.as_launchable_ptr(),
                                    hash_table_size,
                                    join_attr.as_launchable_ptr(),
                                    payload_attr.as_launchable_ptr(),
                                    join_attr_len,
                                    result_set.as_launchable_mut_ptr()
                                    )
                                )? },
                        HashingScheme::LinearProbing => unsafe { launch!(
                                module.[<gpu_ht_probe_aggregate_linearprobing_ $Suffix>]<<<grid, block, 0, stream>>>(
                                    hj.hash_table.mem.as_launchable_ptr(),
                                    hash_table_size,
                                    join_attr.as_launchable_ptr(),
                                    payload_attr.as_launchable_ptr(),
                                    join_attr_len,
                                    result_set.as_launchable_mut_ptr()
                                    )
                                )? },
                    };

                    Ok(())
                }
            }
        }
    };
}

impl_cuda_hash_join_for_type!(i32, int32);
impl_cuda_hash_join_for_type!(i64, int64);

macro_rules! impl_cpu_hash_join_for_type {
    ($Type:ty, $Suffix:expr) => {
        impl CpuHashJoinable<$Type> for $Type {
            paste::item!{
                fn build_impl(hj: &mut CpuHashJoin<$Type>, join_attr: &[$Type], payload_attr: &[$Type]) -> Result<()> {
                    ensure!(
                        join_attr.len() == payload_attr.len(),
                        "Join and payload attributes have different sizes"
                        );
                    ensure!(
                        join_attr.len() <= hj.hash_table.mem.len(),
                        "Hash table is too small for the build data"
                        );

                    let join_attr_len = join_attr.len() as u64;
                    let hash_table_size = hj.hash_table.size as u64;

                    match &hj.hashing_scheme {
                        HashingScheme::Perfect => unsafe {
                            [<cpu_ht_build_perfect_ $Suffix>](
                                hj.hash_table.mem.as_ptr() as *mut $Type,
                                hash_table_size,
                                join_attr.as_ptr(),
                                payload_attr.as_ptr(),
                                join_attr_len,
                                )
                        },
                        HashingScheme::LinearProbing => unsafe {
                            [<cpu_ht_build_linearprobing_ $Suffix>](
                                hj.hash_table.mem.as_ptr() as *mut $Type,
                                hash_table_size,
                                join_attr.as_ptr(),
                                payload_attr.as_ptr(),
                                join_attr_len,
                                )
                        },
                    };

                    Ok(())
                }
            }

            paste::item!{
                fn probe_count_impl(
                    hj: &mut CpuHashJoin<$Type>,
                    join_attr: &[$Type],
                    payload_attr: &[$Type],
                    join_count: &mut u64,
                    ) -> Result<()> {
                    ensure!(
                        join_attr.len() == payload_attr.len(),
                        "Join and payload attributes have different sizes"
                        );

                    let join_attr_len = join_attr.len() as u64;
                    let hash_table_size = hj.hash_table.size as u64;

                    match &hj.hashing_scheme {
                        HashingScheme::Perfect => unsafe {
                            [<cpu_ht_probe_aggregate_perfect_ $Suffix>](
                                hj.hash_table.mem.as_ptr(),
                                hash_table_size,
                                join_attr.as_ptr(),
                                payload_attr.as_ptr(),
                                join_attr_len,
                                join_count,
                                )
                        },
                        HashingScheme::LinearProbing => unsafe {
                            [<cpu_ht_probe_aggregate_linearprobing_ $Suffix>](
                                hj.hash_table.mem.as_ptr(),
                                hash_table_size,
                                join_attr.as_ptr(),
                                payload_attr.as_ptr(),
                                join_attr_len,
                                join_count,
                                )
                        },
                    };

                    Ok(())
                }
            }
        }
    };
}

impl_cpu_hash_join_for_type!(i32, int32);
impl_cpu_hash_join_for_type!(i64, int64);

impl<T: DeviceCopy + NullKey> HashTable<T> {
    // FIXME: make generic for type T
    pub fn new_on_cpu(mut mem: DerefMem<T>, size: usize) -> Result<Self> {
        ensure!(
            size.is_power_of_two(),
            "Hash table size must be a power of two"
        );
        ensure!(
            mem.len() >= size,
            "Provided memory must be larger than hash table size"
        );

        mem.iter_mut().by_ref().for_each(|x| *x = T::null_key());

        Ok(Self {
            mem: mem.into(),
            size,
        })
    }

    pub fn new_on_gpu(mem: Mem<T>, size: usize) -> Result<Self> {
        ensure!(
            size.is_power_of_two(),
            "Hash table size must be a power of two"
        );
        ensure!(
            mem.len() >= size,
            "Provided memory must be larger than hash table size"
        );

        // Initialize hash table
        unsafe {
            cuMemsetD32_v2(
                mem.as_ptr() as *mut c_void as u64,
                T::null_key().as_(),
                mem.len()
                    .checked_mul(size_of::<T>() / size_of::<c_uint>())
                    .ok_or_else(|| {
                        ErrorKind::IntegerOverflow("Failed to compute hash table bytes".to_string())
                    })?,
            )
        }
        .to_result()?;

        Ok(Self { mem, size })
    }
}

impl<T: DeviceCopy + NullKey> ::std::default::Default for CudaHashJoinBuilder<T> {
    fn default() -> Self {
        Self {
            hashing_scheme: HashingScheme::default(),
            hash_table_i: None,
            build_dim_i: (1.into(), 1.into()),
            probe_dim_i: (1.into(), 1.into()),
        }
    }
}

impl<T> CudaHashJoinBuilder<T>
where
    T: Clone + Default + DeviceCopy + NullKey,
{
    const DEFAULT_HT_SIZE: usize = 1024;

    pub fn hashing_scheme(mut self, hashing_scheme: HashingScheme) -> Self {
        self.hashing_scheme = hashing_scheme;
        self
    }

    pub fn hash_table(mut self, ht: HashTable<T>) -> Self {
        self.hash_table_i = Some(ht);
        self
    }

    pub fn build_dim(mut self, grid: GridSize, block: BlockSize) -> Self {
        self.build_dim_i = (grid, block);
        self
    }

    pub fn probe_dim(mut self, grid: GridSize, block: BlockSize) -> Self {
        self.probe_dim_i = (grid, block);
        self
    }

    pub fn build(self) -> Result<CudaHashJoin<T>> {
        ensure!(self.hash_table_i.is_some(), "Hash table not set");

        let module_path = CString::new(env!("CUDAUTILS_PATH")).map_err(|e| {
            Error::with_chain(
                e,
                ErrorKind::InvalidArgument(
                    "Failed to load CUDA module, check your CUDAUTILS_PATH".to_string(),
                ),
            )
        })?;

        let ops = Module::load_from_file(&module_path)?;

        let hash_table = if let Some(ht) = self.hash_table_i {
            ht
        } else {
            HashTable {
                mem: allocator::Allocator::alloc_mem::<T>(
                    allocator::MemType::CudaUniMem,
                    Self::DEFAULT_HT_SIZE,
                ),
                size: Self::DEFAULT_HT_SIZE,
            }
        };

        Ok(CudaHashJoin {
            ops,
            hashing_scheme: self.hashing_scheme,
            hash_table,
            build_dim: self.build_dim_i,
            probe_dim: self.probe_dim_i,
        })
    }
}

impl<T: DeviceCopy + NullKey> ::std::default::Default for CpuHashJoinBuilder<T> {
    fn default() -> Self {
        Self {
            hashing_scheme: HashingScheme::default(),
            hash_table_i: None,
        }
    }
}

impl<T: Default + DeviceCopy + NullKey> CpuHashJoinBuilder<T> {
    const DEFAULT_HT_SIZE: usize = 1024;

    pub fn hashing_scheme(mut self, hashing_scheme: HashingScheme) -> Self {
        self.hashing_scheme = hashing_scheme;
        self
    }

    pub fn hash_table(mut self, hash_table: Arc<HashTable<T>>) -> Self {
        self.hash_table_i = Some(hash_table);
        self
    }

    pub fn build(&self) -> CpuHashJoin<T> {
        let hash_table = match &self.hash_table_i {
            Some(ht) => ht.clone(),
            None => Arc::new(HashTable {
                mem: allocator::Allocator::alloc_mem::<T>(
                    allocator::MemType::SysMem,
                    Self::DEFAULT_HT_SIZE,
                ),
                size: Self::DEFAULT_HT_SIZE,
            }),
        };

        CpuHashJoin {
            hashing_scheme: self.hashing_scheme,
            hash_table,
        }
    }
}

impl<T> ::std::fmt::Display for HashTable<T>
where
    T: DeviceCopy + ::std::fmt::Display + NullKey,
{
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        if let CudaDevMem(_) = self.mem {
            write!(f, "[Cannot print device memory]")?;
        } else {
            write!(f, "[")?;
            match self.mem {
                SysMem(ref m) => m.as_slice(),
                CudaUniMem(ref m) => m.as_slice(),
                _ => &[],
            }
            .iter()
            .take(self.size)
            .map(|entry| write!(f, "{},", entry))
            .collect::<::std::fmt::Result>()?;
            write!(f, "]")?;
        }

        Ok(())
    }
}

impl ::std::default::Default for HashingScheme {
    fn default() -> Self {
        HashingScheme::LinearProbing
    }
}
