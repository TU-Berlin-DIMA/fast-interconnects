/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2019 Clemens Lutz, German Research Center for Artificial Intelligence
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

//! Hash join operators for CPU and GPU.
//!
//! The hash join operator supports x86_64 and PPC64 CPUs and CUDA GPUs.
//! The hash join can be executed in parallel. Heterogeneous parallel execution
//! on CPUs and GPUs is also supported. This is possible because the
//! processor-specific operators use the same underlying hash table.
//!
//! To execute in parallel on a CPU, the `build` and `probe_sum` methods of
//! `CpuHashJoin` must be called from multiple threads on non-overlapping data.
//! `build` must be completed on all threads before calling `probe_sum`.
//! This design was chosen to maximize flexibility on which cores to execute on.
//!
//! To execute in parallel on a GPU, it is sufficient to call `build` and
//! `probe_sum` once. Both methods require grid and block sizes as input,
//! that specify the parallelism with which to execute on the GPU. The join
//! can also be parallelized over multiple GPUs by calling the methods multiple
//! times using different CUDA devices.

use super::HashingScheme;
use crate::error::{ErrorKind, Result};
use cuda_sys::cuda::cuMemsetD32_v2;
use num_traits::cast::AsPrimitive;
use numa_gpu::error::ToResult;
use numa_gpu::runtime::allocator;
use numa_gpu::runtime::memory::*;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::launch;
use rustacuda::memory::DeviceCopy;
use rustacuda::prelude::*;
use std::ffi::CString;
use std::mem::size_of;
use std::os::raw::{c_uint, c_void};
use std::sync::Arc;

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

/// Specifies the null key value of the given type.
///
/// The null key is expected to have a binary representation of all ones. For
/// signed integers, that value equals -1, for unsigned integers, the value
/// equals 0xF...F.
///
/// The null key in Rust must be kept in sync with the null key in C++ and CUDA.
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

/// Specifies that the implementing type can be used as a join key in
/// `CudaHashJoin`.
///
/// `CudaHashJoinable` is a trait for which specialized implementations exist
/// for each implementing type (currently i32 and i64). Specialization is
/// necessary because each type requires a different CUDA function to be called.
///
/// An alternative approach would be to specialize the implementation of
/// `CudaHashJoin` methods for each type. However, this would also require a
/// default implementation for all non-implemented types that throws an
/// exception. The benefit would be less code, but currently Rust stable doesn't
/// support [impl specializations with default implementations](https://github.com/rust-lang/rfcs/blob/master/text/1210-impl-specialization.md).
/// [Rust issue #31844](https://github.com/rust-lang/rust/issues/31844) tracks
/// the RFC.
pub trait CudaHashJoinable: DeviceCopy + NullKey {
    /// Implements `CudaHashJoin::build` for the implementing type.
    fn build_impl(
        hj: &CudaHashJoin<Self>,
        join_attr: LaunchableSlice<Self>,
        payload_attr: LaunchableSlice<Self>,
        stream: &Stream,
    ) -> Result<()>;

    /// Implements `CudaHashJoin::probe_sum` for the implementing type.
    fn probe_sum_impl(
        hj: &CudaHashJoin<Self>,
        join_attr: LaunchableSlice<Self>,
        payload_attr: LaunchableSlice<Self>,
        result_set: &Mem<u64>,
        stream: &Stream,
    ) -> Result<()>;
}

/// Specifies that the implementing type can be used as a join key in
/// `CpuHashJoin`.
///
/// See `CudaHashJoinable` for more details on the design decision.
pub trait CpuHashJoinable: DeviceCopy + NullKey {
    /// Implements `CpuHashJoin::build` for the implementing type.
    fn build_impl(
        hj: &mut CpuHashJoin<Self>,
        join_attr: &[Self],
        payload_attr: &[Self],
    ) -> Result<()>;

    /// Implements `CpuHashJoin::probe_sum` for the implementing type.
    fn probe_sum_impl(
        hj: &mut CpuHashJoin<Self>,
        join_attr: &[Self],
        payload_attr: &[Self],
        join_result: &mut u64,
    ) -> Result<()>;
}

/// GPU hash join implemented in CUDA.
///
/// See the module documentation above for usage details.
///
/// The `build` and `probe_sum` methods are simply wrappers for the
/// corresponding implementations in `CudaHashJoinable`. The wrapping is
/// necessary due to the specialization for each type `T`. See the documentation
/// of `CudaHashJoinable` for details.
#[derive(Debug)]
pub struct CudaHashJoin<T: DeviceCopy + NullKey> {
    ops: Module,
    hashing_scheme: HashingScheme,
    hash_table: Arc<HashTable<T>>,
    build_dim: (GridSize, BlockSize),
    probe_dim: (GridSize, BlockSize),
}

/// CPU hash join implemented in C++.
///
/// See the module documentation above for details.
///
/// The `build` and `probe_sum` methods are simply wrappers for the
/// corresponding implementations in `CpuHashJoinable`. The wrapping is
/// necessary due to the specialization for each type `T`. See the documentation
/// of `CpuHashJoinable` for details.
#[derive(Debug)]
pub struct CpuHashJoin<T: DeviceCopy + NullKey> {
    hashing_scheme: HashingScheme,
    hash_table: Arc<HashTable<T>>,
}

/// Hash table for `CpuHashJoin` and `CudaHashJoin`.
#[derive(Debug)]
pub struct HashTable<T: DeviceCopy + NullKey> {
    mem: Mem<T>,
    size: usize,
}

/// Build a `CudaHashJoin`.
#[derive(Clone, Debug)]
pub struct CudaHashJoinBuilder<T: DeviceCopy + NullKey> {
    hashing_scheme: HashingScheme,
    hash_table_i: Option<Arc<HashTable<T>>>,
    build_dim_i: (GridSize, BlockSize),
    probe_dim_i: (GridSize, BlockSize),
}

/// Build a `CpuHashJoin`.
#[derive(Clone, Debug)]
pub struct CpuHashJoinBuilder<T: DeviceCopy + NullKey> {
    hashing_scheme: HashingScheme,
    hash_table_i: Option<Arc<HashTable<T>>>,
}

impl<T> CudaHashJoin<T>
where
    T: DeviceCopy + NullKey + CudaHashJoinable,
{
    /// Build a hash table on the GPU.
    pub fn build(
        &self,
        join_attr: LaunchableSlice<T>,
        payload_attr: LaunchableSlice<T>,
        stream: &Stream,
    ) -> Result<()> {
        T::build_impl(self, join_attr, payload_attr, stream)
    }

    /// Probe the hash table on the GPU and sum the payload attribute rows.
    ///
    /// This effectively implements the SQL code:
    /// ```SQL
    /// SELECT SUM(s.payload_attr) FROM r JOIN s ON r.join_attr = s.join_attr
    /// ```
    pub fn probe_sum(
        &self,
        join_attr: LaunchableSlice<T>,
        payload_attr: LaunchableSlice<T>,
        result_set: &Mem<u64>,
        stream: &Stream,
    ) -> Result<()> {
        T::probe_sum_impl(self, join_attr, payload_attr, result_set, stream)
    }
}

impl<T> CpuHashJoin<T>
where
    T: DeviceCopy + NullKey + CpuHashJoinable,
{
    /// Build a hash table on the CPU.
    pub fn build(&mut self, join_attr: &[T], payload_attr: &[T]) -> Result<()> {
        T::build_impl(self, join_attr, payload_attr)
    }

    /// Probe the hash table on the CPU and sum the payload attribute rows.
    ///
    /// This effectively implements the SQL code:
    /// ```SQL
    /// SELECT SUM(s.payload_attr) FROM r JOIN s ON r.join_attr = s.join_attr
    /// ```
    pub fn probe_sum(
        &mut self,
        join_attr: &[T],
        payload_attr: &[T],
        join_result: &mut u64,
    ) -> Result<()> {
        T::probe_sum_impl(self, join_attr, payload_attr, join_result)
    }
}

/// A Rust macro for specializing the implementation of a join key type. Each
/// type calls a different CUDA function. The function to be called is specified
/// by the `Suffix` parameter.
macro_rules! impl_cuda_hash_join_for_type {
    ($Type:ty, $Suffix:expr) => {
        impl CudaHashJoinable for $Type {
            paste::item!{
                fn build_impl(
                    hj: &CudaHashJoin<$Type>,
                    join_attr: LaunchableSlice<$Type>,
                    payload_attr: LaunchableSlice<$Type>,
                    stream: &Stream,
                    ) -> Result<()> {

                    if join_attr.len() != payload_attr.len() {
                        Err(ErrorKind::InvalidArgument("Join and payload attributes have different sizes".to_string()))?;
                    }
                    if join_attr.len() > hj.hash_table.mem.len() {
                        Err(ErrorKind::InvalidArgument("Hash table is too small for the build data".to_string()))?;
                    }

                    let (grid, block) = hj.build_dim.clone();

                    let join_attr_len = join_attr.len() as u64;
                    let hash_table_size = hj.hash_table.size as u64;
                    let module = &hj.ops;

                    match &hj.hashing_scheme {
                        HashingScheme::Perfect => unsafe{ launch!(
                                module.[<gpu_ht_build_perfect_ $Suffix>]<<<grid, block, 0, stream>>>(
                                    hj.hash_table.mem.as_launchable_ptr(),
                                    hash_table_size,
                                    join_attr.as_launchable_ptr(),
                                    payload_attr.as_launchable_ptr(),
                                    join_attr_len
                                    )
                                )? },
                        HashingScheme::LinearProbing => unsafe { launch!(
                                module.[<gpu_ht_build_linearprobing_ $Suffix>]<<<grid, block, 0, stream>>>(
                                    hj.hash_table.mem.as_launchable_ptr(),
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
                fn probe_sum_impl(
                    hj: &CudaHashJoin<$Type>,
                    join_attr: LaunchableSlice<$Type>,
                    payload_attr: LaunchableSlice<$Type>,
                    result_set: &Mem<u64>,
                    stream: &Stream,
                    ) -> Result<()> {

                    let (grid, block) = hj.probe_dim.clone();

                    if
                       result_set.len() < (grid.x * block.x) as usize {
                       Err(ErrorKind::InvalidArgument("Result set size is too small, must be at least grid * block size".to_string()))?;
                        }

                        if join_attr.len() != payload_attr.len() {
                        Err(ErrorKind::InvalidArgument("Join and payload attributes have different sizes".to_string()))?;
                        }

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
                                    result_set.as_launchable_ptr()
                                    )
                                )? },
                        HashingScheme::LinearProbing => unsafe { launch!(
                                module.[<gpu_ht_probe_aggregate_linearprobing_ $Suffix>]<<<grid, block, 0, stream>>>(
                                    hj.hash_table.mem.as_launchable_ptr(),
                                    hash_table_size,
                                    join_attr.as_launchable_ptr(),
                                    payload_attr.as_launchable_ptr(),
                                    join_attr_len,
                                    result_set.as_launchable_ptr()
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

/// A Rust macro for specializing the implementation of a join key type. Each
/// type calls a different C++ function. The function to be called is specified
/// by the `Suffix` parameter.
macro_rules! impl_cpu_hash_join_for_type {
    ($Type:ty, $Suffix:expr) => {
        impl CpuHashJoinable for $Type {
            paste::item!{
                fn build_impl(hj: &mut CpuHashJoin<$Type>, join_attr: &[$Type], payload_attr: &[$Type]) -> Result<()> {
                        if join_attr.len() != payload_attr.len() {
                        Err(ErrorKind::InvalidArgument("Join and payload attributes have different sizes".to_string()))?;
                        }
                    if
                        join_attr.len() > hj.hash_table.mem.len() {
                       Err(ErrorKind::InvalidArgument("Hash table is too small for the build data".to_string()))?;
                        }

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
                fn probe_sum_impl(
                    hj: &mut CpuHashJoin<$Type>,
                    join_attr: &[$Type],
                    payload_attr: &[$Type],
                    join_result: &mut u64,
                    ) -> Result<()> {
                        if join_attr.len() != payload_attr.len() {
                       Err(ErrorKind::InvalidArgument("Join and payload attributes have different sizes".to_string()))?;
                        }

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
                                join_result,
                                )
                        },
                        HashingScheme::LinearProbing => unsafe {
                            [<cpu_ht_probe_aggregate_linearprobing_ $Suffix>](
                                hj.hash_table.mem.as_ptr(),
                                hash_table_size,
                                join_attr.as_ptr(),
                                payload_attr.as_ptr(),
                                join_attr_len,
                                join_result,
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
    /// Create a new CPU hash table.
    ///
    /// The hash table can be used on CPUs. In the case of NVLink 2.0 on POWER9,
    /// it can also be used on GPUs.
    pub fn new_on_cpu(mut mem: DerefMem<T>, size: usize) -> Result<Self> {
        if mem.len() < size {
            Err(ErrorKind::InvalidArgument(
                "Provided memory must be larger than hash table size".to_string(),
            ))?;
        }

        mem.iter_mut().by_ref().for_each(|x| *x = T::null_key());

        Ok(Self {
            mem: mem.into(),
            size,
        })
    }

    /// Create a new GPU hash table.
    ///
    /// The hash table can be used on GPUs. It cannot always be used on CPUs,
    /// due to the possibility of using GPU device memory. This also holds true
    /// for NVLink 2.0 on POWER9.
    pub fn new_on_gpu(mut mem: Mem<T>, size: usize) -> Result<Self> {
        if mem.len() < size {
            Err(ErrorKind::InvalidArgument(
                "Provided memory must be larger than hash table size".to_string(),
            ))?;
        }

        let mem_ptr = mem.as_ptr();
        let mem_len = mem.len();

        // Initialize hash table
        match mem {
            Mem::SysMem(ref mut mem) => mem.iter_mut().by_ref().for_each(|x| *x = T::null_key()),
            Mem::NumaMem(ref mut mem) => mem.iter_mut().by_ref().for_each(|x| *x = T::null_key()),
            Mem::CudaPinnedMem(ref mut mem) => {
                mem.iter_mut().by_ref().for_each(|x| *x = T::null_key())
            }
            Mem::DistributedNumaMem(ref mut mem) => {
                mem.iter_mut().by_ref().for_each(|x| *x = T::null_key())
            }
            _ => {
                unsafe {
                    cuMemsetD32_v2(
                        mem_ptr as *mut c_void as u64,
                        T::null_key().as_(),
                        mem_len
                            .checked_mul(size_of::<T>() / size_of::<c_uint>())
                            .ok_or_else(|| {
                                ErrorKind::IntegerOverflow(
                                    "Failed to compute hash table bytes".to_string(),
                                )
                            })?,
                    )
                }
                .to_result()?;
            }
        }

        Ok(Self { mem, size })
    }

    /// Create a new hash table from another hash table.
    ///
    /// Copies the contents of the source hash table into the new hash table.
    pub fn new_from_hash_table(mut mem: Mem<T>, src: &Self) -> Result<Self> {
        mem.copy_from_mem(&src.mem)?;

        Ok(Self {
            mem,
            size: src.size,
        })
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

    pub fn hash_table(mut self, ht: Arc<HashTable<T>>) -> Self {
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

    pub fn build(&self) -> Result<CudaHashJoin<T>> {
        if self.hash_table_i.is_none() {
            Err(ErrorKind::InvalidArgument("Hash table not set".to_string()))?;
        }

        let module_path = CString::new(env!("CUDAUTILS_PATH")).map_err(|_| {
            ErrorKind::NulCharError(
                "Failed to load CUDA module, check your CUDAUTILS_PATH".to_string(),
            )
        })?;

        let ops = Module::load_from_file(&module_path)?;

        let hash_table = if let Some(ht) = self.hash_table_i.clone() {
            ht
        } else {
            Arc::new(HashTable {
                mem: allocator::Allocator::alloc_mem::<T>(
                    allocator::MemType::CudaUniMem,
                    Self::DEFAULT_HT_SIZE,
                ),
                size: Self::DEFAULT_HT_SIZE,
            })
        };

        Ok(CudaHashJoin {
            ops,
            hashing_scheme: self.hashing_scheme,
            hash_table,
            build_dim: self.build_dim_i.clone(),
            probe_dim: self.probe_dim_i.clone(),
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

#[cfg(test)]
mod tests {
    use super::{CpuHashJoinBuilder, CudaHashJoinBuilder, HashTable, HashingScheme};
    use datagen::relation::UniformRelation;
    use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemType};
    use numa_gpu::runtime::memory::Mem;
    use rustacuda::stream::{Stream, StreamFlags};
    use std::convert::TryInto;
    use std::error::Error;
    use std::result::Result;
    use std::sync::Arc;

    macro_rules! test_cpu_seq {
        ($name:ident, $mem_type:expr, $scheme:expr, $type:ty) => {
            #[test]
            fn $name() -> Result<(), Box<dyn Error>> {
                const ROWS: usize = (32 << 20) / std::mem::size_of::<$type>();
                const HT_LEN: usize = 4 * ROWS;
                let alloc_fn = Allocator::deref_mem_alloc_fn::<$type>($mem_type);

                let mut inner_rel_key = alloc_fn(ROWS);
                let mut inner_rel_pay = alloc_fn(ROWS);
                let mut outer_rel_key = alloc_fn(ROWS);
                let mut outer_rel_pay = alloc_fn(ROWS);

                UniformRelation::gen_primary_key(&mut inner_rel_key)?;
                UniformRelation::gen_foreign_key_from_primary_key(
                    &mut outer_rel_key,
                    &inner_rel_key,
                );

                inner_rel_pay
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, x)| *x = (i + 1) as $type);
                outer_rel_pay
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, x)| *x = (i + 1) as $type);

                let ht_mem = alloc_fn(HT_LEN);

                let hash_table = HashTable::new_on_cpu(ht_mem, HT_LEN)?;

                let mut hj_op = CpuHashJoinBuilder::default()
                    .hashing_scheme($scheme)
                    .hash_table(Arc::new(hash_table))
                    .build();

                hj_op.build(&inner_rel_key, &inner_rel_pay)?;
                let mut result_sum: u64 = 0;
                hj_op.probe_sum(&outer_rel_key, &outer_rel_pay, &mut result_sum)?;

                assert_eq!((ROWS as u64 * (ROWS as u64 + 1)) / 2, result_sum);

                Ok(())
            }
        };
    }

    test_cpu_seq!(
        cpu_seq_sysmem_perfect_i32,
        DerefMemType::SysMem,
        HashingScheme::Perfect,
        i32
    );
    test_cpu_seq!(
        cpu_seq_sysmem_perfect_i64,
        DerefMemType::SysMem,
        HashingScheme::Perfect,
        i64
    );
    test_cpu_seq!(
        cpu_seq_sysmem_linearprobing_i32,
        DerefMemType::SysMem,
        HashingScheme::LinearProbing,
        i32
    );
    test_cpu_seq!(
        cpu_seq_sysmem_linearprobing_i64,
        DerefMemType::SysMem,
        HashingScheme::LinearProbing,
        i64
    );

    macro_rules! test_cuda {
        ($name:ident, $mem_type:expr, $scheme:expr, $type:ty) => {
            #[test]
            fn $name() -> Result<(), Box<dyn Error>> {
                const GRID_SIZE: u32 = 16;
                const BLOCK_SIZE: u32 = 1024;
                const ROWS: usize = (32 << 20) / std::mem::size_of::<$type>();
                const HT_LEN: usize = 4 * ROWS;

                let _ctx = rustacuda::quick_init()?;
                let alloc_fn = Allocator::deref_mem_alloc_fn::<$type>($mem_type);

                let mut inner_rel_key = alloc_fn(ROWS);
                let mut inner_rel_pay = alloc_fn(ROWS);
                let mut outer_rel_key = alloc_fn(ROWS);
                let mut outer_rel_pay = alloc_fn(ROWS);

                UniformRelation::gen_primary_key(&mut inner_rel_key)?;
                UniformRelation::gen_foreign_key_from_primary_key(
                    &mut outer_rel_key,
                    &inner_rel_key,
                );

                inner_rel_pay
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, x)| *x = (i + 1) as $type);
                outer_rel_pay
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, x)| *x = (i + 1) as $type);

                let ht_mem = Allocator::alloc_mem(MemType::CudaDevMem, HT_LEN);
                let hash_table = HashTable::new_on_gpu(ht_mem, HT_LEN)?;

                let mut result_sum_per_thread =
                    Allocator::alloc_mem(MemType::CudaUniMem, (GRID_SIZE * BLOCK_SIZE) as usize);

                let hj_op = CudaHashJoinBuilder::default()
                    .hashing_scheme($scheme)
                    .hash_table(Arc::new(hash_table))
                    .build_dim(GRID_SIZE.into(), BLOCK_SIZE.into())
                    .probe_dim(GRID_SIZE.into(), BLOCK_SIZE.into())
                    .build()?;

                let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
                hj_op.build(
                    Mem::from(inner_rel_key).as_launchable_slice(),
                    Mem::from(inner_rel_pay).as_launchable_slice(),
                    &stream,
                )?;
                hj_op.probe_sum(
                    Mem::from(outer_rel_key).as_launchable_slice(),
                    Mem::from(outer_rel_pay).as_launchable_slice(),
                    &mut result_sum_per_thread,
                    &stream,
                )?;
                stream.synchronize()?;

                let result_sum_slice: &[u64] = (&result_sum_per_thread)
                    .try_into()
                    .map_err(|(err, _)| err)?;
                let result_sum = result_sum_slice.iter().sum();

                assert_eq!((ROWS as u64 * (ROWS as u64 + 1)) / 2, result_sum);

                Ok(())
            }
        };
    }

    test_cuda!(
        cuda_pinnedmem_perfect_i32,
        DerefMemType::CudaPinnedMem,
        HashingScheme::Perfect,
        i32
    );
    test_cuda!(
        cuda_pinnedmem_perfect_i64,
        DerefMemType::CudaPinnedMem,
        HashingScheme::Perfect,
        i64
    );
    test_cuda!(
        cuda_pinnedmem_linearprobing_i32,
        DerefMemType::CudaPinnedMem,
        HashingScheme::LinearProbing,
        i32
    );
    test_cuda!(
        cuda_pinnedmem_linearprobing_i64,
        DerefMemType::CudaPinnedMem,
        HashingScheme::LinearProbing,
        i64
    );
    test_cuda!(
        cuda_unimem,
        DerefMemType::CudaUniMem,
        HashingScheme::Perfect,
        i32
    );
}
