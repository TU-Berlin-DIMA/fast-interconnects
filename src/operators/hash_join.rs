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
extern crate rustacuda;

use cuda_sys::cuda::cuMemsetD32_v2;

use self::rustacuda::memory::UnifiedBuffer;
use self::rustacuda::prelude::*;
use self::rustacuda::function::{BlockSize, GridSize};

use std::ffi::CString;
use std::mem::size_of;
use std::os::raw::{c_uint, c_void};
use std::sync::Arc;

use crate::error::{Error, ErrorKind, Result, ToResult};
use crate::runtime::memory::*;

extern "C" {
    fn cpu_ht_build_linearprobing(
        hash_table: *mut i64, // FIXME: replace i64 with atomic_i64
        hash_table_entries: u64,
        join_attr_data: *const i64,
        payload_attr_data: *const i64,
        data_length: u64,
    );

    fn cpu_ht_probe_aggregate_linearprobing(
        hash_table: *const i64, // FIXME: replace i64 with atomic_i64
        hash_table_entries: u64,
        join_attr_data: *const i64,
        payload_attr_data: *const i64,
        data_length: u64,
        aggregation_result: *mut u64,
    );

    fn cpu_ht_build_perfect(
        hash_table: *mut i64, // FIXME: replace i64 with atomic_i64
        hash_table_entries: u64,
        join_attr_data: *const i64,
        payload_attr_data: *const i64,
        data_length: u64,
    );

    fn cpu_ht_probe_aggregate_perfect(
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

#[derive(Debug)]
pub struct CudaHashJoin {
    ops: Module,
    hashing_scheme: HashingScheme,
    hash_table: HashTable,
    build_dim: (GridSize, BlockSize),
    probe_dim: (GridSize, BlockSize),
}

#[derive(Debug)]
pub struct CpuHashJoin {
    hashing_scheme: HashingScheme,
    hash_table: Arc<HashTable>,
}

#[derive(Debug)]
pub struct HashTable {
    // FIXME: replace i64 with atomic_i64 when the type is added to Rust stable
    mem: Mem<i64>,
    size: usize,
}

#[derive(Debug)]
pub struct CudaHashJoinBuilder {
    hashing_scheme: HashingScheme,
    hash_table_i: Option<HashTable>,
    build_dim_i: (GridSize, BlockSize),
    probe_dim_i: (GridSize, BlockSize),
}

#[derive(Debug)]
pub struct CpuHashJoinBuilder {
    hashing_scheme: HashingScheme,
    hash_table_i: Option<Arc<HashTable>>,
}

impl CudaHashJoin {
    pub fn build(
        &mut self,
        join_attr: &Mem<i64>,
        payload_attr: &Mem<i64>,
        stream: &Stream,
    ) -> Result<&mut Self> {
        ensure!(
            join_attr.len() == payload_attr.len(),
            "Join and payload attributes have different sizes"
        );
        ensure!(
            join_attr.len() <= self.hash_table.mem.len(),
            "Hash table is too small for the build data"
        );

        let (grid, block) = self.build_dim.clone();

        let join_attr_len = join_attr.len() as u64;
        let hash_table_size = self.hash_table.size as u64;
        let module = &self.ops;

        match &self.hashing_scheme {
            HashingScheme::Perfect => unsafe{ launch!(
                module.gpu_ht_build_perfect<<<grid, block, 0, stream>>>(
                        *self.hash_table.mem.as_mut_ptr(),
                        hash_table_size,
                        *join_attr.as_ptr(),
                        *payload_attr.as_ptr(),
                        join_attr_len
                    )
            )? },
            HashingScheme::LinearProbing => unsafe { launch!(
                module.gpu_ht_build_linearprobing<<<grid, block, 0, stream>>>(
                        *self.hash_table.mem.as_mut_ptr(),
                        hash_table_size,
                        *join_attr.as_ptr(),
                        *payload_attr.as_ptr(),
                        join_attr_len
                    )
            )? },
        };

        Ok(self)
    }

    pub fn probe_count(
        &mut self,
        join_attr: &Mem<i64>,
        payload_attr: &Mem<i64>,
        result_set: &mut Mem<u64>,
        stream: &Stream,
    ) -> Result<()> {
        let (grid, block) = self.probe_dim.clone();
        ensure!(
            result_set.len() >= (grid.x * block.x) as usize,
            "Result set size is too small, must be at least grid * block size"
        );
        ensure!(
            join_attr.len() == payload_attr.len(),
            "Join and payload attributes have different sizes"
        );

        let join_attr_len = join_attr.len() as u64;
        let hash_table_size = self.hash_table.size as u64;
        let module = &self.ops;

        match &self.hashing_scheme {
            HashingScheme::Perfect => unsafe { launch!(
                module.gpu_ht_probe_aggregate_perfect<<<grid, block, 0, stream>>>(
                        *self.hash_table.mem.as_mut_ptr(),
                        hash_table_size,
                        *join_attr.as_ptr(),
                        *payload_attr.as_ptr(),
                        join_attr_len,
                        *result_set.as_mut_ptr()
                    )
            )? },
            HashingScheme::LinearProbing => unsafe { launch!(
                module.gpu_ht_probe_aggregate_linearprobing<<<grid, block, 0, stream>>>(
                        *self.hash_table.mem.as_mut_ptr(),
                        hash_table_size,
                        *join_attr.as_ptr(),
                        *payload_attr.as_ptr(),
                        join_attr_len,
                        *result_set.as_mut_ptr()
                    )
            )? },
        };

        Ok(())
    }
}

impl CpuHashJoin {
    pub fn build(&mut self, join_attr: &[i64], payload_attr: &[i64]) -> Result<&mut Self> {
        ensure!(
            join_attr.len() == payload_attr.len(),
            "Join and payload attributes have different sizes"
        );
        ensure!(
            join_attr.len() <= self.hash_table.mem.len(),
            "Hash table is too small for the build data"
        );

        let join_attr_len = join_attr.len() as u64;
        let hash_table_size = self.hash_table.size as u64;

        match &self.hashing_scheme {
            HashingScheme::Perfect => unsafe {
                cpu_ht_build_perfect(
                    self.hash_table.mem.as_ptr() as *mut i64,
                    hash_table_size,
                    join_attr.as_ptr(),
                    payload_attr.as_ptr(),
                    join_attr_len,
                )
            },
            HashingScheme::LinearProbing => unsafe {
                cpu_ht_build_linearprobing(
                    self.hash_table.mem.as_ptr() as *mut i64,
                    hash_table_size,
                    join_attr.as_ptr(),
                    payload_attr.as_ptr(),
                    join_attr_len,
                )
            },
        };

        Ok(self)
    }

    pub fn probe_count(
        &mut self,
        join_attr: &[i64],
        payload_attr: &[i64],
        join_count: &mut u64,
    ) -> Result<()> {
        ensure!(
            join_attr.len() == payload_attr.len(),
            "Join and payload attributes have different sizes"
        );

        let join_attr_len = join_attr.len() as u64;
        let hash_table_size = self.hash_table.size as u64;

        match &self.hashing_scheme {
            HashingScheme::Perfect => unsafe {
                cpu_ht_probe_aggregate_perfect(
                    self.hash_table.mem.as_ptr(),
                    hash_table_size,
                    join_attr.as_ptr(),
                    payload_attr.as_ptr(),
                    join_attr_len,
                    join_count,
                )
            },
            HashingScheme::LinearProbing => unsafe {
                cpu_ht_probe_aggregate_linearprobing(
                    self.hash_table.mem.as_ptr(),
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

impl HashTable {
    const NULL_KEY: i64 = -1;

    pub fn new_on_cpu(mut mem: DerefMem<i64>, size: usize) -> Result<Self> {
        ensure!(
            size.is_power_of_two(),
            "Hash table size must be a power of two"
        );
        ensure!(
            mem.len() >= size,
            "Provided memory must be larger than hash table size"
        );

        mem.iter_mut().by_ref().for_each(|x| *x = Self::NULL_KEY);

        Ok(Self {
            mem: mem.into(),
            size,
        })
    }

    pub fn new_on_gpu(mem: Mem<i64>, size: usize) -> Result<Self> {
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
                Self::NULL_KEY as c_uint,
                mem.len() * (size_of::<i64>() / size_of::<c_uint>()),
            )
        }
        .to_result()?;

        Ok(Self { mem, size })
    }
}

impl ::std::default::Default for CudaHashJoinBuilder {
    fn default() -> Self {
        Self {
            hashing_scheme: HashingScheme::default(),
            hash_table_i: None,
            build_dim_i: (1.into(), 1.into()),
            probe_dim_i: (1.into(), 1.into()),
        }
    }
}

impl CudaHashJoinBuilder {
    const DEFAULT_HT_SIZE: usize = 1024;

    pub fn hashing_scheme(mut self, hashing_scheme: HashingScheme) -> Self {
        self.hashing_scheme = hashing_scheme;
        self
    }

    pub fn hash_table(mut self, ht: HashTable) -> Self {
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

    pub fn build(self) -> Result<CudaHashJoin> {
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
                mem: CudaUniMem(UnifiedBuffer::<i64>::new(
                    &HashTable::NULL_KEY,
                    Self::DEFAULT_HT_SIZE,
                )?),
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

impl ::std::default::Default for CpuHashJoinBuilder {
    fn default() -> Self {
        Self {
            hashing_scheme: HashingScheme::default(),
            hash_table_i: None,
        }
    }
}

impl CpuHashJoinBuilder {
    const DEFAULT_HT_SIZE: usize = 1024;

    pub fn hashing_scheme(mut self, hashing_scheme: HashingScheme) -> Self {
        self.hashing_scheme = hashing_scheme;
        self
    }

    pub fn hash_table(mut self, hash_table: Arc<HashTable>) -> Self {
        self.hash_table_i = Some(hash_table);
        self
    }

    pub fn build(&self) -> CpuHashJoin {
        let hash_table = match &self.hash_table_i {
            Some(ht) => ht.clone(),
            None => Arc::new(HashTable {
                mem: SysMem(Vec::<i64>::with_capacity(Self::DEFAULT_HT_SIZE)),
                size: Self::DEFAULT_HT_SIZE,
            }),
        };

        CpuHashJoin {
            hashing_scheme: self.hashing_scheme,
            hash_table,
        }
    }
}

impl ::std::fmt::Display for HashTable {
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
