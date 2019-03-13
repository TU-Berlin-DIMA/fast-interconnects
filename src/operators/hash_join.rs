/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

extern crate accel;
extern crate cuda_sys;

use self::accel::device::sync;
use self::accel::error::Check;
use self::accel::kernel::{Block, Grid};
use self::accel::module::Module;
use self::accel::uvec::UVec;

use self::cuda_sys::cudart::cudaMemset;

use std::mem::size_of;
use std::os::raw::c_void;
use std::path::Path;
use std::sync::Arc;

use crate::error::Result;
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
}

#[derive(Debug)]
pub struct CudaHashJoin {
    ops: Module,
    hash_table: HashTable,
    build_dim: (u32, u32),
    probe_dim: (u32, u32),
}

#[derive(Debug)]
pub struct CpuHashJoin {
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
    hash_table_i: Option<HashTable>,
    build_dim_i: (u32, u32),
    probe_dim_i: (u32, u32),
}

#[derive(Debug)]
pub struct CpuHashJoinBuilder {
    hash_table_i: Arc<HashTable>,
}

impl CudaHashJoin {
    pub fn build(&mut self, join_attr: &Mem<i64>, payload_attr: &Mem<i64>) -> Result<&mut Self> {
        ensure!(
            join_attr.len() == payload_attr.len(),
            "Join and payload attributes have different sizes"
        );
        ensure!(
            join_attr.len() <= self.hash_table.mem.len(),
            "Hash table is too small for the build data"
        );

        let (grid, block) = self.build_dim;

        let join_attr_len = join_attr.len() as u64;
        let hash_table_size = self.hash_table.size as u64;

        cuda!(
            gpu_ht_build_linearprobing << [&self.ops, Grid::x(grid), Block::x(block)]
                >> (
                    *self.hash_table.mem.as_any(),
                    hash_table_size,
                    *join_attr.as_any(),
                    *payload_attr.as_any(),
                    join_attr_len
                )
        )?;

        Ok(self)
    }

    pub fn probe_count(
        &mut self,
        join_attr: &Mem<i64>,
        payload_attr: &Mem<i64>,
        result_set: &mut Mem<u64>,
    ) -> Result<()> {
        let (grid, block) = self.probe_dim;
        ensure!(
            result_set.len() >= (grid * block) as usize,
            "Result set size is too small, must be at least grid * block size"
        );
        ensure!(
            join_attr.len() == payload_attr.len(),
            "Join and payload attributes have different sizes"
        );

        let join_attr_len = join_attr.len() as u64;
        let hash_table_size = self.hash_table.size as u64;

        cuda!(
            gpu_ht_probe_aggregate_linearprobing << [&self.ops, Grid::x(grid), Block::x(block)]
                >> (
                    *self.hash_table.mem.as_any(),
                    hash_table_size,
                    *join_attr.as_any(),
                    *payload_attr.as_any(),
                    join_attr_len,
                    *result_set.as_any()
                )
        )?;

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

        unsafe {
            cpu_ht_build_linearprobing(
                self.hash_table.mem.as_ptr() as *mut i64,
                hash_table_size,
                join_attr.as_ptr(),
                payload_attr.as_ptr(),
                join_attr_len,
            )
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

        unsafe {
            cpu_ht_probe_aggregate_linearprobing(
                self.hash_table.mem.as_ptr(),
                hash_table_size,
                join_attr.as_ptr(),
                payload_attr.as_ptr(),
                join_attr_len,
                join_count,
            )
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
            cudaMemset(
                mem.as_ptr() as *mut c_void,
                Self::NULL_KEY as i32,
                mem.len() * size_of::<i64>(),
            )
        }
        .check()?;

        Ok(Self { mem, size })
    }
}

impl CudaHashJoinBuilder {
    const DEFAULT_HT_SIZE: usize = 1024;

    pub fn default() -> Self {
        Self {
            hash_table_i: None,
            build_dim_i: (1, 1),
            probe_dim_i: (1, 1),
        }
    }

    pub fn hash_table(mut self, ht: HashTable) -> Self {
        self.hash_table_i = Some(ht);
        self
    }

    pub fn build_dim(mut self, grid: u32, block: u32) -> Self {
        self.build_dim_i = (grid, block);
        self
    }

    pub fn probe_dim(mut self, grid: u32, block: u32) -> Self {
        self.probe_dim_i = (grid, block);
        self
    }

    pub fn build(self) -> Result<CudaHashJoin> {
        ensure!(self.hash_table_i.is_some(), "Hash table not set");

        let module_path = Path::new(env!("CUDAUTILS_PATH"));

        // force CUDA to init device and create context
        sync()?;

        let ops = Module::load_file(module_path)?;

        let hash_table = if let Some(ht) = self.hash_table_i {
            ht
        } else {
            HashTable {
                mem: CudaUniMem(UVec::<i64>::new(Self::DEFAULT_HT_SIZE)?),
                size: Self::DEFAULT_HT_SIZE,
            }
        };

        Ok(CudaHashJoin {
            ops,
            hash_table,
            build_dim: self.build_dim_i,
            probe_dim: self.probe_dim_i,
        })
    }
}

impl CpuHashJoinBuilder {
    pub fn new_with_ht(hash_table: Arc<HashTable>) -> Self {
        Self {
            hash_table_i: hash_table,
        }
    }

    pub fn build(&self) -> CpuHashJoin {
        CpuHashJoin {
            hash_table: self.hash_table_i.clone(),
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
