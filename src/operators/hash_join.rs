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

use self::accel::device::sync;
use self::accel::kernel::{Block, Grid};
use self::accel::module::Module;
use self::accel::uvec::UVec;

use std::path::Path;

use error::Result;

#[derive(Debug)]
pub struct CudaHashJoin {
    ops: Module,
    hash_table: HashTable,
    result_set: UVec<u64>,
    build_result: UVec<u64>,
    build_dim: (u32, u32),
    probe_dim: (u32, u32),
}

#[derive(Debug)]
pub struct HashTable {
    data: UVec<i64>,
}

#[derive(Debug)]
pub struct CudaHashJoinBuilder {
    hash_table_i: Option<HashTable>,
    result_set_i: Option<UVec<u64>>,
    build_dim_i: (u32, u32),
    probe_dim_i: (u32, u32),
}

impl CudaHashJoin {
    pub fn build(&mut self, join_attr: UVec<i64>, filter_attr: UVec<i64>) -> Result<&mut Self> {
        ensure!(
            join_attr.len() == filter_attr.len(),
            "Join and filter attributes have different sizes"
        );

        let (grid, block) = self.build_dim;

        let join_attr_len = join_attr.len() as u64;
        let hash_table_len = self.hash_table.data.len() as u64;

        cuda!(
            build_pipeline_kernel << [&self.ops, Grid::x(grid), Block::x(block)]
                >> (
                    join_attr_len,
                    self.build_result,
                    filter_attr,
                    join_attr,
                    hash_table_len,
                    self.hash_table.data
                )
        )?;

        Ok(self)
    }

    pub fn probe(
        &mut self,
        join_attr: UVec<i64>,
        filter_attr: UVec<i64>,
    ) -> Result<&mut UVec<u64>> {
        let (grid, block) = self.probe_dim;
        ensure!(
            self.result_set.len() >= (grid * block) as usize,
            "Result set size is too small, must be at least grid * block size"
        );
        ensure!(
            join_attr.len() == filter_attr.len(),
            "Join and filter attributes have different sizes"
        );

        let join_attr_len = join_attr.len() as u64;
        let hash_table_len = self.hash_table.data.len() as u64;

        cuda!(
            aggregation_kernel << [&self.ops, Grid::x(grid), Block::x(block)]
                >> (
                    join_attr_len,
                    filter_attr,
                    join_attr,
                    self.hash_table.data,
                    hash_table_len,
                    self.result_set
                )
        )?;

        Ok(&mut self.result_set)
    }
}

impl HashTable {
    const NULL_KEY: i64 = -1;

    pub fn new(size: usize) -> Result<Self> {
        ensure!(
            size.is_power_of_two(),
            "Hash table size must be a power of two"
        );

        let mut data = UVec::<i64>::new(size)?;

        // Initialize hash table
        data.as_slice_mut()
            .iter_mut()
            .by_ref()
            .map(|entry| *entry = Self::NULL_KEY)
            .collect::<()>();

        Ok(Self { data })
    }
}

impl CudaHashJoinBuilder {
    const DEFAULT_HT_SIZE: usize = 1024;

    pub fn default() -> Self {
        Self {
            hash_table_i: None,
            result_set_i: None,
            build_dim_i: (1, 1),
            probe_dim_i: (1, 1),
        }
    }

    pub fn hash_table(mut self, ht: HashTable) -> Self {
        self.hash_table_i = Some(ht);
        self
    }

    pub fn result_set(mut self, rs: UVec<u64>) -> Self {
        self.result_set_i = Some(rs);
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
        ensure!(self.result_set_i.is_some(), "Result set array not set");

        let module_path = Path::new(env!("CUDAUTILS_PATH"));

        // force CUDA to init device and create context
        sync()?;

        let ops = Module::load_file(module_path)?;

        let (build_grid, build_block) = self.build_dim_i;
        let build_result_size = build_grid as usize * build_block as usize;

        let (probe_grid, probe_block) = self.probe_dim_i;
        let result_set_size = probe_grid as usize * probe_block as usize;

        let build_result: UVec<u64> = UVec::new(build_result_size)?;

        let hash_table = if let Some(ht) = self.hash_table_i {
            ht
        } else {
            HashTable {
                data: UVec::<i64>::new(Self::DEFAULT_HT_SIZE)?,
            }
        };

        let result_set = if let Some(rs) = self.result_set_i {
            rs
        } else {
            UVec::<u64>::new(result_set_size)?
        };

        Ok(CudaHashJoin {
            ops,
            hash_table,
            result_set,
            build_result,
            build_dim: self.build_dim_i,
            probe_dim: self.probe_dim_i,
        })
    }
}

impl ::std::fmt::Display for HashTable {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "[")?;
        for entry in self.data.as_slice().iter() {
            write!(f, "{},", entry)?;
        }
        write!(f, "]")?;

        Ok(())
    }
}
