/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019-2021, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use crate::error::{ErrorKind, Result};
use csv::{ByteRecord, ReaderBuilder};
use flate2::read::GzDecoder;
use numa_gpu::runtime::allocator::{self, DerefMemType};
use numa_gpu::runtime::memory::*;
use rustacuda::memory::DeviceCopy;
use serde::de::DeserializeOwned;
use std::collections::vec_deque::VecDeque;
use std::fs::File;
use std::io::Read;
use std::time::{Duration, Instant};

pub type JoinDataGenFn<T> = Box<dyn FnMut(&mut [T], &mut [T], &mut [T], &mut [T]) -> Result<()>>;

pub struct JoinData<T: DeviceCopy> {
    pub build_relation_key: Mem<T>,
    pub build_relation_payload: Mem<T>,
    pub probe_relation_key: Mem<T>,
    pub probe_relation_payload: Mem<T>,
}

pub struct JoinDataBuilder {
    inner_len: usize,
    outer_len: usize,
    inner_mem_type: DerefMemType,
    outer_mem_type: DerefMemType,
    do_mlock: bool,
}

impl Default for JoinDataBuilder {
    fn default() -> JoinDataBuilder {
        JoinDataBuilder {
            inner_len: 1,
            outer_len: 1,
            inner_mem_type: DerefMemType::SysMem,
            outer_mem_type: DerefMemType::SysMem,
            do_mlock: false,
        }
    }
}

impl JoinDataBuilder {
    pub fn inner_len(&mut self, inner_len: usize) -> &mut Self {
        self.inner_len = inner_len;
        self
    }

    pub fn outer_len(&mut self, outer_len: usize) -> &mut Self {
        self.outer_len = outer_len;
        self
    }

    pub fn inner_mem_type(&mut self, inner_mem_type: DerefMemType) -> &mut Self {
        self.inner_mem_type = inner_mem_type;
        self
    }

    pub fn outer_mem_type(&mut self, outer_mem_type: DerefMemType) -> &mut Self {
        self.outer_mem_type = outer_mem_type;
        self
    }

    pub fn mlock(&mut self, do_mlock: bool) -> &mut Self {
        self.do_mlock = do_mlock;
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
            (self.inner_len, self.inner_mem_type.clone()),
            (self.inner_len, self.inner_mem_type.clone()),
            (self.outer_len, self.outer_mem_type.clone()),
            (self.outer_len, self.outer_mem_type.clone()),
        ]
        .iter()
        .cloned()
        .map(|(len, mem_type)| {
            let mut mem = allocator::Allocator::alloc_deref_mem(mem_type, len);

            // Force the OS to physically allocate the memory
            if self.do_mlock {
                mem.mlock()?;
            }

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
        mut data_gen_fn: JoinDataGenFn<T>,
    ) -> Result<(JoinData<T>, Duration, Duration)>
    where
        T: Copy + Default + DeviceCopy,
    {
        let (mut inner_key, mut inner_payload, mut outer_key, mut outer_payload, malloc_time) =
            self.allocate_relations()?;

        // Generate dataset
        let gen_timer = Instant::now();
        data_gen_fn(
            inner_key.as_mut_slice(),
            inner_payload.as_mut_slice(),
            outer_key.as_mut_slice(),
            outer_payload.as_mut_slice(),
        )?;
        let gen_time = gen_timer.elapsed();

        Ok((
            JoinData {
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
    ) -> Result<(JoinData<T>, Duration, Duration)>
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
            JoinData {
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
