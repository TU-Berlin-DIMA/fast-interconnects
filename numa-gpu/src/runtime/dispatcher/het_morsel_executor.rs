/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use crate::error::Result;
use crate::runtime::cpu_affinity::CpuAffinity;
use rayon::{ThreadPool, ThreadPoolBuilder};
use rustacuda::context::CurrentContext;
use std::sync::Arc;

pub struct HetMorselExecutor {
    pub(super) morsel_len: usize,
    pub(super) cpu_workers: usize,
    pub(super) gpu_workers: usize,
    pub(super) cpu_thread_pool: ThreadPool,
    pub(super) gpu_thread_pool: ThreadPool,
}

pub struct HetMorselExecutorBuilder {
    morsel_len: usize,
    cpu_threads: usize,
    cpu_affinity: Arc<CpuAffinity>,
    gpu_ids: Vec<u16>,
}

impl HetMorselExecutorBuilder {
    pub fn new() -> Self {
        let morsel_len = 10_000;

        Self {
            morsel_len,
            cpu_threads: 1,
            cpu_affinity: Arc::new(CpuAffinity::default()),
            gpu_ids: Vec::new(),
        }
    }

    pub fn morsel_len(mut self, morsel_len: usize) -> Self {
        self.morsel_len = morsel_len;
        self
    }

    pub fn cpu_threads(mut self, threads: usize) -> Self {
        self.cpu_threads = threads;
        self
    }

    pub fn cpu_affinity(mut self, cpu_affinity: CpuAffinity) -> Self {
        self.cpu_affinity = Arc::new(cpu_affinity);
        self
    }

    pub fn gpu_ids(mut self, gpu_ids: Vec<u16>) -> Self {
        self.gpu_ids = gpu_ids;
        self
    }

    pub fn build(self) -> Result<HetMorselExecutor> {
        let cpu_workers = self.cpu_threads;
        let gpu_workers = self.gpu_ids.len();
        let cpu_affinity = self.cpu_affinity.clone();

        let unowned_context = CurrentContext::get_current()?;

        let cpu_thread_pool = ThreadPoolBuilder::new()
            .num_threads(cpu_workers)
            .start_handler(move |tid| {
                cpu_affinity
                    .clone()
                    .set_affinity(tid as u16)
                    .expect("Couldn't set CPU core affinity");
            })
            .build()?;

        let gpu_thread_pool = ThreadPoolBuilder::new()
            .num_threads(gpu_workers)
            .start_handler(move |_thread_index| {
                CurrentContext::set_current(&unowned_context)
                    .expect("Failed to set CUDA context in GPU worker thread");

                // FIXME: set CPU affinity using nvmlDeviceGetCpuAffinity
            })
            .build()?;

        Ok(HetMorselExecutor {
            morsel_len: self.morsel_len,
            cpu_workers,
            gpu_workers,
            cpu_thread_pool,
            gpu_thread_pool,
        })
    }
}
