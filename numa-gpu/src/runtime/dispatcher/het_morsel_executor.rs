// Copyright 2019-2022 Clemens Lutz
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

use crate::error::Result;
use crate::runtime::cpu_affinity::CpuAffinity;
use rayon::{ThreadPool, ThreadPoolBuilder};
use rustacuda::context::CurrentContext;
use rustacuda::error::CudaError;
use rustacuda::stream::{Stream, StreamFlags};
use std::default::Default;
use std::sync::Arc;

#[derive(Clone)]
pub struct MorselSpec {
    pub cpu_morsel_bytes: usize,
    pub gpu_morsel_bytes: usize,
}

#[derive(Clone, Default)]
pub struct WorkerCpuAffinity {
    pub cpu_workers: CpuAffinity,
    pub gpu_workers: CpuAffinity,
}

pub struct HetMorselExecutor {
    pub(super) morsel_spec: MorselSpec,
    pub(super) cpu_workers: usize,
    pub(super) gpu_workers: usize,
    pub(super) cpu_thread_pool: ThreadPool,
    pub(super) gpu_thread_pool: ThreadPool,
    pub(super) streams: Vec<Stream>,
}

pub struct HetMorselExecutorBuilder {
    morsel_spec: MorselSpec,
    cpu_threads: usize,
    cpu_worker_affinity: Arc<CpuAffinity>,
    gpu_worker_affinity: Arc<CpuAffinity>,
    gpu_ids: Vec<u16>,
}

impl HetMorselExecutorBuilder {
    pub fn new() -> Self {
        let cpu_morsel_bytes = 16_384;
        let gpu_morsel_bytes = 33_554_432;

        Self {
            morsel_spec: MorselSpec {
                cpu_morsel_bytes,
                gpu_morsel_bytes,
            },
            cpu_threads: 0,
            cpu_worker_affinity: Arc::new(CpuAffinity::default()),
            gpu_worker_affinity: Arc::new(CpuAffinity::default()),
            gpu_ids: Vec::new(),
        }
    }

    pub fn morsel_spec(mut self, morsel_spec: MorselSpec) -> Self {
        self.morsel_spec = morsel_spec;
        self
    }

    pub fn cpu_threads(mut self, threads: usize) -> Self {
        self.cpu_threads = threads;
        self
    }

    pub fn worker_cpu_affinity(
        mut self,
        WorkerCpuAffinity {
            cpu_workers,
            gpu_workers,
        }: WorkerCpuAffinity,
    ) -> Self {
        self.cpu_worker_affinity = Arc::new(cpu_workers);
        self.gpu_worker_affinity = Arc::new(gpu_workers);
        self
    }

    pub fn gpu_ids(mut self, gpu_ids: Vec<u16>) -> Self {
        self.gpu_ids = gpu_ids;
        self
    }

    pub fn build(self) -> Result<HetMorselExecutor> {
        let cpu_workers = self.cpu_threads;
        let gpu_workers = self.gpu_ids.len();
        let cpu_affinity = self.cpu_worker_affinity.clone();
        let gpu_affinity = self.gpu_worker_affinity.clone();

        let unowned_context_cpu_pool = CurrentContext::get_current()?;
        let unowned_context_gpu_pool = unowned_context_cpu_pool.clone();

        let cpu_thread_pool = ThreadPoolBuilder::new()
            .num_threads(cpu_workers)
            .start_handler(move |tid| {
                CurrentContext::set_current(&unowned_context_cpu_pool)
                    .expect("Failed to set CUDA context in CPU worker thread");

                cpu_affinity
                    .clone()
                    .set_affinity(tid as u16)
                    .expect("Couldn't set CPU core affinity");
            })
            .build()?;

        let gpu_thread_pool = ThreadPoolBuilder::new()
            .num_threads(gpu_workers)
            .start_handler(move |tid| {
                CurrentContext::set_current(&unowned_context_gpu_pool)
                    .expect("Failed to set CUDA context in GPU worker thread");

                gpu_affinity
                    .clone()
                    .set_affinity(tid as u16)
                    .expect("Couldn't set CPU core affinity");
            })
            .build()?;

        let streams = (0..gpu_workers)
            .map(|_| Stream::new(StreamFlags::NON_BLOCKING, None))
            .collect::<std::result::Result<Vec<_>, CudaError>>()?;

        Ok(HetMorselExecutor {
            morsel_spec: self.morsel_spec,
            cpu_workers,
            gpu_workers,
            cpu_thread_pool,
            gpu_thread_pool,
            streams,
        })
    }
}
