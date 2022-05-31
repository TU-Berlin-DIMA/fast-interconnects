// Copyright 2018-2022 Clemens Lutz
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

use super::data_point::DataPoint;
use super::{Benchmark, ItemBytes, MemoryOperation};
use crate::types::{Cycles, ThreadCount};
use itertools::{iproduct, izip};
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::runtime::memory::DerefMem;
use std::iter;
use std::rc::Rc;
use std::sync::Arc;

pub(super) struct CpuMeasurement {
    threads: Vec<ThreadCount>,
    cpu_affinity: Arc<CpuAffinity>,
    template: DataPoint,
}

impl CpuMeasurement {
    pub(super) fn new(
        threads: Vec<ThreadCount>,
        cpu_affinity: CpuAffinity,
        template: DataPoint,
    ) -> Self {
        let cpu_affinity = Arc::new(cpu_affinity);

        Self {
            threads,
            cpu_affinity,
            template,
        }
    }

    pub(super) fn measure<R, S>(
        &self,
        mem: &DerefMem<u32>,
        mut state: S,
        run: R,
        benches: Vec<Benchmark>,
        ops: Vec<MemoryOperation>,
        item_bytes: Vec<ItemBytes>,
        repeat: u32,
    ) -> Vec<DataPoint>
    where
        R: Fn(
            Benchmark,
            MemoryOperation,
            ItemBytes,
            &mut S,
            &DerefMem<u32>,
            Rc<rayon::ThreadPool>,
        ) -> (u32, u64, Cycles, u64),
    {
        let cpu_affinity = &self.cpu_affinity;

        let data_points: Vec<_> = iproduct!(
            iproduct!(
                benches.iter(),
                ops.iter(),
                item_bytes.iter(),
                self.threads.iter().map(|&ThreadCount(t)| {
                    let cpu_affinity = cpu_affinity.clone();
                    let thread_pool = Rc::new(
                        rayon::ThreadPoolBuilder::new()
                            .num_threads(t)
                            .start_handler(move |tid| {
                                cpu_affinity
                                    .clone()
                                    .set_affinity(tid as u16)
                                    .expect("Couldn't set CPU core affinity")
                            })
                            .build()
                            .expect("Couldn't build Rayon thread pool"),
                    );
                    thread_pool
                })
            ),
            izip!(iter::once(true).chain(iter::repeat(false)), 0..repeat)
        )
        .map(
            |((&bench, &op, &item_bytes, thread_pool), (warm_up, _run_number))| {
                let threads = ThreadCount(thread_pool.current_num_threads());
                let (clock_rate_mhz, memory_accesses, cycles, ns) =
                    run(bench, op, item_bytes, &mut state, mem, thread_pool);

                DataPoint {
                    benchmark: Some(bench),
                    memory_operation: Some(op),
                    item_bytes: Some(item_bytes),
                    warm_up,
                    threads: Some(threads),
                    throttle_reasons: None,
                    clock_rate_mhz: Some(clock_rate_mhz),
                    memory_accesses,
                    cycles,
                    ns,
                    ..self.template.clone()
                }
            },
        )
        .collect();
        data_points
    }
}
