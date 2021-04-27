/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use super::data_point::DataPoint;
use super::{Benchmark, MemoryOperation};
use crate::types::{Cycles, ThreadCount};
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
        repeat: u32,
    ) -> Vec<DataPoint>
    where
        R: Fn(
            Benchmark,
            MemoryOperation,
            &mut S,
            &DerefMem<u32>,
            Rc<rayon::ThreadPool>,
        ) -> (u32, u64, Cycles, u64),
    {
        let cpu_affinity = &self.cpu_affinity;

        let data_points: Vec<_> = benches
            .iter()
            .flat_map(|bench| {
                iter::repeat(bench).zip(ops.iter().flat_map(|op| {
                    iter::repeat(op).zip(self.threads.iter().flat_map(|&ThreadCount(t)| {
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

                        iter::repeat(thread_pool.clone())
                            .zip(iter::once(true).chain(iter::repeat(false)))
                            .zip(0..repeat)
                    }))
                }))
            })
            .map(|(bench, (op, ((thread_pool, warm_up), _run_number)))| {
                let threads = ThreadCount(thread_pool.current_num_threads());
                let (clock_rate_mhz, memory_accesses, cycles, ns) =
                    run(*bench, *op, &mut state, mem, thread_pool);

                DataPoint {
                    benchmark: Some(*bench),
                    memory_operation: Some(*op),
                    warm_up,
                    threads: Some(threads),
                    throttle_reasons: None,
                    clock_rate_mhz: Some(clock_rate_mhz),
                    memory_accesses,
                    cycles,
                    ns,
                    ..self.template.clone()
                }
            })
            .collect();
        data_points
    }
}
