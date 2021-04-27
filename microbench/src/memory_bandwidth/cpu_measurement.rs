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
use numa_gpu::runtime::memory::DerefMem;
use numa_gpu::runtime::numa;
use std::iter;
use std::ops::RangeInclusive;
use std::rc::Rc;

pub(super) struct CpuMeasurement {
    threads: RangeInclusive<ThreadCount>,
    template: DataPoint,
}

impl CpuMeasurement {
    pub(super) fn new(threads: RangeInclusive<ThreadCount>, template: DataPoint) -> Self {
        Self { threads, template }
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
        let (ThreadCount(threads_l), ThreadCount(threads_u)) = self.threads.clone().into_inner();
        let cpu_node = 0;

        let data_points: Vec<_> = benches
            .iter()
            .flat_map(|bench| {
                iter::repeat(bench).zip(ops.iter().flat_map(|op| {
                    let threads_iter = threads_l..=threads_u;
                    iter::repeat(op).zip(threads_iter.flat_map(|t| {
                        let thread_pool = Rc::new(
                            rayon::ThreadPoolBuilder::new()
                                .num_threads(t)
                                .start_handler(move |_tid| {
                                    numa::run_on_node(cpu_node).expect("Couldn't set NUMA node")
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
