/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use super::cpu_memory_bandwidth::CpuBandwidthFn;
use super::data_point::DataPoint;
use super::MemoryOperation;
use crate::types::ThreadCount;
use numa_gpu::runtime::memory::DerefMem;
use numa_gpu::runtime::numa;
use std::iter;
use std::ops::RangeInclusive;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub(super) struct CpuNamedBandwidthFn<'n> {
    pub(super) f: CpuBandwidthFn,
    pub(super) name: &'n str,
}

pub(super) struct CpuMeasurement<'h, 'd, 'n> {
    threads: RangeInclusive<ThreadCount>,
    template: DataPoint<'h, 'd, 'n>,
}

impl<'h, 'd, 'n> CpuMeasurement<'h, 'd, 'n> {
    pub(super) fn new(
        threads: RangeInclusive<ThreadCount>,
        template: DataPoint<'h, 'd, 'n>,
    ) -> Self {
        Self { threads, template }
    }

    pub(super) fn measure<R, S>(
        &self,
        mem: &DerefMem<u32>,
        mut state: S,
        run: R,
        futs: Vec<CpuNamedBandwidthFn<'n>>,
        ops: Vec<MemoryOperation>,
        repeat: u32,
    ) -> Vec<DataPoint<'h, 'd, 'n>>
    where
        R: Fn(
            CpuBandwidthFn,
            MemoryOperation,
            &mut S,
            &DerefMem<u32>,
            Rc<rayon::ThreadPool>,
        ) -> (u32, u64, u64),
    {
        let (ThreadCount(threads_l), ThreadCount(threads_u)) = self.threads.clone().into_inner();
        let cpu_node = 0;

        let data_points: Vec<_> = futs
            .iter()
            .flat_map(|fut| {
                iter::repeat(fut).zip(ops.iter().flat_map(|op| {
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
            .map(|(named_fut, (op, ((thread_pool, warm_up), _run_number)))| {
                let threads = ThreadCount(thread_pool.current_num_threads());
                let CpuNamedBandwidthFn { f: fut, name } = named_fut;
                let (clock_rate_mhz, cycles, ns) = run(*fut, *op, &mut state, mem, thread_pool);

                DataPoint {
                    function_name: name,
                    memory_operation: Some(*op),
                    warm_up,
                    threads: Some(threads),
                    throttle_reasons: None,
                    clock_rate_mhz: Some(clock_rate_mhz),
                    cycles,
                    ns,
                    ..self.template.clone()
                }
            })
            .collect();
        data_points
    }
}
