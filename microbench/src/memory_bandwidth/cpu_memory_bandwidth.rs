/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use super::MemoryOperation;
use numa_gpu::runtime::memory::DerefMem;
use std::iter;
use std::rc::Rc;
use std::time::Instant;

pub(super) type CpuBandwidthFn = unsafe extern "C" fn(
    op: MemoryOperation,
    data: *mut u32,
    size: usize,
    tid: usize,
    num_threads: usize,
);

#[allow(dead_code)]
#[derive(Debug)]
pub(super) struct CpuMemoryBandwidth {
    cpu_node: u16,
}

impl CpuMemoryBandwidth {
    pub(super) fn new(cpu_node: u16) -> Self {
        Self { cpu_node }
    }

    pub(super) fn run(
        f: CpuBandwidthFn,
        op: MemoryOperation,
        _state: &mut Self,
        mem: &DerefMem<u32>,
        thread_pool: Rc<rayon::ThreadPool>,
    ) -> (u32, u64, u64) {
        let threads = thread_pool.current_num_threads();
        let len = mem.len();

        let timer = Instant::now();

        thread_pool.scope(|s| {
            (0..threads)
                .zip(iter::repeat(mem))
                .for_each(|(tid, r_mem)| {
                    s.spawn(move |_| {
                        let ptr = r_mem.as_ptr() as *mut u32;

                        unsafe { f(op, ptr, len, tid, threads) };
                    });
                })
        });

        let duration = timer.elapsed();
        let ns: u64 = duration.as_secs() * 10_u64.pow(9) + duration.subsec_nanos() as u64;
        let cycles = 0;
        (0, cycles, ns)
    }
}
