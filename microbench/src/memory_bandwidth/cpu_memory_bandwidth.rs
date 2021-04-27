/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use super::{Benchmark, MemoryOperation};
use crate::types::Cycles;
use numa_gpu::runtime::memory::DerefMem;
use std::iter;
use std::rc::Rc;
use std::time::Instant;

pub(super) type CpuBandwidthFn = unsafe extern "C" fn(
    data: *mut u32,
    size: usize,
    loop_length: u32,
    target_cycles: u64,
    memory_accesses: *mut u64,
    measured_cycles: *mut u64,
    tid: usize,
    num_threads: usize,
);

extern "C" {
    fn cpu_read_bandwidth_seq(
        data: *mut u32,
        size: usize,
        loop_length: u32,
        target_cycles: u64,
        memory_accesses: *mut u64,
        measured_cycles: *mut u64,
        tid: usize,
        num_threads: usize,
    );
    fn cpu_write_bandwidth_seq(
        data: *mut u32,
        size: usize,
        loop_length: u32,
        target_cycles: u64,
        memory_accesses: *mut u64,
        measured_cycles: *mut u64,
        tid: usize,
        num_threads: usize,
    );
    fn cpu_cas_bandwidth_seq(
        data: *mut u32,
        size: usize,
        loop_length: u32,
        target_cycles: u64,
        memory_accesses: *mut u64,
        measured_cycles: *mut u64,
        tid: usize,
        num_threads: usize,
    );
    fn cpu_read_bandwidth_lcg(
        data: *mut u32,
        size: usize,
        loop_length: u32,
        target_cycles: u64,
        memory_accesses: *mut u64,
        measured_cycles: *mut u64,
        tid: usize,
        num_threads: usize,
    );
    fn cpu_write_bandwidth_lcg(
        data: *mut u32,
        size: usize,
        loop_length: u32,
        target_cycles: u64,
        memory_accesses: *mut u64,
        measured_cycles: *mut u64,
        tid: usize,
        num_threads: usize,
    );
    fn cpu_cas_bandwidth_lcg(
        data: *mut u32,
        size: usize,
        loop_length: u32,
        target_cycles: u64,
        memory_accesses: *mut u64,
        measured_cycles: *mut u64,
        tid: usize,
        num_threads: usize,
    );
}

#[allow(dead_code)]
#[derive(Debug)]
pub(super) struct CpuMemoryBandwidth {
    cpu_node: u16,
    loop_length: u32,
    target_cycles: Cycles,
}

impl CpuMemoryBandwidth {
    pub(super) fn new(cpu_node: u16, loop_length: u32, target_cycles: Cycles) -> Self {
        Self {
            cpu_node,
            loop_length,
            target_cycles,
        }
    }

    pub(super) fn run(
        bench: Benchmark,
        op: MemoryOperation,
        state: &mut Self,
        mem: &DerefMem<u32>,
        thread_pool: Rc<rayon::ThreadPool>,
    ) -> (u32, u64, Cycles, u64) {
        let threads = thread_pool.current_num_threads();
        let len = mem.len();
        let loop_length = state.loop_length;
        let target_cycles = state.target_cycles;

        let f: CpuBandwidthFn = match (bench, op) {
            (Benchmark::Sequential, MemoryOperation::Read) => cpu_read_bandwidth_seq,
            (Benchmark::Sequential, MemoryOperation::Write) => cpu_write_bandwidth_seq,
            (Benchmark::Sequential, MemoryOperation::CompareAndSwap) => cpu_cas_bandwidth_seq,
            (Benchmark::LinearCongruentialGenerator, MemoryOperation::Read) => {
                cpu_read_bandwidth_lcg
            }
            (Benchmark::LinearCongruentialGenerator, MemoryOperation::Write) => {
                cpu_write_bandwidth_lcg
            }
            (Benchmark::LinearCongruentialGenerator, MemoryOperation::CompareAndSwap) => {
                cpu_cas_bandwidth_lcg
            }
        };

        let mut memory_accesses = vec![0; threads];
        let mut measured_cycles = vec![0; threads];

        let timer = Instant::now();

        thread_pool.scope(|s| {
            (0..threads)
                .zip(iter::repeat(mem))
                .zip(memory_accesses.iter_mut())
                .zip(measured_cycles.iter_mut())
                .for_each(|(((tid, r_mem), memory_accesses), measured_cycles)| {
                    s.spawn(move |_| {
                        let ptr = r_mem.as_ptr() as *mut u32;

                        unsafe {
                            f(
                                ptr,
                                len,
                                loop_length,
                                target_cycles.0,
                                memory_accesses,
                                measured_cycles,
                                tid,
                                threads,
                            )
                        };
                    });
                })
        });

        let duration = timer.elapsed();
        let ns: u64 = duration.as_secs() * 10_u64.pow(9) + duration.subsec_nanos() as u64;

        let cycles = Cycles(
            *measured_cycles
                .iter()
                .max()
                .expect("Failed due to empty vector"),
        );
        let memory_accesses = memory_accesses.iter().sum();

        (0, memory_accesses, cycles, ns)
    }
}
