/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use super::{Benchmark, ItemBytes, MemoryOperation};
use crate::types::Cycles;
use numa_gpu::runtime::memory::DerefMem;
use std::rc::Rc;
use std::time::Instant;
use std::{iter, mem};

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

macro_rules! make_benchmark {
    ($function_name:ident) => {
        extern "C" {
            fn $function_name(
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
    };
}

make_benchmark!(cpu_read_bandwidth_seq_4B);
make_benchmark!(cpu_read_bandwidth_seq_8B);
make_benchmark!(cpu_read_bandwidth_seq_16B);

make_benchmark!(cpu_write_bandwidth_seq_4B);
make_benchmark!(cpu_write_bandwidth_seq_8B);
make_benchmark!(cpu_write_bandwidth_seq_16B);

make_benchmark!(cpu_cas_bandwidth_seq_4B);
make_benchmark!(cpu_cas_bandwidth_seq_8B);
make_benchmark!(cpu_cas_bandwidth_seq_16B);

make_benchmark!(cpu_read_bandwidth_lcg_4B);
make_benchmark!(cpu_read_bandwidth_lcg_8B);
make_benchmark!(cpu_read_bandwidth_lcg_16B);

make_benchmark!(cpu_write_bandwidth_lcg_4B);
make_benchmark!(cpu_write_bandwidth_lcg_8B);
make_benchmark!(cpu_write_bandwidth_lcg_16B);

make_benchmark!(cpu_cas_bandwidth_lcg_4B);
make_benchmark!(cpu_cas_bandwidth_lcg_8B);
make_benchmark!(cpu_cas_bandwidth_lcg_16B);

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
        item_bytes: ItemBytes,
        state: &mut Self,
        mem: &DerefMem<u32>,
        thread_pool: Rc<rayon::ThreadPool>,
    ) -> (u32, u64, Cycles, u64) {
        let threads = thread_pool.current_num_threads();
        let loop_length = state.loop_length;
        let target_cycles = state.target_cycles;

        // FIXME: refactor into a function
        let f: CpuBandwidthFn = match (bench, op, item_bytes) {
            (Benchmark::Sequential, MemoryOperation::Read, ItemBytes::Bytes4) => {
                cpu_read_bandwidth_seq_4B
            }
            (Benchmark::Sequential, MemoryOperation::Read, ItemBytes::Bytes8) => {
                cpu_read_bandwidth_seq_8B
            }
            (Benchmark::Sequential, MemoryOperation::Read, ItemBytes::Bytes16) => {
                cpu_read_bandwidth_seq_16B
            }
            (Benchmark::Sequential, MemoryOperation::Write, ItemBytes::Bytes4) => {
                cpu_write_bandwidth_seq_4B
            }
            (Benchmark::Sequential, MemoryOperation::Write, ItemBytes::Bytes8) => {
                cpu_write_bandwidth_seq_8B
            }
            (Benchmark::Sequential, MemoryOperation::Write, ItemBytes::Bytes16) => {
                cpu_write_bandwidth_seq_16B
            }
            (Benchmark::Sequential, MemoryOperation::CompareAndSwap, ItemBytes::Bytes4) => {
                cpu_cas_bandwidth_seq_4B
            }
            (Benchmark::Sequential, MemoryOperation::CompareAndSwap, ItemBytes::Bytes8) => {
                cpu_cas_bandwidth_seq_8B
            }
            (Benchmark::Sequential, MemoryOperation::CompareAndSwap, ItemBytes::Bytes16) => {
                cpu_cas_bandwidth_seq_16B
            }
            (Benchmark::LinearCongruentialGenerator, MemoryOperation::Read, ItemBytes::Bytes4) => {
                cpu_read_bandwidth_lcg_4B
            }
            (Benchmark::LinearCongruentialGenerator, MemoryOperation::Read, ItemBytes::Bytes8) => {
                cpu_read_bandwidth_lcg_8B
            }
            (Benchmark::LinearCongruentialGenerator, MemoryOperation::Read, ItemBytes::Bytes16) => {
                cpu_read_bandwidth_lcg_16B
            }
            (Benchmark::LinearCongruentialGenerator, MemoryOperation::Write, ItemBytes::Bytes4) => {
                cpu_write_bandwidth_lcg_4B
            }
            (Benchmark::LinearCongruentialGenerator, MemoryOperation::Write, ItemBytes::Bytes8) => {
                cpu_write_bandwidth_lcg_8B
            }
            (
                Benchmark::LinearCongruentialGenerator,
                MemoryOperation::Write,
                ItemBytes::Bytes16,
            ) => cpu_write_bandwidth_lcg_16B,
            (
                Benchmark::LinearCongruentialGenerator,
                MemoryOperation::CompareAndSwap,
                ItemBytes::Bytes4,
            ) => cpu_cas_bandwidth_lcg_4B,
            (
                Benchmark::LinearCongruentialGenerator,
                MemoryOperation::CompareAndSwap,
                ItemBytes::Bytes8,
            ) => cpu_cas_bandwidth_lcg_8B,
            (
                Benchmark::LinearCongruentialGenerator,
                MemoryOperation::CompareAndSwap,
                ItemBytes::Bytes16,
            ) => cpu_cas_bandwidth_lcg_16B,
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
                                (mem.len() * mem::size_of::<u32>()) / item_bytes as usize,
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
