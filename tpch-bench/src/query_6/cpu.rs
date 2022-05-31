// Copyright 2020-2022 Clemens Lutz
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

use super::tables::LineItem;
use crate::error::{ErrorKind, Result};
use crate::types::ArgSelectionVariant;
use numa_gpu::runtime::cpu_affinity::CpuAffinity;
use numa_gpu::utils::CachePadded;
use std::sync::Arc;
use std::time::{Duration, Instant};

extern "C" {
    fn tpch_q6_branching(
        length: u64,
        l_shipdate: *const i32,
        l_discount: *const i32,
        l_quantity: *const i32,
        l_extendedprice: *const i32,
        revenue: *mut i64,
    );
    fn tpch_q6_predication(
        length: u64,
        l_shipdate: *const i32,
        l_discount: *const i32,
        l_quantity: *const i32,
        l_extendedprice: *const i32,
        revenue: *mut i64,
    );
}

pub struct Query6Cpu {
    threads: usize,
    cpu_affinity: CpuAffinity,
    selection_variant: ArgSelectionVariant,
}

impl Query6Cpu {
    pub fn new(
        threads: usize,
        cpu_affinity: &CpuAffinity,
        selection_variant: ArgSelectionVariant,
    ) -> Self {
        Self {
            threads,
            cpu_affinity: cpu_affinity.clone(),
            selection_variant,
        }
    }

    pub fn run(&self, lineitem: &LineItem) -> Result<(i64, Duration)> {
        let mut thread_revenue = vec![CachePadded { value: 0_i64 }; self.threads];

        let boxed_cpu_affinity = Arc::new(self.cpu_affinity.clone());
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .start_handler(move |tid| {
                boxed_cpu_affinity
                    .clone()
                    .set_affinity(tid as u16)
                    .expect("Couldn't set CPU core affinity")
            })
            .build()
            .map_err(|_| ErrorKind::RuntimeError("Failed to create thread pool".to_string()))?;
        let chunk_len = (lineitem.len() + self.threads - 1) / self.threads;

        let l_shipdate_chunks: Vec<_> = lineitem.shipdate.as_slice().chunks(chunk_len).collect();
        let l_discount_chunks: Vec<_> = lineitem.discount.as_slice().chunks(chunk_len).collect();
        let l_quantity_chunks: Vec<_> = lineitem.quantity.as_slice().chunks(chunk_len).collect();
        let l_extendedprice_chunks: Vec<_> = lineitem
            .extendedprice
            .as_slice()
            .chunks(chunk_len)
            .collect();

        let q6_f = match self.selection_variant {
            ArgSelectionVariant::Branching => tpch_q6_branching,
            ArgSelectionVariant::Predication => tpch_q6_predication,
        };

        let timer = Instant::now();
        thread_pool.scope(|s| {
            for (((((_tid, l_shipdate), l_discount), l_quantity), l_extendedprice), revenue) in (0
                ..self.threads)
                .zip(l_shipdate_chunks)
                .zip(l_discount_chunks)
                .zip(l_quantity_chunks)
                .zip(l_extendedprice_chunks)
                .zip(thread_revenue.iter_mut())
            {
                s.spawn(move |_| {
                    unsafe {
                        q6_f(
                            l_shipdate.len() as u64,
                            l_shipdate.as_ptr(),
                            l_discount.as_ptr(),
                            l_quantity.as_ptr(),
                            l_extendedprice.as_ptr(),
                            &mut revenue.value,
                        )
                    };
                });
            }
        });
        let time = timer.elapsed();

        let revenue = thread_revenue.iter().map(|padded| padded.value).sum();

        Ok((revenue, time))
    }
}
