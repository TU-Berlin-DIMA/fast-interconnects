/*
 * Copyright 2019 Clemens Lutz, German Research Center for Artificial Intelligence
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use criterion::{criterion_main, BenchmarkId, Criterion, Throughput};
use datagen::relation::UniformRelation;
use numa_gpu::runtime::allocator::{Allocator, DerefMemType};
use papi::criterion::PapiMeasurement;
use papi::Papi;
use sql_ops::partition::radix_partition::{
    CpuRadixPartitionAlgorithm, CpuRadixPartitioner, PartitionedRelation,
};
use std::mem;
use std::ops::RangeInclusive;
use std::time::Duration;

fn papi_setup(event_name: &'static str) -> Criterion<PapiMeasurement> {
    let papi = Papi::init().expect("Failed to initialize PAPI");
    let papi_mnt =
        PapiMeasurement::new(&papi, event_name).expect("Failed to setup PAPI measurement");
    Criterion::default().with_measurement(papi_mnt)
}

fn radix_partition_benchmark(
    algorithm: CpuRadixPartitionAlgorithm,
    group_name: &str,
    event_name: &str,
    c: &mut Criterion<PapiMeasurement>,
) {
    const PAYLOAD_RANGE: RangeInclusive<usize> = 1..=10000;
    const NUMA_NODE: u16 = 0;
    const TUPLES: usize = (1 << 30) / (2 * mem::size_of::<i64>());
    const RADIX_BITS: [u32; 5] = [8, 10, 12, 14, 16];

    let mut data_key = Allocator::alloc_deref_mem::<i64>(DerefMemType::NumaMem(NUMA_NODE), TUPLES);
    let mut data_pay = Allocator::alloc_deref_mem::<i64>(DerefMemType::NumaMem(NUMA_NODE), TUPLES);

    UniformRelation::gen_primary_key(data_key.as_mut_slice()).unwrap();
    UniformRelation::gen_attr(data_pay.as_mut_slice(), PAYLOAD_RANGE).unwrap();

    let mut group = c.benchmark_group(group_name);
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));
    group.throughput(Throughput::Bytes(
        (data_key.len() * mem::size_of::<i64>() + data_pay.len() * mem::size_of::<i64>()) as u64,
    ));

    for &radix_bits in RADIX_BITS.iter() {
        let mut radix_prnr = CpuRadixPartitioner::new(
            algorithm,
            radix_bits,
            Allocator::deref_mem_alloc_fn(DerefMemType::NumaMem(NUMA_NODE)),
        );

        let mut partitioned_relation = PartitionedRelation::new(
            TUPLES,
            radix_bits,
            Allocator::deref_mem_alloc_fn(DerefMemType::NumaMem(NUMA_NODE)),
            Allocator::deref_mem_alloc_fn(DerefMemType::NumaMem(NUMA_NODE)),
        );

        let id = BenchmarkId::new(event_name, format!("{}_radix_bits", radix_bits));

        group.bench_with_input(id, &radix_bits, |b, &_radix_bits| {
            b.iter(|| {
                radix_prnr.partition(
                    data_key.as_slice(),
                    data_pay.as_slice(),
                    &mut partitioned_relation,
                )
            })
        });
    }

    group.finish();
}

fn benches() {
    let event_names = ["PAPI_TOT_CYC", "perf::DTLB-LOAD-MISSES"];

    for event_name in event_names.iter() {
        let mut criterion = papi_setup(event_name).configure_from_args();
        radix_partition_benchmark(
            CpuRadixPartitionAlgorithm::Chunked,
            "radix_partition",
            event_name,
            &mut criterion,
        );
        radix_partition_benchmark(
            CpuRadixPartitionAlgorithm::ChunkedSwwc,
            "radix_partition_swwc",
            event_name,
            &mut criterion,
        );
    }
}

criterion_main!(benches);
