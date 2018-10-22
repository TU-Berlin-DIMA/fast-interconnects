extern crate accel;
#[macro_use]
extern crate criterion;
extern crate numa_gpu;

use accel::device::sync;
use accel::uvec::UVec;

use criterion::Criterion;

use numa_gpu::operators::hash_join;

fn basic_functionality() {
    let hash_table = hash_join::HashTable::new(1024);
    let mut build_join_attr = UVec::<i64>::new(10).unwrap();
    let mut build_selection_attr: UVec<i64> = UVec::new(build_join_attr.len()).unwrap();
    let mut counts_result: UVec<u64> = UVec::new(1 /* global_size */).unwrap();
    let mut probe_join_attr: UVec<i64> = UVec::new(1000).unwrap();
    let mut probe_selection_attr: UVec<i64> = UVec::new(probe_join_attr.len()).unwrap();

    // Generate some random build data
    for (i, x) in build_join_attr.as_slice_mut().iter_mut().enumerate() {
        *x = i as i64;
    }

    // Generate some random probe data
    for (i, x) in probe_join_attr.as_slice_mut().iter_mut().enumerate() {
        *x = (i % build_join_attr.len()) as i64;
    }

    // Initialize counts
    counts_result
        .iter_mut()
        .map(|count| *count = 0)
        .collect::<()>();

    // Set build selection attributes to 100% selectivity
    build_selection_attr
        .iter_mut()
        .map(|x| *x = 2)
        .collect::<()>();

    // Set probe selection attributes to 100% selectivity
    probe_selection_attr
        .iter_mut()
        .map(|x| *x = 2)
        .collect::<()>();

    let mut hj_op = hash_join::CudaHashJoinBuilder::default()
        .build_dim(1, 1)
        .probe_dim(1, 1)
        .hash_table(hash_table)
        .result_set(counts_result)
        .build();

    // println!("{:#?}", hj_op);

    let join_result = hj_op
        .build(build_join_attr, build_selection_attr)
        .probe(probe_join_attr, probe_selection_attr);

    sync().unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("basic functionality", |b| b.iter(|| basic_functionality()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
