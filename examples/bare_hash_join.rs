/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

extern crate accel;
extern crate numa_gpu;

#[macro_use]
extern crate accel_native;

use accel::device::sync;
use accel::kernel::{Block, Grid};
use accel::module::Module;
use accel::uvec::UVec;

use std::path::Path;

fn main() {
    let module_path = Path::new(env!("CUDAUTILS_PATH"));
    sync().unwrap(); // force CUDA to init device and create context

    let ops = Module::load_file(module_path).expect("Cannot load CUDA module");

    // let null_key: i64 = 0xFFFFFFFFFFFFFFFF;
    let null_key: i64 = -1;
    let mut hash_table: UVec<i64> = UVec::new(128).unwrap();
    let mut build_join_attr: UVec<i64> = UVec::new(10).unwrap();
    let mut build_selection_attr: UVec<i64> = UVec::new(10).unwrap();
    let result_size: UVec<u64> = UVec::new(1).unwrap();

    // Generate some random data
    for (i, x) in build_join_attr.iter_mut().enumerate() {
        *x = i as i64;
    }

    // Set selection attributes to 100% selectivity
    build_selection_attr
        .iter_mut()
        .by_ref()
        .map(|x| *x = 2)
        .collect::<()>();

    // Initialize hash table
    hash_table
        .iter_mut()
        .by_ref()
        .map(|entry| *entry = null_key)
        .collect::<()>();

    print!("[");
    for s in build_join_attr.iter() {
        print!("{},", s);
    }
    println!("]");

    let build_attr_len = build_join_attr.len() as u64;
    let hash_table_len = hash_table.len() as u64;

    cuda!(
        build_pipeline_kernel << [ops, Grid::x(1), Block::x(1)]
            >> (
                build_attr_len,
                result_size,
                build_selection_attr,
                build_join_attr,
                hash_table_len,
                hash_table
            )
    ).expect("Cannot launch build kernel");
    sync().unwrap();

    print!("[");
    for s in build_join_attr.iter() {
        print!("{},", s);
    }
    println!("]");
    print!("[");
    for s in build_selection_attr.iter() {
        print!("{},", s);
    }
    println!("]");
    print!("[");
    for s in result_size.iter() {
        print!("{},", s);
    }
    println!("]");
    print!("[");
    for s in hash_table.iter() {
        print!("{},", s);
    }
    println!("]");

    let mut counts_result: UVec<u64> = UVec::new(1 /* global_size */).unwrap();
    let mut probe_join_attr: UVec<i64> = UVec::new(1000).unwrap();
    let mut probe_selection_attr: UVec<i64> = UVec::new(probe_join_attr.len()).unwrap();

    // Generate some random data
    for (i, x) in probe_join_attr.iter_mut().enumerate() {
        *x = (i % build_join_attr.len()) as i64;
    }

    // Set selection attributes to 100% selectivity
    probe_selection_attr
        .iter_mut()
        .by_ref()
        .map(|x| *x = 2)
        .collect::<()>();

    // Initialize counts
    counts_result
        .iter_mut()
        .by_ref()
        .map(|count| *count = 0)
        .collect::<()>();

    let probe_attr_len = probe_join_attr.len() as u64;

    cuda!(
        aggregation_kernel << [ops, Grid::x(1), Block::x(1)]
            >> (
                probe_attr_len,
                probe_selection_attr,
                probe_join_attr,
                hash_table,
                hash_table_len,
                counts_result // size == global_size)
            )
    ).expect("Cannot launch probe kernel");

    sync().unwrap();

    println!("Build result size: {}", result_size[0]);

    print!("HT: [");
    for entry in hash_table.iter() {
        print!("{},", entry);
    }
    println!("]");

    print!("Counts: [");
    for count in counts_result.iter() {
        print!("{},", count);
    }
    println!("]");
}
