/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let resources = "resources/";
    let files = vec!["noop.ptx"];

    let src_paths: Vec<_> = files
        .iter()
        .map(|file| {
            let mut p = PathBuf::new();
            p.push(resources);
            p.push(file);
            p
        })
        .collect();
    let dst_paths: Vec<_> = files
        .iter()
        .map(|file| {
            let mut p = PathBuf::new();
            p.push(&out_dir);
            p.push(file);
            p
        })
        .collect();

    println!("cargo:rustc-env=RESOURCES_PATH={}", out_dir);

    src_paths
        .iter()
        .zip(dst_paths.iter())
        .for_each(|(src, dst)| {
            let _ = fs::copy(src, dst).expect("Failed to copy resources");
        });

    src_paths.iter().for_each(|file| {
        println!(
            "cargo:rerun-if-changed={}",
            file.to_str().expect("Failed to convert path to UTF-8")
        )
    });
}
