// Copyright 2021-2022 Clemens Lutz
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
