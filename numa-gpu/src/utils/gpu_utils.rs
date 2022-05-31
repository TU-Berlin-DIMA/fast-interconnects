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

//! Helper utilities for the GPU.

use crate::error::Result;
use crate::runtime::memory::LaunchablePtr;
use once_cell::sync::Lazy;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::launch;
use rustacuda::memory::DeviceCopy;
use rustacuda::module::Module;
use rustacuda::stream::Stream;
use std::ffi::CStr;
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;

static mut MODULE_OWNER: Option<Module> = None;
static MODULE: Lazy<&'static Module> = Lazy::new(|| {
    let mut module_path = PathBuf::from(env!("RESOURCES_PATH"));
    module_path.push("noop.ptx\0");

    let module_cstr = CStr::from_bytes_with_nul(module_path.as_os_str().as_bytes())
        .expect("Failed to load CUDA module, check your RESOURCES_PATH");
    let module = Module::load_from_file(module_cstr).expect("Failed to load CUDA module");

    unsafe { MODULE_OWNER.get_or_insert(module) }
});

/// Launches an empty GPU kernel that does nothing.
pub fn noop<T, P, G, B>(pointer: P, grid_size: G, block_size: B, stream: &Stream) -> Result<()>
where
    P: Into<LaunchablePtr<T>>,
    T: DeviceCopy,
    G: Into<GridSize>,
    B: Into<BlockSize>,
{
    let module = *MODULE;
    let p: LaunchablePtr<T> = pointer.into();

    unsafe {
        launch!(module.noop<<<grid_size, block_size, 0, stream>>>(p.clone()))?;
    }

    Ok(())
}
