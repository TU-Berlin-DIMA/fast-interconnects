// Copyright 2019-2022 Clemens Lutz
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

//! Set the CPU core affinity of a thread.

use crate::error::{ErrorKind, Result};
use crate::runtime::linux_wrapper::CpuSet;
use std::default::Default;
use std::fs::File;
use std::io::Error as IoError;
use std::io::Read;
use std::path::Path;

#[derive(Clone)]
pub struct CpuAffinity {
    affinity_list: Vec<u16>,
}

impl CpuAffinity {
    /// Takes a slice containing CPU core affinities.
    pub fn from_slice(affinity_list: &[u16]) -> Self {
        Self {
            affinity_list: Vec::from(affinity_list),
        }
    }

    /// Reads a list of CPU core affinities from a file.
    ///
    /// CPU core affinities should be listed in a single line and separated with
    /// spaces. E.g.:
    ///
    /// ```ignore
    /// 0 2 4 6 1 3 5 7
    /// ```
    ///
    /// Lines that start with '#' are ignored as comments.
    ///
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let affinity_list = contents
            .lines()
            .filter(|line| !line.starts_with('#'))
            .take(1)
            .flat_map(|line| line.split_whitespace())
            .filter_map(|item| item.parse::<u16>().ok())
            .collect();

        Ok(Self { affinity_list })
    }

    /// Maps a thread ID to a CPU core ID.
    ///
    /// Returns a CPU core ID, or `None` if the thread ID is out-of-bounds.
    /// Core IDs are guaranteed to be in the order given on construction.
    ///
    pub fn thread_to_cpu(&self, tid: u16) -> Option<u16> {
        let index: usize = tid.into();
        self.affinity_list.get(index).map(|&tid| tid)
    }

    /// Binds the current thread to the CPU core by the given thread ID.
    ///
    /// Core IDs are guaranteed to be in the order given on construction.
    ///
    pub fn set_affinity(&self, tid: u16) -> Result<()> {
        let core_id = self
            .thread_to_cpu(tid)
            .ok_or_else(|| ErrorKind::InvalidArgument("Thread ID is out-of-bounds".to_string()))?;

        let mut cpu_set = CpuSet::new();
        cpu_set.add(core_id);

        unsafe {
            if libc::sched_setaffinity(
                0,
                cpu_set.bytes(),
                cpu_set.as_slice().as_ptr() as *const libc::cpu_set_t,
            ) == -1
            {
                Err(ErrorKind::Io(IoError::last_os_error()))?
            }
        }

        Ok(())
    }

    /// Returns the number of CPU core IDs currently stored in the mapping.
    pub fn len(&self) -> usize {
        self.affinity_list.len()
    }

    /// Returns the CPU core ID that the calling thread is currently running on.
    pub fn get_cpu() -> Result<u16> {
        unsafe {
            match libc::sched_getcpu() {
                -1 => Err(ErrorKind::Io(IoError::last_os_error()))?,
                cpu_id => Ok(cpu_id as u16),
            }
        }
    }
}

impl Default for CpuAffinity {
    fn default() -> Self {
        let mut cpu_set = CpuSet::new();

        unsafe {
            if libc::sched_getaffinity(
                0,
                cpu_set.bytes(),
                cpu_set.as_mut_slice().as_mut_ptr() as *mut libc::cpu_set_t,
            ) == -1
            {
                Err(ErrorKind::Io(IoError::last_os_error()))
                    .expect("Couldn't get the list of available CPU affinities from the OS")
            }
        }

        let affinity_list: Vec<_> = (0..cpu_set.max_id())
            .filter(|&cpu_id| cpu_set.is_set(cpu_id))
            .collect();

        Self { affinity_list }
    }
}
