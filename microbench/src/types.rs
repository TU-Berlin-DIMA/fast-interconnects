// Copyright 2018-2022 Clemens Lutz
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

use super::ArgPageType;
use numa_gpu::runtime::allocator;
use numa_gpu::utils::DeviceType;
use serde_derive::Serialize;

/// The device type and it's ID
///
/// Used to specify where a task should be run.
/// For example, CPU ID for numactl or GPU ID for CUDA
pub type DeviceId = DeviceType<u16, u32>;

#[derive(Copy, Clone, Debug, PartialEq, Serialize)]
pub enum BareMemType {
    System,
    Numa,
    NumaPinned,
    Pinned,
    Unified,
    Device,
}

#[derive(Debug)]
pub struct MemTypeDescription {
    pub bare_mem_type: BareMemType,
    pub location: Option<u16>,
    pub page_type: ArgPageType,
}

impl From<&allocator::MemType> for MemTypeDescription {
    fn from(mem_type: &allocator::MemType) -> Self {
        let (bare_mem_type, location, page_type) = match mem_type {
            allocator::MemType::SysMem => (BareMemType::System, None, ArgPageType::Default),
            allocator::MemType::AlignedSysMem { .. } => unimplemented!(),
            allocator::MemType::NumaMem { node, page_type } => {
                (BareMemType::Numa, Some(*node), (*page_type).into())
            }
            allocator::MemType::NumaPinnedMem { node, page_type } => {
                (BareMemType::NumaPinned, Some(*node), (*page_type).into())
            }
            allocator::MemType::DistributedNumaMem { .. } => unimplemented!(),
            allocator::MemType::DistributedNumaMemWithLen { .. } => unimplemented!(),
            allocator::MemType::CudaPinnedMem => (BareMemType::Pinned, None, ArgPageType::Default),
            allocator::MemType::CudaUniMem => (BareMemType::Unified, None, ArgPageType::Default),
            allocator::MemType::CudaDevMem => (BareMemType::Device, None, ArgPageType::Default),
        };

        Self {
            bare_mem_type,
            location,
            page_type,
        }
    }
}

/// Clock cycles
#[derive(Debug, Copy, Clone, Default, Serialize, Eq, Ord, PartialEq, PartialOrd)]
pub struct Cycles(pub u64);

/// CUDA grid size
#[derive(Debug, Copy, Clone, Default, Serialize, Eq, Ord, PartialEq, PartialOrd)]
pub struct Grid(pub u32);

/// CUDA block size
#[derive(Debug, Copy, Clone, Default, Serialize, Eq, Ord, PartialEq, PartialOrd)]
pub struct Block(pub u32);

/// Thread count
///
/// The number of CPU threads.
#[derive(Copy, Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ThreadCount(pub usize);
