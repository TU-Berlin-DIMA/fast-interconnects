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

use numa_gpu::runtime::allocator;
use numa_gpu::runtime::numa::{NodeRatio, PageType};
use serde_derive::Serialize;
use structopt::clap::arg_enum;

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgMemType {
        System,
        Numa,
        NumaPinned,
        DistributedNuma,
        Pinned,
        Unified,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgPageType {
        Default,
        Small,
        TransparentHuge,
        Huge2MB,
        Huge16MB,
        Huge1GB,
        Huge16GB,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgExecutionMethod {
        Cpu,
        Gpu,
        GpuStream,
        Het,
        GpuBuildHetProbe,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgSelectionVariant {
        Branching,
        Predication,
    }
}

#[derive(Debug)]
pub struct ArgMemTypeHelper {
    pub mem_type: ArgMemType,
    pub node_ratios: Box<[NodeRatio]>,
    pub page_type: ArgPageType,
}

impl From<ArgMemTypeHelper> for allocator::DerefMemType {
    fn from(
        ArgMemTypeHelper {
            mem_type,
            node_ratios,
            page_type,
        }: ArgMemTypeHelper,
    ) -> Self {
        match mem_type {
            ArgMemType::System => allocator::DerefMemType::SysMem,
            ArgMemType::Numa => allocator::DerefMemType::NumaMem {
                node: node_ratios[0].node,
                page_type: page_type.into(),
            },
            ArgMemType::NumaPinned => allocator::DerefMemType::NumaPinnedMem {
                node: node_ratios[0].node,
                page_type: page_type.into(),
            },
            ArgMemType::DistributedNuma => allocator::DerefMemType::DistributedNumaMem {
                nodes: node_ratios,
                page_type: page_type.into(),
            },
            ArgMemType::Pinned => allocator::DerefMemType::CudaPinnedMem,
            ArgMemType::Unified => allocator::DerefMemType::CudaUniMem,
        }
    }
}

impl From<ArgPageType> for PageType {
    fn from(arg_page_type: ArgPageType) -> PageType {
        match arg_page_type {
            ArgPageType::Default => PageType::Default,
            ArgPageType::Small => PageType::Small,
            ArgPageType::TransparentHuge => PageType::TransparentHuge,
            ArgPageType::Huge2MB => PageType::Huge2MB,
            ArgPageType::Huge16MB => PageType::Huge16MB,
            ArgPageType::Huge1GB => PageType::Huge1GB,
            ArgPageType::Huge16GB => PageType::Huge16GB,
        }
    }
}
