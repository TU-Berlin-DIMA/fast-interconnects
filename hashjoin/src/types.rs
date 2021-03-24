/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use numa_gpu::runtime::allocator;
use numa_gpu::runtime::cuda::CudaTransferStrategy;
use numa_gpu::runtime::numa::{NodeRatio, PageType};
use serde_derive::Serialize;
use serde_repr::Serialize_repr;
use sql_ops::join::HashingScheme;
use structopt::clap::arg_enum;

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub enum ArgDataSet {
        Blanas,
        Blanas4MB,
        Kim,
        Test,
        Lutz2Gv32G,
        Lutz32Gv32G,
        Custom,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgDataDistribution {
        Uniform,
        Zipf,
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DataDistribution {
    Uniform,
    Zipf(f64),
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgMemType {
        System,
        Numa,
        NumaLazyPinned,
        DistributedNuma,
        Pinned,
        Unified,
        Device,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgPageType {
        Default,
        Small,
        TransparentHuge,
        Huge2MB,
        Huge1GB,
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
    pub enum ArgTransferStrategy {
        PageableCopy,
        PinnedCopy,
        LazyPinnedCopy,
        Unified,
        Coherence,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgHashingScheme {
        Perfect,
        LinearProbing,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize_repr)]
    #[repr(usize)]
    pub enum ArgTupleBytes {
        Bytes8 = 8,
        Bytes16 = 16,
    }
}

#[derive(Debug)]
pub struct ArgMemTypeHelper {
    pub mem_type: ArgMemType,
    pub node_ratios: Box<[NodeRatio]>,
    pub page_type: ArgPageType,
}

impl From<ArgMemTypeHelper> for allocator::MemType {
    fn from(
        ArgMemTypeHelper {
            mem_type,
            node_ratios,
            page_type,
        }: ArgMemTypeHelper,
    ) -> Self {
        match mem_type {
            ArgMemType::System => allocator::MemType::SysMem,
            ArgMemType::Numa => allocator::MemType::NumaMem {
                node: node_ratios[0].node,
                page_type: page_type.into(),
            },
            ArgMemType::NumaLazyPinned => allocator::MemType::NumaPinnedMem {
                node: node_ratios[0].node,
                page_type: page_type.into(),
            },
            ArgMemType::DistributedNuma => allocator::MemType::DistributedNumaMem {
                nodes: node_ratios,
                page_type: page_type.into(),
            },
            ArgMemType::Pinned => allocator::MemType::CudaPinnedMem,
            ArgMemType::Unified => allocator::MemType::CudaUniMem,
            ArgMemType::Device => allocator::MemType::CudaDevMem,
        }
    }
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
            ArgMemType::NumaLazyPinned => allocator::DerefMemType::NumaPinnedMem {
                node: node_ratios[0].node,
                page_type: page_type.into(),
            },
            ArgMemType::DistributedNuma => allocator::DerefMemType::DistributedNumaMem {
                nodes: node_ratios,
                page_type: page_type.into(),
            },
            ArgMemType::Pinned => allocator::DerefMemType::CudaPinnedMem,
            ArgMemType::Unified => allocator::DerefMemType::CudaUniMem,
            ArgMemType::Device => panic!("Error: Device memory not supported in this context!"),
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
            ArgPageType::Huge1GB => PageType::Huge1GB,
        }
    }
}

impl From<ArgTransferStrategy> for CudaTransferStrategy {
    fn from(asm: ArgTransferStrategy) -> Self {
        match asm {
            ArgTransferStrategy::PageableCopy => CudaTransferStrategy::PageableCopy,
            ArgTransferStrategy::PinnedCopy => CudaTransferStrategy::PinnedCopy,
            ArgTransferStrategy::LazyPinnedCopy => CudaTransferStrategy::LazyPinnedCopy,
            ArgTransferStrategy::Unified => {
                panic!("Error: Unified memory cannot be handled by CudaTransferStrategy!")
            }
            ArgTransferStrategy::Coherence => CudaTransferStrategy::Coherence,
        }
    }
}

impl From<ArgHashingScheme> for HashingScheme {
    fn from(ahs: ArgHashingScheme) -> Self {
        match ahs {
            ArgHashingScheme::Perfect => HashingScheme::Perfect,
            ArgHashingScheme::LinearProbing => HashingScheme::LinearProbing,
        }
    }
}
