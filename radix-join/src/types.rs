/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2021, Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use numa_gpu::runtime::allocator;
use numa_gpu::runtime::numa::{NodeRatio, PageType};
use numa_gpu::utils::DeviceType;
use serde_derive::Serialize;
use serde_repr::Serialize_repr;
use sql_ops::join::HashingScheme;
use sql_ops::partition::cpu_radix_partition::CpuHistogramAlgorithm;
use sql_ops::partition::gpu_radix_partition::{GpuHistogramAlgorithm, GpuRadixPartitionAlgorithm};
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
        NumaPinned,
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
        Huge16MB,
        Huge1GB,
        Huge16GB,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgHistogramAlgorithm {
        CpuChunked,
        CpuChunkedSimd,
        GpuChunked,
        GpuContiguous,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgRadixPartitionAlgorithm {
        NC,
        LASWWC,
        SSWWC,
        SSWWCNT,
        SSWWCv2,
        HSSWWC,
        HSSWWCv2,
        HSSWWCv3,
        HSSWWCv4,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgExecutionMethod {
        GpuRadixJoinTwoPass,
        GpuTritonJoinTwoPass,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq, Serialize)]
    pub enum ArgHashingScheme {
        Perfect,
        LinearProbing,
        BucketChaining,
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
            ArgMemType::NumaPinned => allocator::MemType::NumaPinnedMem {
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
            ArgPageType::Huge16MB => PageType::Huge16MB,
            ArgPageType::Huge1GB => PageType::Huge1GB,
            ArgPageType::Huge16GB => PageType::Huge16GB,
        }
    }
}

impl Into<GpuRadixPartitionAlgorithm> for ArgRadixPartitionAlgorithm {
    fn into(self) -> GpuRadixPartitionAlgorithm {
        match self {
            Self::NC => GpuRadixPartitionAlgorithm::NC,
            Self::LASWWC => GpuRadixPartitionAlgorithm::LASWWC,
            Self::SSWWC => GpuRadixPartitionAlgorithm::SSWWC,
            Self::SSWWCNT => GpuRadixPartitionAlgorithm::SSWWCNT,
            Self::SSWWCv2 => GpuRadixPartitionAlgorithm::SSWWCv2,
            Self::HSSWWC => GpuRadixPartitionAlgorithm::HSSWWC,
            Self::HSSWWCv2 => GpuRadixPartitionAlgorithm::HSSWWCv2,
            Self::HSSWWCv3 => GpuRadixPartitionAlgorithm::HSSWWCv3,
            Self::HSSWWCv4 => GpuRadixPartitionAlgorithm::HSSWWCv4,
        }
    }
}

impl Into<DeviceType<CpuHistogramAlgorithm, GpuHistogramAlgorithm>> for ArgHistogramAlgorithm {
    fn into(self) -> DeviceType<CpuHistogramAlgorithm, GpuHistogramAlgorithm> {
        match self {
            Self::CpuChunked => DeviceType::Cpu(CpuHistogramAlgorithm::Chunked),
            Self::CpuChunkedSimd => DeviceType::Cpu(CpuHistogramAlgorithm::ChunkedSimd),
            Self::GpuChunked => DeviceType::Gpu(GpuHistogramAlgorithm::Chunked),
            Self::GpuContiguous => DeviceType::Gpu(GpuHistogramAlgorithm::Contiguous),
        }
    }
}

impl From<ArgHashingScheme> for HashingScheme {
    fn from(ahs: ArgHashingScheme) -> Self {
        match ahs {
            ArgHashingScheme::Perfect => HashingScheme::Perfect,
            ArgHashingScheme::LinearProbing => HashingScheme::LinearProbing,
            ArgHashingScheme::BucketChaining => HashingScheme::BucketChaining,
        }
    }
}
