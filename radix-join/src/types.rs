/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019-2020, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use numa_gpu::runtime::allocator;
use numa_gpu::runtime::numa::NodeRatio;
use serde_derive::Serialize;
use serde_repr::Serialize_repr;
use sql_ops::join::HashingScheme;
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
    pub enum ArgHistogramAlgorithm {
        CpuChunked,
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
        GpuRJ,
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
    pub huge_pages: Option<bool>,
}

impl From<ArgMemTypeHelper> for allocator::MemType {
    fn from(
        ArgMemTypeHelper {
            mem_type,
            node_ratios,
            huge_pages,
        }: ArgMemTypeHelper,
    ) -> Self {
        match mem_type {
            ArgMemType::System => allocator::MemType::SysMem,
            ArgMemType::Numa => allocator::MemType::NumaMem(node_ratios[0].node, huge_pages),
            ArgMemType::NumaPinned => {
                allocator::MemType::NumaPinnedMem(node_ratios[0].node, huge_pages)
            }
            ArgMemType::DistributedNuma => allocator::MemType::DistributedNumaMem(node_ratios),
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
            huge_pages,
        }: ArgMemTypeHelper,
    ) -> Self {
        match mem_type {
            ArgMemType::System => allocator::DerefMemType::SysMem,
            ArgMemType::Numa => allocator::DerefMemType::NumaMem(node_ratios[0].node, huge_pages),
            ArgMemType::NumaPinned => {
                allocator::DerefMemType::NumaPinnedMem(node_ratios[0].node, huge_pages)
            }
            ArgMemType::DistributedNuma => allocator::DerefMemType::DistributedNumaMem(node_ratios),
            ArgMemType::Pinned => allocator::DerefMemType::CudaPinnedMem,
            ArgMemType::Unified => allocator::DerefMemType::CudaUniMem,
            ArgMemType::Device => panic!("Error: Device memory not supported in this context!"),
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

impl Into<GpuHistogramAlgorithm> for ArgHistogramAlgorithm {
    fn into(self) -> GpuHistogramAlgorithm {
        match self {
            Self::CpuChunked => GpuHistogramAlgorithm::CpuChunked,
            Self::GpuChunked => GpuHistogramAlgorithm::GpuChunked,
            Self::GpuContiguous => GpuHistogramAlgorithm::GpuContiguous,
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
