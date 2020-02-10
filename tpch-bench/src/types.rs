/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2020, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use numa_gpu::runtime::allocator;
use numa_gpu::runtime::numa::NodeRatio;
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
}

impl From<ArgMemTypeHelper> for allocator::DerefMemType {
    fn from(
        ArgMemTypeHelper {
            mem_type,
            node_ratios,
        }: ArgMemTypeHelper,
    ) -> Self {
        match mem_type {
            ArgMemType::System => allocator::DerefMemType::SysMem,
            ArgMemType::Numa => allocator::DerefMemType::NumaMem(node_ratios[0].node),
            ArgMemType::NumaPinned => allocator::DerefMemType::NumaPinnedMem(node_ratios[0].node),
            ArgMemType::DistributedNuma => allocator::DerefMemType::DistributedNumaMem(node_ratios),
            ArgMemType::Pinned => allocator::DerefMemType::CudaPinnedMem,
            ArgMemType::Unified => allocator::DerefMemType::CudaUniMem,
        }
    }
}
