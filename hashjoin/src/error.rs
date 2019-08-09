/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use datagen::error::Error as DatagenError;
use error_chain::error_chain;
use numa_gpu::error::Error as NumaGpuError;
use rayon::ThreadPoolBuildError;
use rustacuda::error::CudaError;

error_chain! {
    errors {
        InvalidArgument(msg: String) {
            description("Invalid argument error")
            display("Aborted with: {}", msg)
        }
        InvalidConversion(msg: &'static str) {
            description("Conversion error")
            display("Aborting with: {}", msg)
        }
        IntegerOverflow(msg: String) {
            description("Integer overflow error")
            display("Aborted with: {}", msg)
        }
        LogicError(msg: String) {
            description("Logic error")
            display("Aborting with: {}", msg)
        }
        RuntimeError(msg: String) {
            description("Runtime error")
            display("Aborting with: {}", msg)
        }
    }

    foreign_links {
        Csv(csv::Error);
        Cuda(CudaError);
        Datagen(DatagenError);
        Io(::std::io::Error);
        NumaGpu(NumaGpuError);
        RayonThreadPoolBuild(ThreadPoolBuildError);
    }
}
