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

use data_store::error::Error as DataStoreError;
use datagen::error::Error as DatagenError;
use error_chain::error_chain;
use numa_gpu::error::Error as NumaGpuError;
use rayon::ThreadPoolBuildError;
use rustacuda::error::CudaError;
use sql_ops::error::Error as SqlOpsError;

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
        DataStore(DataStoreError);
        Datagen(DatagenError);
        Io(::std::io::Error);
        NumaGpu(NumaGpuError);
        SqlOps(SqlOpsError);
        RayonThreadPoolBuild(ThreadPoolBuildError);
    }
}
