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

use cuda_driver_sys::CUresult;
use error_chain::error_chain;
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
        Cuda(CudaError);
        Io(::std::io::Error);
        ProcFs(procfs::ProcError);
        RayonThreadPoolBuild(rayon::ThreadPoolBuildError);
    }
}

/// Converts raw C CUresult into Rust-ified Result type
///
/// Copied from Rustacuda, because visibility is constrained to crate scope.
pub trait ToResult {
    fn to_result(self) -> Result<()>;
}

impl ToResult for CUresult {
    fn to_result(self) -> Result<()> {
        match self {
            CUresult::CUDA_SUCCESS => Ok(()),
            CUresult::CUDA_ERROR_INVALID_VALUE => Err(CudaError::InvalidValue),
            CUresult::CUDA_ERROR_OUT_OF_MEMORY => Err(CudaError::OutOfMemory),
            CUresult::CUDA_ERROR_NOT_INITIALIZED => Err(CudaError::NotInitialized),
            CUresult::CUDA_ERROR_DEINITIALIZED => Err(CudaError::Deinitialized),
            CUresult::CUDA_ERROR_PROFILER_DISABLED => Err(CudaError::ProfilerDisabled),
            CUresult::CUDA_ERROR_PROFILER_NOT_INITIALIZED => Err(CudaError::ProfilerNotInitialized),
            CUresult::CUDA_ERROR_PROFILER_ALREADY_STARTED => Err(CudaError::ProfilerAlreadyStarted),
            CUresult::CUDA_ERROR_PROFILER_ALREADY_STOPPED => Err(CudaError::ProfilerAlreadyStopped),
            CUresult::CUDA_ERROR_NO_DEVICE => Err(CudaError::NoDevice),
            CUresult::CUDA_ERROR_INVALID_DEVICE => Err(CudaError::InvalidDevice),
            CUresult::CUDA_ERROR_INVALID_IMAGE => Err(CudaError::InvalidImage),
            CUresult::CUDA_ERROR_INVALID_CONTEXT => Err(CudaError::InvalidContext),
            CUresult::CUDA_ERROR_CONTEXT_ALREADY_CURRENT => Err(CudaError::ContextAlreadyCurrent),
            CUresult::CUDA_ERROR_MAP_FAILED => Err(CudaError::MapFailed),
            CUresult::CUDA_ERROR_UNMAP_FAILED => Err(CudaError::UnmapFailed),
            CUresult::CUDA_ERROR_ARRAY_IS_MAPPED => Err(CudaError::ArrayIsMapped),
            CUresult::CUDA_ERROR_ALREADY_MAPPED => Err(CudaError::AlreadyMapped),
            CUresult::CUDA_ERROR_NO_BINARY_FOR_GPU => Err(CudaError::NoBinaryForGpu),
            CUresult::CUDA_ERROR_ALREADY_ACQUIRED => Err(CudaError::AlreadyAcquired),
            CUresult::CUDA_ERROR_NOT_MAPPED => Err(CudaError::NotMapped),
            CUresult::CUDA_ERROR_NOT_MAPPED_AS_ARRAY => Err(CudaError::NotMappedAsArray),
            CUresult::CUDA_ERROR_NOT_MAPPED_AS_POINTER => Err(CudaError::NotMappedAsPointer),
            CUresult::CUDA_ERROR_ECC_UNCORRECTABLE => Err(CudaError::EccUncorrectable),
            CUresult::CUDA_ERROR_UNSUPPORTED_LIMIT => Err(CudaError::UnsupportedLimit),
            CUresult::CUDA_ERROR_CONTEXT_ALREADY_IN_USE => Err(CudaError::ContextAlreadyInUse),
            CUresult::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => Err(CudaError::PeerAccessUnsupported),
            CUresult::CUDA_ERROR_INVALID_PTX => Err(CudaError::InvalidPtx),
            CUresult::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => Err(CudaError::InvalidGraphicsContext),
            CUresult::CUDA_ERROR_NVLINK_UNCORRECTABLE => Err(CudaError::NvlinkUncorrectable),
            CUresult::CUDA_ERROR_INVALID_SOURCE => Err(CudaError::InvalidSouce),
            CUresult::CUDA_ERROR_FILE_NOT_FOUND => Err(CudaError::FileNotFound),
            CUresult::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => {
                Err(CudaError::SharedObjectSymbolNotFound)
            }
            CUresult::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => {
                Err(CudaError::SharedObjectInitFailed)
            }
            CUresult::CUDA_ERROR_OPERATING_SYSTEM => Err(CudaError::OperatingSystemError),
            CUresult::CUDA_ERROR_INVALID_HANDLE => Err(CudaError::InvalidHandle),
            CUresult::CUDA_ERROR_NOT_FOUND => Err(CudaError::NotFound),
            CUresult::CUDA_ERROR_NOT_READY => Err(CudaError::NotReady),
            CUresult::CUDA_ERROR_ILLEGAL_ADDRESS => Err(CudaError::IllegalAddress),
            CUresult::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => Err(CudaError::LaunchOutOfResources),
            CUresult::CUDA_ERROR_LAUNCH_TIMEOUT => Err(CudaError::LaunchTimeout),
            CUresult::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => {
                Err(CudaError::LaunchIncompatibleTexturing)
            }
            CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => {
                Err(CudaError::PeerAccessAlreadyEnabled)
            }
            CUresult::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => Err(CudaError::PeerAccessNotEnabled),
            CUresult::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => Err(CudaError::PrimaryContextActive),
            CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED => Err(CudaError::ContextIsDestroyed),
            CUresult::CUDA_ERROR_ASSERT => Err(CudaError::AssertError),
            CUresult::CUDA_ERROR_TOO_MANY_PEERS => Err(CudaError::TooManyPeers),
            CUresult::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => {
                Err(CudaError::HostMemoryAlreadyRegistered)
            }
            CUresult::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => {
                Err(CudaError::HostMemoryNotRegistered)
            }
            CUresult::CUDA_ERROR_HARDWARE_STACK_ERROR => Err(CudaError::HardwareStackError),
            CUresult::CUDA_ERROR_ILLEGAL_INSTRUCTION => Err(CudaError::IllegalInstruction),
            CUresult::CUDA_ERROR_MISALIGNED_ADDRESS => Err(CudaError::MisalignedAddress),
            CUresult::CUDA_ERROR_INVALID_ADDRESS_SPACE => Err(CudaError::InvalidAddressSpace),
            CUresult::CUDA_ERROR_INVALID_PC => Err(CudaError::InvalidProgramCounter),
            CUresult::CUDA_ERROR_LAUNCH_FAILED => Err(CudaError::LaunchFailed),
            CUresult::CUDA_ERROR_NOT_PERMITTED => Err(CudaError::NotPermitted),
            CUresult::CUDA_ERROR_NOT_SUPPORTED => Err(CudaError::NotSupported),
            _ => Err(CudaError::UnknownError),
        }?;
        Ok(())
    }
}
