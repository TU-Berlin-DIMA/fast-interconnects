/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use cuda_sys::cuda::cudaError_t;

use error_chain::error_chain;

use rustacuda::error::CudaError;

use rayon::ThreadPoolBuildError;

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
        RayonThreadPoolBuild(ThreadPoolBuildError);
    }
}

/// Converts raw C cudaError_t into Rust-ified Result type
///
/// Copied from Rustacuda, because visibility is constrained to crate scope.
pub trait ToResult {
    fn to_result(self) -> Result<()>;
}

impl ToResult for cudaError_t {
    fn to_result(self) -> Result<()> {
        match self {
            cudaError_t::CUDA_SUCCESS => Ok(()),
            cudaError_t::CUDA_ERROR_INVALID_VALUE => Err(CudaError::InvalidValue),
            cudaError_t::CUDA_ERROR_OUT_OF_MEMORY => Err(CudaError::OutOfMemory),
            cudaError_t::CUDA_ERROR_NOT_INITIALIZED => Err(CudaError::NotInitialized),
            cudaError_t::CUDA_ERROR_DEINITIALIZED => Err(CudaError::Deinitialized),
            cudaError_t::CUDA_ERROR_PROFILER_DISABLED => Err(CudaError::ProfilerDisabled),
            cudaError_t::CUDA_ERROR_PROFILER_NOT_INITIALIZED => {
                Err(CudaError::ProfilerNotInitialized)
            }
            cudaError_t::CUDA_ERROR_PROFILER_ALREADY_STARTED => {
                Err(CudaError::ProfilerAlreadyStarted)
            }
            cudaError_t::CUDA_ERROR_PROFILER_ALREADY_STOPPED => {
                Err(CudaError::ProfilerAlreadyStopped)
            }
            cudaError_t::CUDA_ERROR_NO_DEVICE => Err(CudaError::NoDevice),
            cudaError_t::CUDA_ERROR_INVALID_DEVICE => Err(CudaError::InvalidDevice),
            cudaError_t::CUDA_ERROR_INVALID_IMAGE => Err(CudaError::InvalidImage),
            cudaError_t::CUDA_ERROR_INVALID_CONTEXT => Err(CudaError::InvalidContext),
            cudaError_t::CUDA_ERROR_CONTEXT_ALREADY_CURRENT => {
                Err(CudaError::ContextAlreadyCurrent)
            }
            cudaError_t::CUDA_ERROR_MAP_FAILED => Err(CudaError::MapFailed),
            cudaError_t::CUDA_ERROR_UNMAP_FAILED => Err(CudaError::UnmapFailed),
            cudaError_t::CUDA_ERROR_ARRAY_IS_MAPPED => Err(CudaError::ArrayIsMapped),
            cudaError_t::CUDA_ERROR_ALREADY_MAPPED => Err(CudaError::AlreadyMapped),
            cudaError_t::CUDA_ERROR_NO_BINARY_FOR_GPU => Err(CudaError::NoBinaryForGpu),
            cudaError_t::CUDA_ERROR_ALREADY_ACQUIRED => Err(CudaError::AlreadyAcquired),
            cudaError_t::CUDA_ERROR_NOT_MAPPED => Err(CudaError::NotMapped),
            cudaError_t::CUDA_ERROR_NOT_MAPPED_AS_ARRAY => Err(CudaError::NotMappedAsArray),
            cudaError_t::CUDA_ERROR_NOT_MAPPED_AS_POINTER => Err(CudaError::NotMappedAsPointer),
            cudaError_t::CUDA_ERROR_ECC_UNCORRECTABLE => Err(CudaError::EccUncorrectable),
            cudaError_t::CUDA_ERROR_UNSUPPORTED_LIMIT => Err(CudaError::UnsupportedLimit),
            cudaError_t::CUDA_ERROR_CONTEXT_ALREADY_IN_USE => Err(CudaError::ContextAlreadyInUse),
            cudaError_t::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => {
                Err(CudaError::PeerAccessUnsupported)
            }
            cudaError_t::CUDA_ERROR_INVALID_PTX => Err(CudaError::InvalidPtx),
            cudaError_t::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => {
                Err(CudaError::InvalidGraphicsContext)
            }
            cudaError_t::CUDA_ERROR_NVLINK_UNCORRECTABLE => Err(CudaError::NvlinkUncorrectable),
            cudaError_t::CUDA_ERROR_INVALID_SOURCE => Err(CudaError::InvalidSouce),
            cudaError_t::CUDA_ERROR_FILE_NOT_FOUND => Err(CudaError::FileNotFound),
            cudaError_t::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => {
                Err(CudaError::SharedObjectSymbolNotFound)
            }
            cudaError_t::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => {
                Err(CudaError::SharedObjectInitFailed)
            }
            cudaError_t::CUDA_ERROR_OPERATING_SYSTEM => Err(CudaError::OperatingSystemError),
            cudaError_t::CUDA_ERROR_INVALID_HANDLE => Err(CudaError::InvalidHandle),
            cudaError_t::CUDA_ERROR_NOT_FOUND => Err(CudaError::NotFound),
            cudaError_t::CUDA_ERROR_NOT_READY => Err(CudaError::NotReady),
            cudaError_t::CUDA_ERROR_ILLEGAL_ADDRESS => Err(CudaError::IllegalAddress),
            cudaError_t::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => Err(CudaError::LaunchOutOfResources),
            cudaError_t::CUDA_ERROR_LAUNCH_TIMEOUT => Err(CudaError::LaunchTimeout),
            cudaError_t::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => {
                Err(CudaError::LaunchIncompatibleTexturing)
            }
            cudaError_t::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => {
                Err(CudaError::PeerAccessAlreadyEnabled)
            }
            cudaError_t::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => Err(CudaError::PeerAccessNotEnabled),
            cudaError_t::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => Err(CudaError::PrimaryContextActive),
            cudaError_t::CUDA_ERROR_CONTEXT_IS_DESTROYED => Err(CudaError::ContextIsDestroyed),
            cudaError_t::CUDA_ERROR_ASSERT => Err(CudaError::AssertError),
            cudaError_t::CUDA_ERROR_TOO_MANY_PEERS => Err(CudaError::TooManyPeers),
            cudaError_t::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => {
                Err(CudaError::HostMemoryAlreadyRegistered)
            }
            cudaError_t::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => {
                Err(CudaError::HostMemoryNotRegistered)
            }
            cudaError_t::CUDA_ERROR_HARDWARE_STACK_ERROR => Err(CudaError::HardwareStackError),
            cudaError_t::CUDA_ERROR_ILLEGAL_INSTRUCTION => Err(CudaError::IllegalInstruction),
            cudaError_t::CUDA_ERROR_MISALIGNED_ADDRESS => Err(CudaError::MisalignedAddress),
            cudaError_t::CUDA_ERROR_INVALID_ADDRESS_SPACE => Err(CudaError::InvalidAddressSpace),
            cudaError_t::CUDA_ERROR_INVALID_PC => Err(CudaError::InvalidProgramCounter),
            cudaError_t::CUDA_ERROR_LAUNCH_FAILED => Err(CudaError::LaunchFailed),
            cudaError_t::CUDA_ERROR_NOT_PERMITTED => Err(CudaError::NotPermitted),
            cudaError_t::CUDA_ERROR_NOT_SUPPORTED => Err(CudaError::NotSupported),
            _ => Err(CudaError::UnknownError),
        }?;
        Ok(())
    }
}
