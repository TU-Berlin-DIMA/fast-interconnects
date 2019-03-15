//! Collection of Rust-ified wrappers for commonly-used CUDA functions.

extern crate accel;
extern crate cuda_sys;

use self::accel::error::Check;
use self::accel::uvec::UVec;

use self::cuda_sys::cudart::{
    cudaHostRegister, cudaHostRegisterDefault, cudaHostUnregister, cudaMemPrefetchAsync,
    cudaStream_t,
};

use std::mem::size_of;
use std::os::raw::c_void;

use crate::error::{Error, Result};

/// Page-lock an existing memory range for efficient GPU transfers.
///
/// # Unsafety
///
/// Page-locked memory must be unregistered with host_unregister().
pub unsafe fn host_register<T>(mem: &mut [T]) -> Result<()> {
    cudaHostRegister(
        mem.as_mut_ptr() as *mut c_void,
        mem.len() * size_of::<T>(),
        cudaHostRegisterDefault,
    )
    .check()
    .map_err(|e| Error::with_chain::<Error, _>(e.into(), "Failed to dynamically pin memory"))
}

/// Unregisters a memory range that was page-locked with host_register().
///
/// # Unsafety
///
/// Memory range must have been page-locked with host_register().
pub unsafe fn host_unregister<T>(mem: &mut [T]) -> Result<()> {
    cudaHostUnregister(mem.as_mut_ptr() as *mut c_void)
        .check()
        .map_err(|e| {
            Error::with_chain::<Error, _>(
                e.into(),
                "Failed to unregister dynamically pinned memory",
            )
        })
}

/// Prefetch memory to the specified destination device.
///
/// GPU device must have non-zero identifier. If device is zero, then the memory
/// range will be prefetched to main-memory.
pub fn prefetch_async<T>(mem: &UVec<T>, device: u16, stream: cudaStream_t) -> Result<()> {
    unsafe {
        cudaMemPrefetchAsync(
            mem.as_ptr() as *const c_void,
            mem.len() * size_of::<T>(),
            device.into(),
            stream,
        )
        .check()
        .map_err(|e| {
            Error::with_chain::<Error, _>(
                e.into(),
                format!("Failed to prefetch unified memory to device {}", device),
            )
        })
    }
}
