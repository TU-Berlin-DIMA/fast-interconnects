//! Collection of Rust-ified wrappers for commonly-used CUDA functions.

extern crate cuda_sys;
extern crate rustacuda;

use rustacuda::memory::{DeviceCopy, UnifiedPointer};
use rustacuda::stream::Stream;

use self::cuda_sys::cuda::{
    cuCtxGetDevice, cuMemHostRegister_v2, cuMemHostUnregister, cuMemPrefetchAsync, CUdevice,
    CUstream, CU_MEMHOSTREGISTER_DEVICEMAP, CU_MEMHOSTREGISTER_PORTABLE,
};

use std::mem::{size_of, transmute_copy, zeroed};
use std::os::raw::c_void;

use crate::error::{Error, Result, ToResult};

/// Page-lock an existing memory range for efficient GPU transfers.
///
/// # Unsafety
///
/// Page-locked memory must be unregistered with host_unregister().
pub unsafe fn host_register<T>(mem: &mut [T]) -> Result<()> {
    cuMemHostRegister_v2(
        mem.as_mut_ptr() as *mut c_void,
        mem.len() * size_of::<T>(),
        CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP,
    )
    .to_result()
    .map_err(|e| Error::with_chain::<Error, _>(e.into(), "Failed to dynamically pin memory"))
}

/// Unregisters a memory range that was page-locked with host_register().
///
/// # Unsafety
///
/// Memory range must have been page-locked with host_register().
pub unsafe fn host_unregister<T>(mem: &mut [T]) -> Result<()> {
    cuMemHostUnregister(mem.as_mut_ptr() as *mut c_void)
        .to_result()
        .map_err(|e| {
            Error::with_chain::<Error, _>(
                e.into(),
                "Failed to unregister dynamically pinned memory",
            )
        })
}

/// Prefetch memory to the device specified in the current context.
pub fn prefetch_async<T: DeviceCopy>(
    mem: UnifiedPointer<T>,
    len: usize,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        let mut cu_device: CUdevice = zeroed();
        cuCtxGetDevice(&mut cu_device).to_result().map_err(|e| {
            Error::with_chain::<Error, _>(e.into(), "Failed to get CUDA device in prefetch_async")
        })?;

        // FIXME: Find a safer solution to replace transmute_copy!!!
        let cu_stream = transmute_copy::<Stream, CUstream>(stream);
        cuMemPrefetchAsync(
            mem.as_raw() as *const c_void as u64,
            len * size_of::<T>(),
            cu_device,
            cu_stream,
        )
        .to_result()
        .map_err(|e| {
            Error::with_chain::<Error, _>(
                e.into(),
                format!("Failed to prefetch unified memory to device {}", cu_device),
            )
        })
    }
}
