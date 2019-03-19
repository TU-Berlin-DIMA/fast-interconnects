//! Collection of Rust-ified wrappers for commonly-used CUDA functions.

extern crate cuda_sys;
extern crate rustacuda;

use rustacuda::memory::{DeviceCopy, UnifiedBuffer};
use rustacuda::stream::Stream;

use self::cuda_sys::cuda::{
    cuMemHostRegister_v2, cuMemHostUnregister, cuMemPrefetchAsync, CUstream,
    CU_MEMHOSTREGISTER_DEVICEMAP, CU_MEMHOSTREGISTER_PORTABLE,
};

use std::mem::size_of;
use std::mem::transmute_copy;
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

/// Prefetch memory to the specified destination device.
///
/// GPU device must have non-zero identifier. If device is zero, then the memory
/// range will be prefetched to main-memory.
pub fn prefetch_async<T: DeviceCopy>(
    mem: &UnifiedBuffer<T>,
    device: u16,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        // FIXME: Find a safer solution to replace transmute_copy!!!
        let cu_stream = transmute_copy::<Stream, CUstream>(stream);
        cuMemPrefetchAsync(
            mem.as_ptr() as *const c_void as u64,
            mem.len() * size_of::<T>(),
            device.into(),
            cu_stream,
        )
        .to_result()
        .map_err(|e| {
            Error::with_chain::<Error, _>(
                e.into(),
                format!("Failed to prefetch unified memory to device {}", device),
            )
        })
    }
}
