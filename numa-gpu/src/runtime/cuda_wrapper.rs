//! Collection of Rust-ified wrappers for commonly-used CUDA functions.

extern crate cuda_sys;
extern crate rustacuda;

use rustacuda::memory::{DeviceCopy, UnifiedPointer};
use rustacuda::stream::Stream;

use self::cuda_sys::cuda::{
    cuCtxGetDevice, cuMemAdvise, cuMemHostRegister_v2, cuMemHostUnregister, cuMemPrefetchAsync,
    cuMemcpyAsync, CUdevice, CUstream, CU_MEMHOSTREGISTER_DEVICEMAP, CU_MEMHOSTREGISTER_PORTABLE,
};

use std::mem::{size_of, transmute_copy, zeroed};
use std::os::raw::c_void;

// re-export mem_advise enum
pub use self::cuda_sys::cuda::CUmem_advise_enum as MemAdviseFlags;

use crate::error::{Error, Result, ToResult};

/// Page-lock an existing memory range for efficient GPU transfers.
///
/// # Unsafety
///
/// Page-locked memory must be unregistered with host_unregister().
pub unsafe fn host_register<T>(mem: &[T]) -> Result<()> {
    cuMemHostRegister_v2(
        mem.as_ptr() as *mut c_void,
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
pub unsafe fn host_unregister<T>(mem: &[T]) -> Result<()> {
    cuMemHostUnregister(mem.as_ptr() as *mut c_void)
        .to_result()
        .map_err(|e| {
            Error::with_chain::<Error, _>(
                e.into(),
                "Failed to unregister dynamically pinned memory",
            )
        })
}

pub const CPU_DEVICE_ID: CUdevice = -1;

pub fn current_device_id() -> Result<CUdevice> {
    unsafe {
        let mut cu_device: CUdevice = zeroed();
        cuCtxGetDevice(&mut cu_device).to_result().map_err(|e| {
            Error::with_chain::<Error, _>(e.into(), "Failed to get current CUDA device ID")
        })?;
        Ok(cu_device)
    }
}

/// Prefetch memory to the device specified in the current context.
pub fn prefetch_async<T: DeviceCopy>(
    mem: UnifiedPointer<T>,
    len: usize,
    destination_device: CUdevice,
    stream: &Stream,
) -> Result<()> {
    unsafe {
        // FIXME: Find a safer solution to replace transmute_copy!!!
        let cu_stream = transmute_copy::<Stream, CUstream>(stream);
        cuMemPrefetchAsync(
            mem.as_raw() as *const c_void as u64,
            len * size_of::<T>(),
            destination_device,
            cu_stream,
        )
        .to_result()
        .map_err(|e| {
            Error::with_chain::<Error, _>(
                e.into(),
                format!(
                    "Failed to prefetch unified memory to device {}",
                    destination_device
                ),
            )
        })?;
    }

    Ok(())
}

/// Advise how the memory range will be used.
pub fn mem_advise<T: DeviceCopy>(
    mem: UnifiedPointer<T>,
    len: usize,
    advice: MemAdviseFlags,
    device: CUdevice,
) -> Result<()> {
    unsafe {
        cuMemAdvise(
            mem.as_raw() as *const c_void as u64,
            len * size_of::<T>(),
            advice,
            device,
        )
        .to_result()
        .map_err(|e| {
            Error::with_chain::<Error, _>(e.into(), format!("Failed to advise memory location"))
        })?;
    }

    Ok(())
}

/// Copy a slice using CUDA's memcpyAsync function.
///
/// CUDA infers the type of copy from the underlying pointers. E.g., host-to-host,
/// host-to-device, and so on.
pub fn async_copy<T: DeviceCopy>(dst: &mut [T], src: &[T], stream: &Stream) -> Result<()> {
    assert!(
        src.len() == dst.len(),
        "Source and destination slices have different lengths"
    );

    unsafe {
        // FIXME: Find a safer solution to replace transmute_copy!!!
        let cu_stream = transmute_copy::<Stream, CUstream>(stream);

        let bytes = size_of::<T>() * src.len();
        if bytes != 0 {
            cuMemcpyAsync(
                dst.as_mut_ptr() as u64,
                src.as_ptr() as u64,
                bytes,
                cu_stream,
            )
            .to_result()?
        }
    }

    Ok(())
}
