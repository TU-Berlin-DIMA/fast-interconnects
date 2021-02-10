//! Collection of Rust-ified wrappers for commonly-used CUDA functions.

use crate::error::{Error, ErrorKind, Result, ToResult};
use crate::runtime::memory::LaunchableMutSlice;
use cuda_sys::cuda::{
    cuCtxGetDevice, cuMemAdvise, cuMemGetInfo_v2, cuMemHostRegister_v2, cuMemHostUnregister,
    cuMemPrefetchAsync, cuMemcpyAsync, cuMemsetD32Async, CUdevice, CUstream,
    CU_MEMHOSTREGISTER_DEVICEMAP, CU_MEMHOSTREGISTER_PORTABLE,
};
use rustacuda::memory::{DeviceCopy, UnifiedPointer};
use rustacuda::stream::Stream;
use std::mem::{size_of, transmute_copy, zeroed};
use std::os::raw::{c_uint, c_void};

// re-export mem_advise enum
pub use cuda_sys::cuda::CUmem_advise_enum as MemAdviseFlags;

/// CUDA memory information
pub struct CudaMemInfo {
    /// Free bytes
    pub free: usize,

    /// Total bytes
    pub total: usize,
}

/// Returns the free and total device memory in bytes
///
/// The result is a tuple: `(free, total)`.
pub fn mem_info() -> Result<CudaMemInfo> {
    let mut free: usize = 0;
    let mut total: usize = 0;

    unsafe {
        cuMemGetInfo_v2(&mut free, &mut total)
            .to_result()
            .map_err(|e| {
                Error::with_chain::<Error, _>(e.into(), "Failed to get memory information")
            })?;
    }

    Ok(CudaMemInfo { free, total })
}

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

/// Fill a launchable slice using the CUDA `memset_async` function
///
/// # Limitations
///
///  - The size of `T` must be an even multiple of a 32-bit integer.
///  - Only fill values of type `i32` are supported. Fill a `T` with an `i32`
///    value might result in an invalid initialization.
pub fn memset_async<T: DeviceCopy>(
    mem: LaunchableMutSlice<T>,
    value: i32,
    stream: &Stream,
) -> Result<()> {
    assert!(
        size_of::<T>() >= size_of::<i32>(),
        "Size of type T must be larger than i32"
    );
    assert!(
        size_of::<T>() % size_of::<i32>() == 0,
        "Size of type T must be divisible by i32"
    );

    unsafe {
        // FIXME: Find a safer solution to replace transmute_copy!!!
        let cu_stream = transmute_copy::<Stream, CUstream>(&stream);

        cuMemsetD32Async(
            mem.as_ptr() as u64,
            value as c_uint,
            mem.len()
                .checked_mul(size_of::<T>() / size_of::<c_uint>())
                .ok_or_else(|| {
                    ErrorKind::IntegerOverflow("Failed to compute memset length".to_string())
                })?,
            cu_stream,
        )
        .to_result()
        .map_err(|e| {
            Error::with_chain::<Error, _>(e.into(), format!("Failed to schedule memset"))
        })?;
    }

    Ok(())
}
