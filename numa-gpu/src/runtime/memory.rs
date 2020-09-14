/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use rustacuda::memory::{
    CopyDestination, DeviceBuffer, DeviceCopy, DevicePointer, DeviceSlice, LockedBuffer,
    UnifiedBuffer, UnifiedPointer,
};

use std::convert::{TryFrom, TryInto};
use std::ffi;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ptr;

use super::numa::{DistributedNumaMemory, NumaMemory};
use crate::error::{Error, ErrorKind, Result};

/// A trait for memory that can be page-locked by CUDA.
///
/// Page-locked memory can be copied to the GPU more efficiently.
///
/// # Invariant
///
/// A page-locked memory range must be page-unlocked before it is freed. Thus,
/// the drop() method of the implementation target must call page_unlock().
pub trait PageLock {
    fn page_lock(&mut self) -> Result<()>;
    fn page_unlock(&mut self) -> Result<()>;
}

/// A heterogeneous memory type
///
/// Some memory types cannot be directly accessed on the host, e.g., CudaDevMem.
pub use self::Mem::*;
#[derive(Debug)]
pub enum Mem<T: DeviceCopy> {
    /// System memory allocated with Rust's global allocator
    SysMem(Vec<T>),
    /// NUMA memory allocated on the specified NUMA node
    NumaMem(NumaMemory<T>),
    /// NUMA memory distributed over multiple NUMA nodes
    DistributedNumaMem(DistributedNumaMemory<T>),
    /// CUDA pinned memory (using cudaHostAlloc())
    CudaPinnedMem(LockedBuffer<T>),
    /// CUDA unified memory
    CudaDevMem(DeviceBuffer<T>),
    /// CUDA device memory
    CudaUniMem(UnifiedBuffer<T>),
}

impl<T: DeviceCopy> Mem<T> {
    pub fn len(&self) -> usize {
        match self {
            SysMem(ref m) => m.len(),
            NumaMem(ref m) => m.len(),
            DistributedNumaMem(ref m) => m.len(),
            CudaPinnedMem(ref m) => m.len(),
            CudaDevMem(ref m) => m.len(),
            CudaUniMem(ref m) => m.len(),
        }
    }

    pub fn as_ptr(&self) -> *const T {
        match self {
            SysMem(m) => m.as_ptr(),
            NumaMem(m) => m.as_ptr(),
            DistributedNumaMem(m) => m.as_ptr(),
            CudaPinnedMem(m) => m.as_ptr(),
            CudaDevMem(m) => m.as_ptr(),
            CudaUniMem(m) => m.as_ptr(),
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        match self {
            SysMem(m) => m.as_mut_ptr(),
            NumaMem(m) => m.as_mut_ptr(),
            DistributedNumaMem(m) => m.as_mut_ptr(),
            CudaPinnedMem(m) => m.as_mut_ptr(),
            CudaDevMem(m) => m.as_mut_ptr(),
            CudaUniMem(m) => m.as_mut_ptr(),
        }
    }

    pub fn as_launchable_slice(&self) -> LaunchableSlice<'_, T> {
        // Note: This is implementation is a short-cut. The proper way is
        // implemented in as_launchable_mut_ptr(). The reason we don't do the
        // proper way here is because RUSTACuda doesn't have a *const T
        // equivalent for UnifiedPointer and DevicePointer.
        unsafe { LaunchableSlice(std::slice::from_raw_parts(self.as_ptr(), self.len())) }
    }

    pub fn as_launchable_mut_slice(&mut self) -> LaunchableMutSlice<'_, T> {
        // Note: This is implementation is a short-cut. The proper way is
        // implemented in as_launchable_mut_ptr(). The reason we don't do the
        // proper way here is because RUSTACuda doesn't have a *const T
        // equivalent for UnifiedPointer and DevicePointer.
        unsafe {
            LaunchableMutSlice(std::slice::from_raw_parts_mut(
                self.as_mut_ptr(),
                self.len(),
            ))
        }
    }

    pub fn as_launchable_ptr(&self) -> LaunchablePtr<T> {
        // Note: This is implementation is a short-cut. The proper way is
        // implemented in as_launchable_mut_ptr(). The reason we don't do the
        // proper way here is because RUSTACuda doesn't have a *const T
        // equivalent for UnifiedPointer and DevicePointer.
        LaunchablePtr(self.as_ptr())
    }

    pub fn as_launchable_mut_ptr(&mut self) -> LaunchableMutPtr<T> {
        match self {
            SysMem(m) => LaunchableMutPtr(m.as_mut_ptr()),
            NumaMem(m) => LaunchableMutPtr(m.as_mut_ptr()),
            DistributedNumaMem(m) => LaunchableMutPtr(m.as_mut_ptr()),
            CudaPinnedMem(m) => LaunchableMutPtr(m.as_mut_ptr()),
            CudaDevMem(m) => m.as_device_ptr().into(),
            CudaUniMem(m) => m.as_unified_ptr().into(),
        }
    }
}

impl<T: Copy + DeviceCopy> Mem<T> {
    pub fn copy_from_mem(&mut self, src: &Self) -> Result<()> {
        assert!(
            self.len() == src.len(),
            "Copy destination length must be equal to source"
        );

        let mut convert_result: std::result::Result<&'_ mut [T], (Error, &'_ mut DeviceBuffer<T>)> =
            self.try_into();

        match convert_result {
            Ok(ref mut demem) => {
                match src.try_into() {
                    Ok(src_slice) => demem[..].copy_from_slice(src_slice),
                    Err((_, src_dev_mem)) => src_dev_mem.copy_to(demem)?,
                };
            }
            Err((_, dst_dev_mem)) => {
                match src.try_into() {
                    Ok(src_slice) => {
                        let _annotate_type: &[T] = src_slice;
                        dst_dev_mem.copy_from(src_slice)?
                    }
                    Err((_, src_dev_mem)) => {
                        let _annotate_type: &DeviceBuffer<T> = src_dev_mem;
                        dst_dev_mem.copy_from(&src_dev_mem[..])?
                    }
                };
            }
        };

        Ok(())
    }
}

impl<'t, T: DeviceCopy> TryInto<&'t [T]> for &'t Mem<T> {
    type Error = (Error, &'t DeviceBuffer<T>);

    fn try_into(self) -> std::result::Result<&'t [T], Self::Error> {
        match self {
            Mem::SysMem(m) => Ok(m.as_slice()),
            Mem::NumaMem(m) => Ok(m.as_slice()),
            Mem::DistributedNumaMem(m) => Ok(m.as_slice()),
            Mem::CudaPinnedMem(m) => Ok(m.as_slice()),
            Mem::CudaUniMem(m) => Ok(m.as_slice()),
            Mem::CudaDevMem(m) => Err((
                ErrorKind::InvalidConversion("Cannot convert device memory to &[T] slice").into(),
                m,
            )),
        }
    }
}

impl<'t, T: DeviceCopy> TryInto<&'t mut [T]> for &'t mut Mem<T> {
    type Error = (Error, &'t mut DeviceBuffer<T>);

    fn try_into(self) -> std::result::Result<&'t mut [T], Self::Error> {
        match self {
            Mem::SysMem(m) => Ok(m.as_mut_slice()),
            Mem::NumaMem(m) => Ok(m.as_mut_slice()),
            Mem::DistributedNumaMem(m) => Ok(m.as_mut_slice()),
            Mem::CudaPinnedMem(m) => Ok(m.as_mut_slice()),
            Mem::CudaUniMem(m) => Ok(m.as_mut_slice()),
            Mem::CudaDevMem(m) => Err((
                ErrorKind::InvalidConversion("Cannot convert device memory to &mut [T] slice")
                    .into(),
                m,
            )),
        }
    }
}

impl<T: DeviceCopy> From<DerefMem<T>> for Mem<T> {
    fn from(demem: DerefMem<T>) -> Mem<T> {
        match demem {
            DerefMem::SysMem(m) => Mem::SysMem(m),
            DerefMem::NumaMem(m) => Mem::NumaMem(m),
            DerefMem::DistributedNumaMem(m) => Mem::DistributedNumaMem(m),
            DerefMem::CudaPinnedMem(m) => Mem::CudaPinnedMem(m),
            DerefMem::CudaUniMem(m) => Mem::CudaUniMem(m),
        }
    }
}

/// A CPU-dereferencable memory type
///
/// These memory types can be directly accessed on the host.
#[derive(Debug)]
pub enum DerefMem<T: DeviceCopy> {
    /// System memory allocated with Rust's global allocator
    SysMem(Vec<T>),
    /// NUMA memory allocated on the specified NUMA node
    NumaMem(NumaMemory<T>),
    /// NUMA memory distributed over multiple NUMA nodes
    DistributedNumaMem(DistributedNumaMemory<T>),
    /// CUDA pinned memory (using cudaHostAlloc())
    CudaPinnedMem(LockedBuffer<T>),
    /// CUDA unified memory
    CudaUniMem(UnifiedBuffer<T>),
}

impl<T: DeviceCopy> DerefMem<T> {
    pub fn as_slice(&self) -> &[T] {
        match self {
            DerefMem::SysMem(m) => m.as_slice(),
            DerefMem::NumaMem(m) => m.as_slice(),
            DerefMem::DistributedNumaMem(m) => m.as_slice(),
            DerefMem::CudaPinnedMem(m) => m.as_slice(),
            DerefMem::CudaUniMem(m) => m.as_slice(),
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            DerefMem::SysMem(m) => m.as_mut_slice(),
            DerefMem::NumaMem(m) => m.as_mut_slice(),
            DerefMem::DistributedNumaMem(m) => m.as_mut_slice(),
            DerefMem::CudaPinnedMem(m) => m.as_mut_slice(),
            DerefMem::CudaUniMem(m) => m.as_mut_slice(),
        }
    }

    pub fn as_launchable_slice(&self) -> LaunchableSlice<'_, T> {
        // Note: This is implementation is a short-cut. The proper way is
        // implemented in as_launchable_mut_ptr(). The reason we don't do the
        // proper way here is because RUSTACuda doesn't have a *const T
        // equivalent for UnifiedPointer and DevicePointer.
        unsafe { LaunchableSlice(std::slice::from_raw_parts(self.as_ptr(), self.len())) }
    }

    pub fn as_launchable_mut_slice(&mut self) -> LaunchableMutSlice<'_, T> {
        // Note: This is implementation is a short-cut. The proper way is
        // implemented in as_launchable_mut_ptr(). The reason we don't do the
        // proper way here is because RUSTACuda doesn't have a *const T
        // equivalent for UnifiedPointer and DevicePointer.
        unsafe {
            LaunchableMutSlice(std::slice::from_raw_parts_mut(
                self.as_mut_ptr(),
                self.len(),
            ))
        }
    }

    pub fn as_launchable_ptr(&self) -> LaunchablePtr<T> {
        // Note: This is implementation is a short-cut. The proper way is
        // implemented in as_launchable_mut_ptr(). The reason we don't do the
        // proper way here is because RUSTACuda doesn't have a *const T
        // equivalent for UnifiedPointer and DevicePointer.
        LaunchablePtr(self.as_ptr())
    }

    pub fn as_launchable_mut_ptr(&mut self) -> LaunchableMutPtr<T> {
        match self {
            Self::SysMem(m) => LaunchableMutPtr(m.as_mut_ptr()),
            Self::NumaMem(m) => LaunchableMutPtr(m.as_mut_ptr()),
            Self::DistributedNumaMem(m) => LaunchableMutPtr(m.as_mut_ptr()),
            Self::CudaPinnedMem(m) => LaunchableMutPtr(m.as_mut_ptr()),
            Self::CudaUniMem(m) => m.as_unified_ptr().into(),
        }
    }
}

impl<T: DeviceCopy> Deref for DerefMem<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            DerefMem::SysMem(m) => m.as_slice(),
            DerefMem::NumaMem(m) => m.as_slice(),
            DerefMem::DistributedNumaMem(m) => m.as_slice(),
            DerefMem::CudaPinnedMem(m) => m.as_slice(),
            DerefMem::CudaUniMem(m) => m.as_slice(),
        }
    }
}

impl<T: DeviceCopy> DerefMut for DerefMem<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            DerefMem::SysMem(m) => m.as_mut_slice(),
            DerefMem::NumaMem(m) => m.as_mut_slice(),
            DerefMem::DistributedNumaMem(m) => m.as_mut_slice(),
            DerefMem::CudaPinnedMem(m) => m.as_mut_slice(),
            DerefMem::CudaUniMem(m) => m.as_mut_slice(),
        }
    }
}

impl<T: DeviceCopy> TryFrom<Mem<T>> for DerefMem<T> {
    type Error = (Error, Mem<T>);

    fn try_from(mem: Mem<T>) -> std::result::Result<Self, Self::Error> {
        match mem {
            Mem::SysMem(m) => Ok(DerefMem::SysMem(m)),
            Mem::NumaMem(m) => Ok(DerefMem::NumaMem(m)),
            Mem::DistributedNumaMem(m) => Ok(DerefMem::DistributedNumaMem(m)),
            Mem::CudaPinnedMem(m) => Ok(DerefMem::CudaPinnedMem(m)),
            Mem::CudaUniMem(m) => Ok(DerefMem::CudaUniMem(m)),
            Mem::CudaDevMem(_) => Err((
                ErrorKind::InvalidConversion("Cannot convert device memory to DerefMem").into(),
                mem,
            )),
        }
    }
}

/// GPU-accessible memory.
///
/// By implementing `LaunchableMem` for a type, you specify that the memory can
/// be directly accessed on the GPU.
pub trait LaunchableMem {
    /// The type of elements stored in the memory range.
    type Item;

    /// Returns a launchable pointer to the beginning of the memory range.
    fn as_launchable_ptr(&self) -> LaunchablePtr<Self::Item>;

    /// Returns a launchable slice to the entire memory range.
    fn as_launchable_slice(&self) -> LaunchableSlice<'_, Self::Item>;

    /// Returns a launchable mutable pointer to the beginning of the memory range.
    fn as_launchable_mut_ptr(&mut self) -> LaunchableMutPtr<Self::Item>;

    /// Returns a launchable mutable slice to the entire memory range.
    fn as_launchable_mut_slice(&mut self) -> LaunchableMutSlice<'_, Self::Item>;
}

/// Directly derefencing a main-memory slice on the GPU requires that the GPU
/// has cache-coherent access to main-memory. For example, on POWER9 and Tesla
/// V100 with NVLink 2.0.
///
/// On non-cache-coherent GPUs, derefencing main-memory will lead to a
/// segmentation fault!
impl<'a, T> LaunchableMem for [T] {
    type Item = T;

    fn as_launchable_ptr(&self) -> LaunchablePtr<T> {
        LaunchablePtr(self.as_ptr())
    }

    fn as_launchable_slice(&self) -> LaunchableSlice<'_, T> {
        LaunchableSlice(self)
    }

    fn as_launchable_mut_ptr(&mut self) -> LaunchableMutPtr<T> {
        LaunchableMutPtr(self.as_mut_ptr())
    }

    fn as_launchable_mut_slice(&mut self) -> LaunchableMutSlice<'_, T> {
        LaunchableMutSlice(self)
    }
}

impl<'a, T> LaunchableMem for DeviceBuffer<T> {
    type Item = T;

    fn as_launchable_ptr(&self) -> LaunchablePtr<T> {
        LaunchablePtr(self.as_ptr())
    }

    fn as_launchable_slice(&self) -> LaunchableSlice<'_, T> {
        unsafe { LaunchableSlice(std::slice::from_raw_parts(self.as_ptr(), self.len())) }
    }

    fn as_launchable_mut_ptr(&mut self) -> LaunchableMutPtr<T> {
        LaunchableMutPtr(self.as_mut_ptr())
    }

    fn as_launchable_mut_slice(&mut self) -> LaunchableMutSlice<'_, T> {
        unsafe {
            LaunchableMutSlice(std::slice::from_raw_parts_mut(
                self.as_mut_ptr(),
                self.len(),
            ))
        }
    }
}

impl<'a, T> LaunchableMem for DeviceSlice<T> {
    type Item = T;

    fn as_launchable_ptr(&self) -> LaunchablePtr<T> {
        LaunchablePtr(self.as_ptr())
    }

    fn as_launchable_slice(&self) -> LaunchableSlice<'_, T> {
        unsafe { LaunchableSlice(std::slice::from_raw_parts(self.as_ptr(), self.len())) }
    }

    fn as_launchable_mut_ptr(&mut self) -> LaunchableMutPtr<T> {
        LaunchableMutPtr(self.as_mut_ptr())
    }

    fn as_launchable_mut_slice(&mut self) -> LaunchableMutSlice<'_, T> {
        unsafe {
            LaunchableMutSlice(std::slice::from_raw_parts_mut(
                self.as_mut_ptr(),
                self.len(),
            ))
        }
    }
}

impl<'a, T: DeviceCopy> LaunchableMem for UnifiedBuffer<T> {
    type Item = T;

    fn as_launchable_ptr(&self) -> LaunchablePtr<T> {
        LaunchablePtr(self.as_ptr())
    }

    fn as_launchable_slice(&self) -> LaunchableSlice<'_, T> {
        unsafe { LaunchableSlice(std::slice::from_raw_parts(self.as_ptr(), self.len())) }
    }

    fn as_launchable_mut_ptr(&mut self) -> LaunchableMutPtr<T> {
        LaunchableMutPtr(self.as_mut_ptr())
    }

    fn as_launchable_mut_slice(&mut self) -> LaunchableMutSlice<'_, T> {
        unsafe {
            LaunchableMutSlice(std::slice::from_raw_parts_mut(
                self.as_mut_ptr(),
                self.len(),
            ))
        }
    }
}

/// A pointer to immutable memory that can be dereferenced on the GPU.
///
/// `LaunchablePtr` is intended to be passed as an argument to a CUDA kernel
/// function. For example, it can be passed to RUSTACuda's `launch!()` macro.
///
/// `LaunchablePtr` must not be dereferenced on the CPU, as it may point to
/// device memory.
///
/// `LaunchablePtr` is guaranteed to have an equivalent internal
/// representation to a raw pointer. Thus, it can be safely reinterpreted or
/// transmuted to `*const T`.
#[repr(C)]
#[derive(Copy, Debug, Eq, Hash, PartialEq)]
pub struct LaunchablePtr<T>(*const T);

impl<T: DeviceCopy> LaunchablePtr<T> {
    /// Creates a null launchable pointer.
    pub fn null() -> Self {
        Self(ptr::null())
    }

    /// Cast internal pointer to void pointer.
    pub fn as_void(&self) -> LaunchablePtr<ffi::c_void> {
        LaunchablePtr(self.0 as *const ffi::c_void)
    }

    /// Calculates the offset from a pointer
    pub unsafe fn offset(self, count: isize) -> LaunchablePtr<T> {
        Self(self.0.offset(count))
    }
}

/// Implement Clone trait to support cloning `std::ffi::c_void` pointers.
/// `c_void` doesn't implement Clone, thus can't be derived automatically.
impl<T> Clone for LaunchablePtr<T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

unsafe impl<T: DeviceCopy> DeviceCopy for LaunchablePtr<T> {}

impl<T: DeviceCopy> From<UnifiedPointer<T>> for LaunchablePtr<T> {
    fn from(unified_ptr: UnifiedPointer<T>) -> Self {
        Self(unified_ptr.as_raw())
    }
}

impl<T> From<DevicePointer<T>> for LaunchablePtr<T> {
    fn from(device_ptr: DevicePointer<T>) -> Self {
        Self(device_ptr.as_raw())
    }
}

/// A pointer to mutable memory that can be dereferenced on the GPU.
///
/// `LaunchableMutPtr` is intended to be passed as an argument to a CUDA kernel
/// function. For example, it can be passed to RUSTACuda's `launch!()` macro.
///
/// `LaunchableMutPtr` must not be dereferenced on the CPU, as it may point to
/// device memory.
///
/// `LaunchableMutPtr` is guaranteed to have an equivalent internal
/// representation to a raw pointer. Thus, it can be safely reinterpreted or
/// transmuted to `*mut T`.
#[repr(C)]
#[derive(Copy, Debug, Eq, Hash, PartialEq)]
pub struct LaunchableMutPtr<T>(*mut T);

impl<T: DeviceCopy> LaunchableMutPtr<T> {
    /// Creates a null launchable mutable pointer.
    pub fn null_mut() -> Self {
        Self(ptr::null_mut())
    }

    /// Cast internal pointer to void pointer.
    pub fn as_void(&self) -> LaunchableMutPtr<ffi::c_void> {
        LaunchableMutPtr(self.0 as *mut ffi::c_void)
    }

    /// Calculates the offset from a pointer
    pub unsafe fn offset(self, count: isize) -> LaunchableMutPtr<T> {
        Self(self.0.offset(count))
    }
}

/// Implement Clone trait to support cloning `std::ffi::c_void` pointers.
/// `c_void` doesn't implement Clone, thus can't be derived automatically.
impl<T> Clone for LaunchableMutPtr<T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

unsafe impl<T: DeviceCopy> DeviceCopy for LaunchableMutPtr<T> {}

impl<T: DeviceCopy> From<UnifiedPointer<T>> for LaunchableMutPtr<T> {
    fn from(mut unified_ptr: UnifiedPointer<T>) -> Self {
        Self(unified_ptr.as_raw_mut())
    }
}

impl<T> From<DevicePointer<T>> for LaunchableMutPtr<T> {
    fn from(mut device_ptr: DevicePointer<T>) -> Self {
        Self(device_ptr.as_raw_mut())
    }
}

/// A slice of immutable memory that can be dereferenced on the GPU.
///
/// `LaunchableSlice` is intended to be passed to a function that executes a
/// CUDA kernel with the slice as input parameter.
#[derive(Debug)]
pub struct LaunchableSlice<'a, T>(&'a [T]);

unsafe impl<'a, T: DeviceCopy> DeviceCopy for LaunchableSlice<'a, T> {}

impl<'a, T> LaunchableSlice<'a, T> {
    /// Returns the length of the slice.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns a launchable pointer to the beginning of the slice.
    pub fn as_launchable_ptr(&self) -> LaunchablePtr<T> {
        LaunchablePtr(self.0.as_ptr())
    }

    /// Returns a regular `slice`.
    ///
    /// This is unsafe because dereferencing on the CPU might lead to a segfault.
    pub unsafe fn as_slice(&self) -> &'a [T] {
        self.0
    }
}

/// A slice of mutable memory that can be dereferenced on the GPU.
///
/// `LaunchableMutSlice` is intended to be passed to a function that executes a
/// CUDA kernel with the slice as input parameter.
#[derive(Debug)]
pub struct LaunchableMutSlice<'a, T>(&'a mut [T]);

unsafe impl<'a, T: DeviceCopy> DeviceCopy for LaunchableMutSlice<'a, T> {}

impl<'a, T> LaunchableMutSlice<'a, T> {
    /// Returns the length of the slice.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns a launchable pointer to the beginning of the slice.
    pub fn as_launchable_mut_ptr(&mut self) -> LaunchableMutPtr<T> {
        LaunchableMutPtr(self.0.as_mut_ptr())
    }

    /// Returns a regular mutable `slice`.
    ///
    /// This is unsafe because dereferencing on the CPU might lead to a segfault.
    pub unsafe fn as_mut_slice(self) -> &'a mut [T] {
        self.0
    }
}
