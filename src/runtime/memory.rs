/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

extern crate rustacuda;

use self::rustacuda::memory::{DeviceBuffer, DeviceCopy, LockedBuffer, UnifiedBuffer, UnifiedPointer, DevicePointer};

use std::ops::Deref;
use std::ops::DerefMut;

use super::backend::NumaMemory;
use crate::error::Result;

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

pub use self::Mem::*;
#[derive(Debug)]
pub enum Mem<T: DeviceCopy> {
    SysMem(Vec<T>),
    NumaMem(NumaMemory<T>),
    CudaPinnedMem(LockedBuffer<T>),
    CudaDevMem(DeviceBuffer<T>),
    CudaUniMem(UnifiedBuffer<T>),
}

impl<T: DeviceCopy> Mem<T> {
    pub fn len(&self) -> usize {
        match self {
            SysMem(ref m) => m.len(),
            NumaMem(ref m) => m.len(),
            CudaPinnedMem(ref m) => m.len(),
            CudaDevMem(ref m) => m.len(),
            CudaUniMem(ref m) => m.len(),
        }
    }

    pub fn as_ptr(&self) -> *const T {
        match self {
            SysMem(m) => m.as_ptr(),
            NumaMem(m) => m.as_ptr(),
            CudaPinnedMem(m) => m.as_ptr(),
            CudaDevMem(m) => m.as_ptr(),
            CudaUniMem(m) => m.as_ptr(),
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        match self {
            SysMem(m) => m.as_mut_ptr(),
            NumaMem(m) => m.as_mut_ptr(),
            CudaPinnedMem(m) => m.as_mut_ptr(),
            CudaDevMem(m) => m.as_mut_ptr(),
            CudaUniMem(m) => m.as_mut_ptr(),
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
            CudaPinnedMem(m) => LaunchableMutPtr(m.as_mut_ptr()),
            CudaDevMem(m) => m.as_device_ptr().into(),
            CudaUniMem(m) => m.as_unified_ptr().into(),
        }
    }
}

impl<T: DeviceCopy> From<DerefMem<T>> for Mem<T> {
    fn from(demem: DerefMem<T>) -> Mem<T> {
        match demem {
            DerefMem::SysMem(m) => Mem::SysMem(m),
            DerefMem::NumaMem(m) => Mem::NumaMem(m),
            DerefMem::CudaPinnedMem(m) => Mem::CudaPinnedMem(m),
            DerefMem::CudaUniMem(m) => Mem::CudaUniMem(m),
        }
    }
}

#[derive(Debug)]
pub enum DerefMem<T: DeviceCopy> {
    SysMem(Vec<T>),
    NumaMem(NumaMemory<T>),
    CudaPinnedMem(LockedBuffer<T>),
    CudaUniMem(UnifiedBuffer<T>),
}

impl<T: DeviceCopy> DerefMem<T> {
    pub fn as_slice(&self) -> &[T] {
        match self {
            DerefMem::SysMem(m) => m.as_slice(),
            DerefMem::NumaMem(m) => m.as_slice(),
            DerefMem::CudaPinnedMem(m) => m.as_slice(),
            DerefMem::CudaUniMem(m) => m.as_slice(),
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            DerefMem::SysMem(m) => m.as_mut_slice(),
            DerefMem::NumaMem(m) => m.as_mut_slice(),
            DerefMem::CudaPinnedMem(m) => m.as_mut_slice(),
            DerefMem::CudaUniMem(m) => m.as_mut_slice(),
        }
    }
}

impl<T: DeviceCopy> Deref for DerefMem<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            DerefMem::SysMem(m) => m.as_slice(),
            DerefMem::NumaMem(m) => m.as_slice(),
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
            DerefMem::CudaPinnedMem(m) => m.as_mut_slice(),
            DerefMem::CudaUniMem(m) => m.as_mut_slice(),
        }
    }
}

/// A pointer to constant memory that can be dereferenced on the GPU.
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
pub struct LaunchablePtr<T: DeviceCopy>(*const T);

unsafe impl<T: DeviceCopy> DeviceCopy for LaunchablePtr<T> {}

impl<T: DeviceCopy> From<UnifiedPointer<T>> for LaunchablePtr<T> {
    fn from(unified_ptr: UnifiedPointer<T>) -> Self {
        Self(unified_ptr.as_raw())
    }
}

impl<T: DeviceCopy> From<DevicePointer<T>> for LaunchablePtr<T> {
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
pub struct LaunchableMutPtr<T: DeviceCopy>(*mut T);

unsafe impl<T: DeviceCopy> DeviceCopy for LaunchableMutPtr<T> {}

impl<T: DeviceCopy> From<UnifiedPointer<T>> for LaunchableMutPtr<T> {
    fn from(mut unified_ptr: UnifiedPointer<T>) -> Self {
        Self(unified_ptr.as_raw_mut())
    }
}

impl<T: DeviceCopy> From<DevicePointer<T>> for LaunchableMutPtr<T> {
    fn from(mut device_ptr: DevicePointer<T>) -> Self {
        Self(device_ptr.as_raw_mut())
    }
}
