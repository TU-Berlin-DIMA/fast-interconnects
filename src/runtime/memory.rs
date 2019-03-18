/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

extern crate accel;

use self::accel::mvec::MVec;
use self::accel::uvec::UVec;

use std::any::Any;
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
pub enum Mem<T> {
    SysMem(Vec<T>),
    NumaMem(NumaMemory<T>),
    CudaPinnedMem(Vec<T>), // FIXME: allocate with cudaHostAlloc()
    CudaDevMem(MVec<T>),
    CudaUniMem(UVec<T>),
}

impl<T: Any + Copy> Mem<T> {
    pub fn len(&self) -> usize {
        match self {
            SysMem(ref m) => m.len(),
            NumaMem(ref m) => m.len(),
            CudaPinnedMem(ref m) => m.len(),
            CudaDevMem(ref m) => m.len(),
            CudaUniMem(ref m) => m.len(),
        }
    }

    pub fn as_any(&self) -> &Any {
        match self {
            SysMem(ref m) => m as &Any,
            NumaMem(ref m) => m as &Any,
            CudaPinnedMem(ref m) => m as &Any,
            CudaDevMem(ref m) => m as &Any,
            CudaUniMem(ref m) => m as &Any,
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
}

impl<T> From<DerefMem<T>> for Mem<T> {
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
pub enum DerefMem<T> {
    SysMem(Vec<T>),
    NumaMem(NumaMemory<T>),
    CudaPinnedMem(Vec<T>), // FIXME: allocate with cudaHostAlloc()
    CudaUniMem(UVec<T>),
}

impl<T> DerefMem<T> {
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
            DerefMem::CudaUniMem(m) => m.as_slice_mut(),
        }
    }
}

impl<T> Deref for DerefMem<T> {
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

impl<T> DerefMut for DerefMem<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            DerefMem::SysMem(m) => m.as_mut_slice(),
            DerefMem::NumaMem(m) => m.as_mut_slice(),
            DerefMem::CudaPinnedMem(m) => m.as_mut_slice(),
            DerefMem::CudaUniMem(m) => m.as_slice_mut(),
        }
    }
}
