/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2021 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

//! Heterogeneous memory allocator.
//!
//! Presents a consistent interface for allocating memory with specific
//! properties. Examples include allocating NUMA-local memory, or allocating
//! CUDA device memory.
//!
//! The allocated memory is of type Mem, and specialized to DerefMem whenever
//! possible.

use rustacuda::memory::{DeviceBuffer, DeviceCopy, LockedBuffer, UnifiedBuffer};

use std::alloc::{self, Layout};
use std::convert::TryFrom;
use std::default::Default;
use std::mem::size_of;
use std::slice;

use super::memory::{DerefMem, Mem, PageLock};
use super::numa::{DistributedNumaMemory, NodeLen, NodeRatio, NumaMemory};
use crate::error::{Error, ErrorKind, Result};

/// Heterogeneous memory allocator.
pub struct Allocator;

/// Memory type specifier
///
/// Some memory types cannot be directly accessed on the host, e.g., CudaDevMem.
#[derive(Clone, Debug, PartialEq)]
pub enum MemType {
    /// System memory allocated with Rust's global allocator
    SysMem,
    /// Aligned system memory allocated with Rust's global allocator
    ///
    /// Alignment is specified in bytes.
    AlignedSysMem(usize),
    /// NUMA memory allocated on the specified NUMA node and with the specified huge page option
    NumaMem(u16, Option<bool>),
    /// NUMA memory allocated on the specified NUMA node, with the specified huge page option, and pinned with CUDA
    NumaPinnedMem(u16, Option<bool>),
    /// NUMA memory distributed in proportion to a ratio over multiple NUMA nodes
    DistributedNumaMem(Box<[NodeRatio]>),
    /// NUMA memory distributed over multiple NUMA nodes using a length per node
    DistributedNumaMemWithLen(Box<[NodeLen]>),
    /// CUDA pinned memory (using cudaHostAlloc())
    CudaPinnedMem,
    /// CUDA unified memory
    CudaUniMem,
    /// CUDA device memory
    CudaDevMem,
}

/// Dereferencable memory type specifier
///
/// These memory types can be directly accessed on the host.
#[derive(Clone, Debug, PartialEq)]
pub enum DerefMemType {
    /// System memory allocated with Rust's global allocator
    SysMem,
    /// Aligned system memory allocated with Rust's global allocator
    ///
    /// Alignment is specified in bytes.
    AlignedSysMem(usize),
    /// NUMA memory allocated on the specified NUMA node and with the specified huge page option
    NumaMem(u16, Option<bool>),
    /// NUMA memory allocated on the specified NUMA node, with the specified huge page option, and pinned with CUDA
    NumaPinnedMem(u16, Option<bool>),
    /// NUMA memory distributed in proportion to a ratio over multiple NUMA nodes
    DistributedNumaMem(Box<[NodeRatio]>),
    /// NUMA memory distributed over multiple NUMA nodes using a length per node
    DistributedNumaMemWithLen(Box<[NodeLen]>),
    /// CUDA pinned memory (using cudaHostAlloc())
    CudaPinnedMem,
    /// CUDA unified memory
    CudaUniMem,
}

impl From<DerefMemType> for MemType {
    fn from(dmt: DerefMemType) -> Self {
        match dmt {
            DerefMemType::SysMem => MemType::SysMem,
            DerefMemType::AlignedSysMem(alignment) => MemType::AlignedSysMem(alignment),
            DerefMemType::NumaMem(node, huge_pages) => MemType::NumaMem(node, huge_pages),
            DerefMemType::NumaPinnedMem(node, huge_pages) => {
                MemType::NumaPinnedMem(node, huge_pages)
            }
            DerefMemType::DistributedNumaMem(nodes) => MemType::DistributedNumaMem(nodes),
            DerefMemType::DistributedNumaMemWithLen(nodes) => {
                MemType::DistributedNumaMemWithLen(nodes)
            }
            DerefMemType::CudaPinnedMem => MemType::CudaPinnedMem,
            DerefMemType::CudaUniMem => MemType::CudaUniMem,
        }
    }
}

impl TryFrom<MemType> for DerefMemType {
    type Error = Error;

    fn try_from(mt: MemType) -> Result<Self> {
        match mt {
            MemType::SysMem => Ok(DerefMemType::SysMem),
            MemType::AlignedSysMem(alignment) => Ok(DerefMemType::AlignedSysMem(alignment)),
            MemType::NumaMem(node, huge_pages) => Ok(DerefMemType::NumaMem(node, huge_pages)),
            MemType::NumaPinnedMem(node, huge_pages) => {
                Ok(DerefMemType::NumaPinnedMem(node, huge_pages))
            }
            MemType::DistributedNumaMem(nodes) => Ok(DerefMemType::DistributedNumaMem(nodes)),
            MemType::DistributedNumaMemWithLen(nodes) => {
                Ok(DerefMemType::DistributedNumaMemWithLen(nodes))
            }
            MemType::CudaPinnedMem => Ok(DerefMemType::CudaPinnedMem),
            MemType::CudaUniMem => Ok(DerefMemType::CudaUniMem),
            MemType::CudaDevMem => Err(ErrorKind::InvalidConversion(
                "Cannot convert device memory to &[T] slice",
            )
            .into()),
        }
    }
}

/// Generic memory allocator for Mem that hides concrete memory type
///
/// The intended use-case is when a callee (such as a library) must allocate
/// memory. In this case, the caller can pass in a generic memory allocator
/// This allows the callee to generalize over all memory types.
pub type MemAllocFn<T> = Box<dyn Fn(usize) -> Mem<T>>;

/// Generic memory allocator for DerefMem that hides concrete memory type
///
/// The intended use-case is when a callee (such as a library) must allocate
/// memory. In this case, the caller can pass in a generic memory allocator
/// This allows the callee to generalize over all memory types.
pub type DerefMemAllocFn<T> = Box<dyn Fn(usize) -> DerefMem<T>>;

impl Allocator {
    /// Allocates memory of the specified type
    pub fn alloc_mem<T: Clone + Default + DeviceCopy>(mem_type: MemType, len: usize) -> Mem<T> {
        match mem_type {
            MemType::SysMem => Self::alloc_system(len).into(),
            MemType::AlignedSysMem(alignment) => Self::alloc_aligned(len, alignment).into(),
            MemType::NumaMem(node, huge_pages) => Self::alloc_numa(len, node, huge_pages).into(),
            MemType::NumaPinnedMem(node, huge_pages) => {
                Self::alloc_numa_pinned(len, node, huge_pages).into()
            }
            MemType::DistributedNumaMem(nodes) => Self::alloc_distributed_numa(len, nodes).into(),
            MemType::DistributedNumaMemWithLen(nodes) => {
                Self::alloc_distributed_numa_with_len(len, nodes).into()
            }
            MemType::CudaPinnedMem => Self::alloc_cuda_pinned(len).into(),
            MemType::CudaUniMem => Self::alloc_cuda_unified(len).into(),
            MemType::CudaDevMem => Self::alloc_cuda_device(len),
        }
    }

    /// Allocates host-dereferencable memory of the specified type
    pub fn alloc_deref_mem<T: Clone + Default + DeviceCopy>(
        mem_type: DerefMemType,
        len: usize,
    ) -> DerefMem<T> {
        match mem_type {
            DerefMemType::SysMem => Self::alloc_system(len),
            DerefMemType::AlignedSysMem(alignment) => Self::alloc_aligned(len, alignment).into(),
            DerefMemType::NumaMem(node, huge_pages) => Self::alloc_numa(len, node, huge_pages),
            DerefMemType::NumaPinnedMem(node, huge_pages) => {
                Self::alloc_numa_pinned(len, node, huge_pages).into()
            }
            DerefMemType::DistributedNumaMem(nodes) => {
                Self::alloc_distributed_numa(len, nodes).into()
            }
            DerefMemType::DistributedNumaMemWithLen(nodes) => {
                Self::alloc_distributed_numa_with_len(len, nodes).into()
            }
            DerefMemType::CudaPinnedMem => Self::alloc_cuda_pinned(len),
            DerefMemType::CudaUniMem => Self::alloc_cuda_unified(len),
        }
    }

    /// Returns a generic 'Mem' memory allocator that allocates memory of the
    /// specified 'Mem' type.
    pub fn mem_alloc_fn<T: Clone + Default + DeviceCopy>(mem_type: MemType) -> MemAllocFn<T> {
        match mem_type {
            MemType::SysMem => Box::new(|len| Self::alloc_system(len).into()),
            MemType::AlignedSysMem(alignment) => {
                Box::new(move |len| Self::alloc_aligned(len, alignment).into())
            }
            MemType::NumaMem(node, huge_pages) => {
                Box::new(move |len| Self::alloc_numa(len, node, huge_pages).into())
            }
            MemType::NumaPinnedMem(node, huge_pages) => {
                Box::new(move |len| Self::alloc_numa_pinned(len, node, huge_pages).into())
            }
            MemType::DistributedNumaMem(nodes) => {
                Box::new(move |len| Self::alloc_distributed_numa(len, nodes.clone()).into())
            }
            MemType::DistributedNumaMemWithLen(nodes) => Box::new(move |len| {
                Self::alloc_distributed_numa_with_len(len, nodes.clone()).into()
            }),
            MemType::CudaPinnedMem => Box::new(|len| Self::alloc_cuda_pinned(len).into()),
            MemType::CudaUniMem => Box::new(|len| Self::alloc_cuda_unified(len).into()),
            MemType::CudaDevMem => Box::new(|len| Self::alloc_cuda_device(len)),
        }
    }

    /// Returns a generic 'DerefMem' memory allocator that allocates memory of
    /// the specified 'DerefMem' type.
    pub fn deref_mem_alloc_fn<T: Clone + Default + DeviceCopy>(
        mem_type: DerefMemType,
    ) -> DerefMemAllocFn<T> {
        match mem_type {
            DerefMemType::SysMem => Box::new(|len| Self::alloc_system(len)),
            DerefMemType::AlignedSysMem(alignment) => {
                Box::new(move |len| Self::alloc_aligned(len, alignment).into())
            }
            DerefMemType::NumaMem(node, huge_pages) => {
                Box::new(move |len| Self::alloc_numa(len, node, huge_pages))
            }
            DerefMemType::NumaPinnedMem(node, huge_pages) => {
                Box::new(move |len| Self::alloc_numa_pinned(len, node, huge_pages))
            }
            DerefMemType::DistributedNumaMem(nodes) => {
                Box::new(move |len| Self::alloc_distributed_numa(len, nodes.clone()).into())
            }
            DerefMemType::DistributedNumaMemWithLen(nodes) => Box::new(move |len| {
                Self::alloc_distributed_numa_with_len(len, nodes.clone()).into()
            }),
            DerefMemType::CudaPinnedMem => Box::new(|len| Self::alloc_cuda_pinned(len)),
            DerefMemType::CudaUniMem => Box::new(|len| Self::alloc_cuda_unified(len)),
        }
    }

    /// Allocates system memory using Rust's global allocator.
    fn alloc_system<T: Clone + Default + DeviceCopy>(len: usize) -> DerefMem<T> {
        DerefMem::SysMem(vec![T::default(); len])
    }

    /// Allocates aligned system memory using Rust's global allocator.
    fn alloc_aligned<T: Clone + Default + DeviceCopy>(len: usize, alignment: usize) -> DerefMem<T> {
        let mem = unsafe {
            let layout = Layout::from_size_align(len * size_of::<T>(), alignment)
                .expect("Memory alignment must be at least size of T");
            let ptr = alloc::alloc(layout) as *mut T;
            assert!(!ptr.is_null(), "Failed to allocate aligned memory");

            let slice = slice::from_raw_parts_mut(ptr, len);
            slice.iter_mut().for_each(|x| *x = T::default());

            let output: Box<[T]> = Box::from_raw(slice);
            output
        };
        DerefMem::BoxedSysMem(mem)
    }

    /// Allocates memory on the specified NUMA node.
    fn alloc_numa<T: DeviceCopy>(len: usize, node: u16, huge_pages: Option<bool>) -> DerefMem<T> {
        DerefMem::NumaMem(NumaMemory::new(len, node, huge_pages))
    }

    /// Allocates pinned memory on the specified NUMA node.
    fn alloc_numa_pinned<T: DeviceCopy>(
        len: usize,
        node: u16,
        huge_pages: Option<bool>,
    ) -> DerefMem<T> {
        let mut mem = NumaMemory::new(len, node, huge_pages);
        mem.page_lock().expect("Failed to pin memory");
        DerefMem::NumaMem(mem)
    }

    /// Allocates memory on multiple, specified NUMA nodes.
    fn alloc_distributed_numa<T: DeviceCopy>(len: usize, nodes: Box<[NodeRatio]>) -> DerefMem<T> {
        DerefMem::DistributedNumaMem(DistributedNumaMemory::new_with_ratio(len, nodes))
    }

    /// Allocates memory on multiple, specified NUMA nodes.
    fn alloc_distributed_numa_with_len<T: DeviceCopy>(
        len: usize,
        nodes: Box<[NodeLen]>,
    ) -> DerefMem<T> {
        DerefMem::DistributedNumaMem(DistributedNumaMemory::new_with_len(len, nodes))
    }

    /// Allocates CUDA pinned memory using cudaHostAlloc
    ///
    /// Warning: Returns uninitialized memory. The reason is that CUDA allocates
    /// the memory local to the processor that first touches the memory. This
    /// decision is left to the user.
    fn alloc_cuda_pinned<T: Clone + Default + DeviceCopy>(len: usize) -> DerefMem<T> {
        DerefMem::CudaPinnedMem(LockedBuffer::<T>::new(&T::default(), len).expect(&format!(
            "Failed dot allocate {} bytes of CUDA pinned memory",
            len * size_of::<T>()
        )))
    }

    /// Allocates CUDA unified memory.
    ///
    /// Warning: Returns uninitialized memory. The reason is that CUDA allocates
    /// the memory local to the processor that first touches the memory. This
    /// decision is left to the user.
    fn alloc_cuda_unified<T: Clone + Default + DeviceCopy>(len: usize) -> DerefMem<T> {
        unsafe {
            DerefMem::CudaUniMem(UnifiedBuffer::<T>::uninitialized(len).expect(&format!(
                "Failed dot allocate {} bytes of CUDA unified memory",
                len * size_of::<T>()
            )))
        }
    }

    /// Allocates CUDA device memory.
    ///
    /// Device memory cannot be dereferenced on the host. To access it, use
    /// cudaMemcpy() to copy it to the host.
    ///
    /// Warning: Returns uninitialized memory. The reason is that the allocator
    /// cannot initialize the memory asynchronously, due to the user not
    /// providing a CUDA stream in the API.
    fn alloc_cuda_device<T: DeviceCopy>(len: usize) -> Mem<T> {
        unsafe {
            Mem::CudaDevMem(DeviceBuffer::<T>::uninitialized(len).expect(&format!(
                "Failed to allocate {} bytes of CUDA device memory",
                len * size_of::<T>()
            )))
        }
    }
}
