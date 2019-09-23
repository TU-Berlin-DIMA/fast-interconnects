/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019, Clemens Lutz <lutzcle@cml.li>
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

use std::default::Default;
use std::mem::size_of;

use super::memory::{DerefMem, Mem};
use super::numa::{DistributedNumaMemory, NodeRatio, NumaMemory};

/// Heterogeneous memory allocator.
pub struct Allocator;

/// Memory type specifier
///
/// Some memory types cannot be directly accessed on the host, e.g., CudaDevMem.
#[derive(Clone, Debug, PartialEq)]
pub enum MemType {
    /// System memory allocated with Rust's global allocator
    SysMem,
    /// NUMA memory allocated on the specified NUMA node
    NumaMem(u16),
    /// NUMA memory distributed over multiple NUMA nodes
    DistributedNumaMem(Box<[NodeRatio]>),
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
    /// NUMA memory allocated on the specified NUMA node
    NumaMem(u16),
    /// NUMA memory distributed over multiple NUMA nodes
    DistributedNumaMem(Box<[NodeRatio]>),
    /// CUDA pinned memory (using cudaHostAlloc())
    CudaPinnedMem,
    /// CUDA unified memory
    CudaUniMem,
}

impl From<DerefMemType> for MemType {
    fn from(dmt: DerefMemType) -> Self {
        match dmt {
            DerefMemType::SysMem => MemType::SysMem,
            DerefMemType::NumaMem(node) => MemType::NumaMem(node),
            DerefMemType::DistributedNumaMem(nodes) => MemType::DistributedNumaMem(nodes),
            DerefMemType::CudaPinnedMem => MemType::CudaPinnedMem,
            DerefMemType::CudaUniMem => MemType::CudaUniMem,
        }
    }
}

/// Generic memory allocator for Mem that hides concrete memory type
///
/// The intended use-case is when a callee (such as a library) must allocate
/// memory. In this case, the caller can pass in a generic memory allocator
/// This allows the callee to generalize over all memory types.
pub type MemAllocFn<T> = Box<Fn(usize) -> Mem<T>>;

/// Generic memory allocator for DerefMem that hides concrete memory type
///
/// The intended use-case is when a callee (such as a library) must allocate
/// memory. In this case, the caller can pass in a generic memory allocator
/// This allows the callee to generalize over all memory types.
pub type DerefMemAllocFn<T> = Box<Fn(usize) -> DerefMem<T>>;

impl Allocator {
    /// Allocates memory of the specified type
    pub fn alloc_mem<T: Clone + Default + DeviceCopy>(mem_type: MemType, len: usize) -> Mem<T> {
        match mem_type {
            MemType::SysMem => Self::alloc_system(len).into(),
            MemType::NumaMem(node) => Self::alloc_numa(len, node).into(),
            MemType::DistributedNumaMem(nodes) => Self::alloc_distributed_numa(len, nodes).into(),
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
            DerefMemType::NumaMem(node) => Self::alloc_numa(len, node),
            DerefMemType::DistributedNumaMem(nodes) => {
                Self::alloc_distributed_numa(len, nodes).into()
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
            MemType::NumaMem(node) => Box::new(move |len| Self::alloc_numa(len, node).into()),
            MemType::DistributedNumaMem(nodes) => {
                Box::new(move |len| Self::alloc_distributed_numa(len, nodes.clone()).into())
            }
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
            DerefMemType::NumaMem(node) => Box::new(move |len| Self::alloc_numa(len, node)),
            DerefMemType::DistributedNumaMem(nodes) => {
                Box::new(move |len| Self::alloc_distributed_numa(len, nodes.clone()).into())
            }
            DerefMemType::CudaPinnedMem => Box::new(|len| Self::alloc_cuda_pinned(len)),
            DerefMemType::CudaUniMem => Box::new(|len| Self::alloc_cuda_unified(len)),
        }
    }

    /// Allocates system memory using Rust's global allocator.
    fn alloc_system<T: Clone + Default + DeviceCopy>(len: usize) -> DerefMem<T> {
        DerefMem::SysMem(vec![T::default(); len])
    }

    /// Allocates memory on the specified NUMA node.
    fn alloc_numa<T: DeviceCopy>(len: usize, node: u16) -> DerefMem<T> {
        DerefMem::NumaMem(NumaMemory::new(len, node))
    }

    /// Allocates memory on multiple, specified NUMA nodes.
    fn alloc_distributed_numa<T: DeviceCopy>(len: usize, nodes: Box<[NodeRatio]>) -> DerefMem<T> {
        DerefMem::DistributedNumaMem(DistributedNumaMemory::new(len, nodes))
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
    /// Warning: Returns zeroed memory. Zero may not be a valid bit-pattern for
    /// type `T`.
    fn alloc_cuda_device<T: DeviceCopy>(len: usize) -> Mem<T> {
        unsafe {
            Mem::CudaDevMem(DeviceBuffer::<T>::zeroed(len).expect(&format!(
                "Failed to allocate {} bytes of CUDA device memory",
                len * size_of::<T>()
            )))
        }
    }
}
