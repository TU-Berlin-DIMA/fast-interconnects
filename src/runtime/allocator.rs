//! Heterogeneous memory allocator.

extern crate accel;

use self::accel::mvec::MVec;
use self::accel::uvec::UVec;

use std::default::Default;
use std::mem::size_of;

use super::backend::NumaMemory;
use super::memory::{DerefMem, Mem};

/// Heterogeneous memory allocator.
///
/// Presents a consistent interface for allocating memory with specific
/// properties. Examples include allocating NUMA-local memory, or allocating
/// CUDA device memory.
///
/// The allocated memory is of type Mem, and specialized to DerefMem whenever
/// possible.
pub struct Allocator;

/// Memory type specifier
///
/// - SysMem:     System memory allocated with Rust's global allocator
/// - NumaMem:    NUMA memory allocated on the specified NUMA node
/// - CudaDevMem: CUDA device memory
/// - CudaUniMem: CUDA unified memory
///
/// Some memory types cannot be directly accessed on the host, e.g., CudaDevMem.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MemType {
    SysMem,
    NumaMem(u16),
    CudaDevMem,
    CudaUniMem,
}

/// Dereferencable memory type specifier
///
/// - SysMem:     System memory allocated with Rust's global allocator
/// - NumaMem:    NUMA memory allocated on the specified NUMA node
/// - CudaUniMem: CUDA unified memory
///
/// These memory types can be directly accessed on the host.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DerefMemType {
    SysMem,
    NumaMem(u16),
    CudaUniMem,
}

impl From<DerefMemType> for MemType {
    fn from(dmt: DerefMemType) -> Self {
        match dmt {
            DerefMemType::SysMem => MemType::SysMem,
            DerefMemType::NumaMem(node) => MemType::NumaMem(node),
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
    pub fn alloc_mem<T: Clone + Copy + Default>(mem_type: MemType, len: usize) -> Mem<T> {
        match mem_type {
            MemType::SysMem => Self::alloc_system(len).into(),
            MemType::NumaMem(node) => Self::alloc_numa(len, node).into(),
            MemType::CudaDevMem => Self::alloc_cuda_device(len),
            MemType::CudaUniMem => Self::alloc_cuda_unified(len).into(),
        }
    }

    /// Allocates host-dereferencable memory of the specified type
    pub fn alloc_deref_mem<T: Clone + Default>(mem_type: DerefMemType, len: usize) -> DerefMem<T> {
        match mem_type {
            DerefMemType::SysMem => Self::alloc_system(len),
            DerefMemType::NumaMem(node) => Self::alloc_numa(len, node),
            DerefMemType::CudaUniMem => Self::alloc_cuda_unified(len),
        }
    }

    /// Returns a generic 'Mem' memory allocator that allocates memory of the
    /// specified 'Mem' type.
    pub fn mem_alloc_fn<T: Clone + Copy + Default>(mem_type: MemType) -> MemAllocFn<T> {
        match mem_type {
            MemType::SysMem => Box::new(|len| Self::alloc_system(len).into()),
            MemType::NumaMem(node) => Box::new(move |len| Self::alloc_numa(len, node).into()),
            MemType::CudaDevMem => Box::new(|len| Self::alloc_cuda_device(len)),
            MemType::CudaUniMem => Box::new(|len| Self::alloc_cuda_unified(len).into()),
        }
    }

    /// Returns a generic 'DerefMem' memory allocator that allocates memory of
    /// the specified 'DerefMem' type.
    pub fn deref_mem_alloc_fn<T: Clone + Default>(mem_type: DerefMemType) -> DerefMemAllocFn<T> {
        match mem_type {
            DerefMemType::SysMem => Box::new(|len| Self::alloc_system(len)),
            DerefMemType::NumaMem(node) => Box::new(move |len| Self::alloc_numa(len, node)),
            DerefMemType::CudaUniMem => Box::new(|len| Self::alloc_cuda_unified(len)),
        }
    }

    /// Allocates system memory using Rust's global allocator.
    fn alloc_system<T: Clone + Default>(len: usize) -> DerefMem<T> {
        DerefMem::SysMem(vec![T::default(); len])
    }

    /// Allocates memory on the specified NUMA node.
    fn alloc_numa<T>(len: usize, node: u16) -> DerefMem<T> {
        DerefMem::NumaMem(NumaMemory::alloc_on_node(len, node))
    }

    /// Allocates CUDA unified memory.
    fn alloc_cuda_unified<T>(len: usize) -> DerefMem<T> {
        DerefMem::CudaUniMem(UVec::<T>::new(len).expect(&format!(
            "Failed dot allocate {} bytes of CUDA unified memory",
            len * size_of::<T>()
        )))
    }

    /// Allocates CUDA device memory.
    ///
    /// Device memory cannot be dereferenced on the host. To access it, use
    /// cudaMemcpy() to copy it to the host.
    fn alloc_cuda_device<T: Copy>(len: usize) -> Mem<T> {
        Mem::CudaDevMem(MVec::<T>::new(len).expect(&format!(
            "Failed to allocate {} bytes of CUDA device memory",
            len * size_of::<T>()
        )))
    }
}