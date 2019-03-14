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

impl Allocator {
    /// Allocate system memory using Rust's global allocator.
    pub fn new_system<T: Clone + Default>(len: usize) -> DerefMem<T> {
        DerefMem::SysMem(vec![T::default(); len])
    }

    /// Allocate memory on the specified NUMA node.
    pub fn new_numa<T>(len: usize, node: u16) -> DerefMem<T> {
        DerefMem::NumaMem(NumaMemory::alloc_on_node(len, node))
    }

    /// Allocate CUDA unified memory on the specified device.
    ///
    /// If the specified device is zero, then the memory will be allocated on
    /// the host.
    pub fn new_cuda_unified<T>(len: usize, device: u32) -> DerefMem<T> {
        DerefMem::CudaUniMem(UVec::<T>::new(len).expect(&format!(
            "Failed dot allocate {} bytes of CUDA unified memory on device {}",
            len * size_of::<T>(),
            device
        )))
    }

    /// Allocate CUDA device memory on the specified device.
    ///
    /// Device memory cannot be dereferenced on the host. To access it, use
    /// cudaMemcpy() to copy it to the host.
    pub fn new_cuda_device<T: Copy>(len: usize, device: u32) -> Mem<T> {
        Mem::CudaDevMem(MVec::<T>::new(len).expect(&format!(
            "Failed to allocate {} bytes of CUDA device memory on device {}",
            len * size_of::<T>(),
            device
        )))
    }
}
