/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
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
use std::cell::RefCell;
use std::convert::TryFrom;
use std::default::Default;
use std::mem::size_of;
use std::rc::Rc;
use std::slice;

use super::hw_info::ProcessorCache;
use super::memory::{DerefMem, Mem, PageLock};
use super::numa::{DistributedNumaMemory, NodeLen, NodeRatio, NumaMemory, PageType};
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
    AlignedSysMem { align_bytes: usize },
    /// NUMA memory allocated on the specified NUMA node and with the specified page type
    NumaMem { node: u16, page_type: PageType },
    /// NUMA memory allocated on the specified NUMA node and pinned with CUDA
    NumaPinnedMem { node: u16, page_type: PageType },
    /// NUMA memory distributed in proportion to a ratio over multiple NUMA nodes
    DistributedNumaMem {
        nodes: Box<[NodeRatio]>,
        page_type: PageType,
    },
    /// NUMA memory distributed over multiple NUMA nodes using a length per node
    DistributedNumaMemWithLen {
        nodes: Box<[NodeLen]>,
        page_type: PageType,
    },
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
    AlignedSysMem { align_bytes: usize },
    /// NUMA memory allocated on the specified NUMA node and with the specified page type
    NumaMem { node: u16, page_type: PageType },
    /// NUMA memory allocated on the specified NUMA node, with the specified page type, and pinned with CUDA
    NumaPinnedMem { node: u16, page_type: PageType },
    /// NUMA memory distributed in proportion to a ratio over multiple NUMA nodes
    DistributedNumaMem {
        nodes: Box<[NodeRatio]>,
        page_type: PageType,
    },
    /// NUMA memory distributed over multiple NUMA nodes using a length per node
    DistributedNumaMemWithLen {
        nodes: Box<[NodeLen]>,
        page_type: PageType,
    },
    /// CUDA pinned memory (using cudaHostAlloc())
    CudaPinnedMem,
    /// CUDA unified memory
    CudaUniMem,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CacheSpillType {
    NoSpill(MemType),
    CacheAndSpill {
        cache_node: u16,
        spill_node: u16,
        page_type: PageType,
    },
}

impl MemType {
    pub fn page_type(&self) -> PageType {
        match *self {
            MemType::NumaMem { page_type, .. } => page_type,
            MemType::NumaPinnedMem { page_type, .. } => page_type,
            MemType::DistributedNumaMem { page_type, .. } => page_type,
            MemType::DistributedNumaMemWithLen { page_type, .. } => page_type,
            MemType::SysMem
            | MemType::AlignedSysMem { .. }
            | MemType::CudaPinnedMem
            | MemType::CudaUniMem => PageType::Default,
            MemType::CudaDevMem => PageType::Default,
        }
    }
}

impl DerefMemType {
    pub fn page_type(&self) -> PageType {
        match *self {
            DerefMemType::NumaMem { page_type, .. } => page_type,
            DerefMemType::NumaPinnedMem { page_type, .. } => page_type,
            DerefMemType::DistributedNumaMem { page_type, .. } => page_type,
            DerefMemType::DistributedNumaMemWithLen { page_type, .. } => page_type,
            DerefMemType::SysMem
            | DerefMemType::AlignedSysMem { .. }
            | DerefMemType::CudaPinnedMem
            | DerefMemType::CudaUniMem => PageType::Default,
        }
    }
}

impl From<DerefMemType> for MemType {
    fn from(dmt: DerefMemType) -> Self {
        match dmt {
            DerefMemType::SysMem => MemType::SysMem,
            DerefMemType::AlignedSysMem { align_bytes } => MemType::AlignedSysMem { align_bytes },
            DerefMemType::NumaMem { node, page_type } => MemType::NumaMem { node, page_type },
            DerefMemType::NumaPinnedMem { node, page_type } => {
                MemType::NumaPinnedMem { node, page_type }
            }
            DerefMemType::DistributedNumaMem { nodes, page_type } => {
                MemType::DistributedNumaMem { nodes, page_type }
            }
            DerefMemType::DistributedNumaMemWithLen { nodes, page_type } => {
                MemType::DistributedNumaMemWithLen { nodes, page_type }
            }
            DerefMemType::CudaPinnedMem => MemType::CudaPinnedMem,
            DerefMemType::CudaUniMem => MemType::CudaUniMem,
        }
    }
}

impl From<MemType> for CacheSpillType {
    fn from(mem_type: MemType) -> CacheSpillType {
        CacheSpillType::NoSpill(mem_type)
    }
}

impl TryFrom<MemType> for DerefMemType {
    type Error = Error;

    fn try_from(mt: MemType) -> Result<Self> {
        match mt {
            MemType::SysMem => Ok(DerefMemType::SysMem),
            MemType::AlignedSysMem { align_bytes } => {
                Ok(DerefMemType::AlignedSysMem { align_bytes })
            }
            MemType::NumaMem { node, page_type } => Ok(DerefMemType::NumaMem { node, page_type }),
            MemType::NumaPinnedMem { node, page_type } => {
                Ok(DerefMemType::NumaPinnedMem { node, page_type })
            }
            MemType::DistributedNumaMem { nodes, page_type } => {
                Ok(DerefMemType::DistributedNumaMem { nodes, page_type })
            }
            MemType::DistributedNumaMemWithLen { nodes, page_type } => {
                Ok(DerefMemType::DistributedNumaMemWithLen { nodes, page_type })
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

/// A curried memory allocator for caching and spilling memory
///
/// Takes as an argument the maximum GPU cache length.
pub type MemSpillAllocFn<T> = Box<dyn Fn(usize) -> MemAllocFn<T>>;

impl Allocator {
    /// Allocates memory of the specified type
    pub fn alloc_mem<T: Clone + Default + DeviceCopy>(mem_type: MemType, len: usize) -> Mem<T> {
        match mem_type {
            MemType::SysMem => Self::alloc_system(len).into(),
            MemType::AlignedSysMem { align_bytes } => Self::alloc_aligned(len, align_bytes).into(),
            MemType::NumaMem { node, page_type } => Self::alloc_numa(len, node, page_type).into(),
            MemType::NumaPinnedMem { node, page_type } => {
                Self::alloc_numa_pinned(len, node, page_type).into()
            }
            MemType::DistributedNumaMem { nodes, page_type } => {
                Self::alloc_distributed_numa(len, nodes, page_type).into()
            }
            MemType::DistributedNumaMemWithLen { nodes, page_type } => {
                Self::alloc_distributed_numa_with_len(len, nodes, page_type).into()
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
            DerefMemType::AlignedSysMem { align_bytes } => {
                Self::alloc_aligned(len, align_bytes).into()
            }
            DerefMemType::NumaMem { node, page_type } => Self::alloc_numa(len, node, page_type),
            DerefMemType::NumaPinnedMem { node, page_type } => {
                Self::alloc_numa_pinned(len, node, page_type).into()
            }
            DerefMemType::DistributedNumaMem { nodes, page_type } => {
                Self::alloc_distributed_numa(len, nodes, page_type).into()
            }
            DerefMemType::DistributedNumaMemWithLen { nodes, page_type } => {
                Self::alloc_distributed_numa_with_len(len, nodes, page_type).into()
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
            MemType::AlignedSysMem { align_bytes } => {
                Box::new(move |len| Self::alloc_aligned(len, align_bytes).into())
            }
            MemType::NumaMem { node, page_type } => {
                Box::new(move |len| Self::alloc_numa(len, node, page_type).into())
            }
            MemType::NumaPinnedMem { node, page_type } => {
                Box::new(move |len| Self::alloc_numa_pinned(len, node, page_type).into())
            }
            MemType::DistributedNumaMem { nodes, page_type } => Box::new(move |len| {
                Self::alloc_distributed_numa(len, nodes.clone(), page_type).into()
            }),
            MemType::DistributedNumaMemWithLen { nodes, page_type } => Box::new(move |len| {
                Self::alloc_distributed_numa_with_len(len, nodes.clone(), page_type).into()
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
            DerefMemType::AlignedSysMem { align_bytes } => {
                Box::new(move |len| Self::alloc_aligned(len, align_bytes).into())
            }
            DerefMemType::NumaMem { node, page_type } => {
                Box::new(move |len| Self::alloc_numa(len, node, page_type))
            }
            DerefMemType::NumaPinnedMem { node, page_type } => {
                Box::new(move |len| Self::alloc_numa_pinned(len, node, page_type))
            }
            DerefMemType::DistributedNumaMem { nodes, page_type } => Box::new(move |len| {
                Self::alloc_distributed_numa(len, nodes.clone(), page_type).into()
            }),
            DerefMemType::DistributedNumaMemWithLen { nodes, page_type } => Box::new(move |len| {
                Self::alloc_distributed_numa_with_len(len, nodes.clone(), page_type).into()
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
    fn alloc_numa<T: DeviceCopy>(len: usize, node: u16, page_type: PageType) -> DerefMem<T> {
        DerefMem::NumaMem(NumaMemory::new(len, node, page_type))
    }

    /// Allocates pinned memory on the specified NUMA node.
    fn alloc_numa_pinned<T: DeviceCopy>(len: usize, node: u16, page_type: PageType) -> DerefMem<T> {
        let mut mem = NumaMemory::new(len, node, page_type);
        mem.page_lock().expect("Failed to pin memory");
        DerefMem::NumaMem(mem)
    }

    /// Allocates memory on multiple, specified NUMA nodes.
    fn alloc_distributed_numa<T: DeviceCopy>(
        len: usize,
        nodes: Box<[NodeRatio]>,
        page_type: PageType,
    ) -> DerefMem<T> {
        DerefMem::DistributedNumaMem(DistributedNumaMemory::new_with_ratio(len, nodes, page_type))
    }

    /// Allocates memory on multiple, specified NUMA nodes.
    fn alloc_distributed_numa_with_len<T: DeviceCopy>(
        len: usize,
        nodes: Box<[NodeLen]>,
        page_type: PageType,
    ) -> DerefMem<T> {
        DerefMem::DistributedNumaMem(DistributedNumaMemory::new_with_len(len, nodes, page_type))
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

    /// Captures the cache memory type and returns a function that returns an allocator
    ///
    /// Effectively, we're gathering the arguments by currying
    /// `mem_spill_alloc_fn_internal`.
    ///
    /// The returned allocator uses GPU memory until `cache_max_len` is reached.
    /// Then, the allocator spills the remainder to CPU memory.
    ///
    /// The functional programming approach used here solves the leaky abstraction
    /// problem. Oftentimes, the knowledge about how much cache space is available,
    /// and the total required space for the data relation reside in two different
    /// code modules. Thus, we aggregate this knowledge in a closure to retain
    /// modularity, instead of leaking internal details of the modules.
    ///
    /// Returns a "future" that is set to the cached length when the allocator is
    /// invoked, for logging purposes.
    pub fn mem_spill_alloc_fn<T>(
        cache_spill_type: CacheSpillType,
    ) -> (MemSpillAllocFn<T>, Rc<RefCell<Option<usize>>>)
    where
        T: Clone + Default + DeviceCopy,
    {
        let cached_len_future = Rc::new(RefCell::new(None));
        let cached_len_setter = cached_len_future.clone();

        let alloc: MemSpillAllocFn<T> = match cache_spill_type {
            CacheSpillType::CacheAndSpill {
                cache_node,
                spill_node,
                page_type,
            } => Box::new(move |cache_max_len| {
                Self::mem_spill_alloc_fn_internal(
                    cache_max_len,
                    cache_node,
                    spill_node,
                    page_type,
                    cached_len_setter.clone(),
                )
            }),
            CacheSpillType::NoSpill(mem_type) => {
                cached_len_setter.replace(None);
                let mem_alloc: MemSpillAllocFn<T> =
                    Box::new(move |_| Self::mem_alloc_fn(mem_type.clone()));
                mem_alloc
            }
        };

        (alloc, cached_len_future)
    }

    /// Builds an allocator that caches data in GPU memory
    fn mem_spill_alloc_fn_internal<T>(
        cache_max_len: usize,
        gpu_node: u16,
        cpu_node: u16,
        page_type: PageType,
        cached_len_setter: Rc<RefCell<Option<usize>>>,
    ) -> MemAllocFn<T>
    where
        T: Clone + Default + DeviceCopy,
    {
        // Round down to the page size. Assume huge pages, as this will also work
        // with small pages, but not vice-versa.
        let page_size = ProcessorCache::huge_page_size().expect("Failed to get the huge page size");
        let cache_max_len = (cache_max_len / page_size) * page_size;

        let alloc = move |len| {
            let (cached_len, spilled_len) = if len <= cache_max_len {
                (len, 0)
            } else {
                (cache_max_len, len - cache_max_len)
            };

            cached_len_setter.replace(Some(cached_len));

            let mem_type = MemType::DistributedNumaMemWithLen {
                nodes: Box::new([
                    NodeLen {
                        node: gpu_node,
                        len: cached_len,
                    },
                    NodeLen {
                        node: cpu_node,
                        len: spilled_len,
                    },
                ]),
                page_type,
            };

            Allocator::alloc_mem(mem_type, len)
        };

        Box::new(alloc)
    }
}
