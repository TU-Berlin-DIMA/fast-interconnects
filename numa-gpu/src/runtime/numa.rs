/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

//! Rust bindings to Linux's 'numa' library.

use super::cuda_wrapper::{host_register, host_unregister};
use super::hw_info::ProcessorCache;
use super::linux_wrapper::{mbind, CpuSet, MemBindFlags, MemPolicyModes};
use super::memory::PageLock;
use crate::error::{ErrorKind, Result, ResultExt};

use libc::{mmap, munmap};

use std::io::Error as IoError;
use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::os::raw::{c_int, c_void};
use std::ptr;
use std::slice;

use num_rational::Ratio;

mod bindings {
    use super::*;

    #[link(name = "numa")]
    extern "C" {
        pub fn numa_alloc_onnode(size: usize, node: c_int) -> *mut c_void;
        pub fn numa_free(start: *mut c_void, size: usize);
    }
}

/// Re-export Linux's NUMA bindings
pub use super::linux_wrapper::{
    numa_run_on_node as run_on_node, numa_set_strict as set_strict,
    numa_tonode_memory as tonode_memory,
};

/// A contiguous memory region that is dynamically allocated on the specified
/// NUMA node.
#[derive(Debug)]
pub struct NumaMemory<T> {
    pointer: *mut T,
    len: usize,
    node: u16,
    is_page_locked: bool,
}

impl<T> NumaMemory<T> {
    /// Allocates a new memory region with the specified capacity on the
    /// specified NUMA node.
    ///
    /// numa_alloc_onnode is currently (as of 2019-05-09, using libnuma-2.0.11)
    /// implemented by allocating memory using mmap() followed by NUMA-binding
    /// it using mbind(). As MMAP_ANONYMOUS allocates pages, it's not necessary
    /// to do manual page alignment.
    pub fn new(len: usize, node: u16) -> Self {
        assert_ne!(len, 0);

        let size = len * size_of::<T>();
        let pointer = unsafe { bindings::numa_alloc_onnode(size, node.into()) } as *mut T;
        if pointer.is_null() {
            panic!("Couldn't allocate memory on NUMA node {}", node);
        }

        Self {
            pointer,
            len,
            node,
            is_page_locked: false,
        }
    }

    /// Extracts a slice of the entire memory region.
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.pointer, self.len) }
    }

    /// Extracts a mutable slice of the entire memory region.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.pointer, self.len) }
    }

    /// Returns the NUMA node that the memory region is allocated on.
    pub fn node(&self) -> u16 {
        self.node
    }
}

impl<T> Deref for NumaMemory<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.pointer, self.len) }
    }
}

impl<T> DerefMut for NumaMemory<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.pointer, self.len) }
    }
}

impl<T> PageLock for NumaMemory<T> {
    fn page_lock(&mut self) -> Result<()> {
        unsafe {
            host_register(self.as_slice()).chain_err(|| {
                ErrorKind::RuntimeError("Failed to page-lock NUMA memory region".to_string())
            })?
        };
        self.is_page_locked = true;
        Ok(())
    }

    fn page_unlock(&mut self) -> Result<()> {
        if self.is_page_locked {
            unsafe {
                host_unregister(self.as_slice())?;
            }
            self.is_page_locked = false;
        }
        Ok(())
    }
}

impl<T> Drop for NumaMemory<T> {
    fn drop(&mut self) {
        // Unregister if memory is page-locked to uphold the invariant.
        // In drop() method, we can only handle error by panicking.
        if self.is_page_locked {
            unsafe {
                host_unregister(self.as_slice()).unwrap();
            }
        }

        let size = self.len * size_of::<T>();
        unsafe { bindings::numa_free(self.pointer as *mut c_void, size) };
    }
}

unsafe impl<T> Send for NumaMemory<T> {}
unsafe impl<T> Sync for NumaMemory<T> {}

/// Specifies the ratio of total memory allocated on the NUMA node
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeRatio {
    /// The NUMA node
    pub node: u16,
    /// A ratio smaller or equal than 1
    pub ratio: Ratio<usize>,
}

/// A contiguous memory region that is dynamically allocated on multiple NUMA
/// nodes.
///
/// The proportion of memory allocated on each node is specified with `NodeRatio`.
/// Therefore, the sum of all ratios always equals one.
#[derive(Debug)]
pub struct DistributedNumaMemory<T> {
    ptr: *mut T,
    len: usize,
    node_ratios: Box<[NodeRatio]>,
    is_page_locked: bool,
}

impl<T> DistributedNumaMemory<T> {
    /// Allocates a new memory region.
    ///
    /// Memory is allocated proportionally on the specified NUMA nodes according
    /// to their ratios. The actual ratios can slightly differ from the specified
    /// ratios, because the allocation granularity is a page.
    ///
    /// Note that the sum of all ratios must equal 1.
    pub fn new(len: usize, node_ratios: Box<[NodeRatio]>) -> Self {
        assert_ne!(len, 0);
        {
            let total: Ratio<usize> = node_ratios.iter().map(|n| n.ratio).sum();
            assert_eq!(total, 1.into());
        }

        // Allocate memory with mmap
        let size = len * size_of::<T>();
        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_LOCKED | libc::MAP_POPULATE,
                0,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            std::result::Result::Err::<(), _>(IoError::last_os_error())
                .expect("Failed to mmap memory");
        }

        // Calculate number of pages, rounded up
        let page_size = ProcessorCache::page_size();
        let pages = (size + page_size - 1) / page_size;

        // Scale the specified ratios by the number of pages and round down to
        // the nearest integer
        let mut scaled_ratios: Box<[NodeRatio]> = node_ratios
            .iter()
            .map(|NodeRatio { node, ratio }| NodeRatio {
                node: *node,
                ratio: (*ratio * pages).trunc(),
            })
            .collect();

        // Ensure that we still account for all pages after rounding down above
        let pages_diff = Ratio::<usize>::from_integer(pages)
            - scaled_ratios
                .iter()
                .map(|node| node.ratio)
                .sum::<Ratio<usize>>();
        if pages_diff != 0.into() {
            scaled_ratios[0].ratio += pages_diff;
        }

        // Bind all pages to their NUMA nodes, and calculate the actual ratios
        let final_node_ratios = scaled_ratios
            .iter()
            .scan(0, |page_offset, NodeRatio { node, ratio }| {
                let old = *page_offset;
                let page_len = ratio.to_integer();
                *page_offset = *page_offset + page_len;
                Some((*node, old, page_len))
            })
            .map(|(node, page_offset, page_len)| {
                let mut node_set = CpuSet::new();
                node_set.add(node);

                unsafe {
                    let slice = slice::from_raw_parts(
                        ptr.add(page_offset as usize * page_size),
                        page_len as usize * page_size,
                    );

                    mbind(
                        slice,
                        MemPolicyModes::PREFERRED,
                        node_set,
                        MemBindFlags::STRICT,
                    )?;
                }

                Ok(NodeRatio {
                    node,
                    ratio: Ratio::<usize>::new(page_len, pages),
                })
            })
            .collect::<Result<Box<[NodeRatio]>>>()
            .expect("Failed to mbind memory to the specified NUMA nodes");

        // Return self
        Self {
            ptr: ptr as *mut T,
            len,
            node_ratios: final_node_ratios,
            is_page_locked: false,
        }
    }

    /// Extracts a slice of the entire memory region.
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Extracts a mutable slice of the entire memory region.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Returns the NUMA nodes that the memory region is allocated on.
    pub fn node_ratios(&self) -> &[NodeRatio] {
        &self.node_ratios
    }
}

impl<T> Drop for DistributedNumaMemory<T> {
    fn drop(&mut self) {
        // Unregister if memory is page-locked to uphold the invariant.
        // In drop() method, we can only handle error by panicking.
        if self.is_page_locked {
            unsafe {
                host_unregister(self.as_slice()).unwrap();
            }
        }

        let size = self.len * size_of::<T>();
        unsafe {
            if munmap(self.ptr as *mut libc::c_void, size) == -1 {
                std::result::Result::Err::<(), _>(IoError::last_os_error())
                    .expect("Failed to munmap memory");
            }
        }
    }
}

unsafe impl<T> Send for DistributedNumaMemory<T> {}
unsafe impl<T> Sync for DistributedNumaMemory<T> {}

impl<T> Deref for DistributedNumaMemory<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> DerefMut for DistributedNumaMemory<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> PageLock for DistributedNumaMemory<T> {
    fn page_lock(&mut self) -> Result<()> {
        unsafe {
            host_register(self.as_slice()).chain_err(|| {
                ErrorKind::RuntimeError("Failed to page-lock NUMA memory region".to_string())
            })?
        };
        self.is_page_locked = true;
        Ok(())
    }

    fn page_unlock(&mut self) -> Result<()> {
        if self.is_page_locked {
            unsafe {
                host_unregister(self.as_slice())?;
            }
            self.is_page_locked = false;
        }
        Ok(())
    }
}
