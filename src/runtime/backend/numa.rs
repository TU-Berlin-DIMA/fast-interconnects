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

use std::io;
use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::os::raw::{c_int, c_void};
use std::slice;
use std::u8;

use super::ProcessorCache;
use crate::error::{Error, ErrorKind, Result, ResultExt};
use crate::runtime::cuda_wrapper::{host_register, host_unregister};
use crate::runtime::memory::PageLock;

#[link(name = "numa")]
extern "C" {
    pub fn numa_run_on_node(node: c_int) -> c_int;
    pub fn numa_set_strict(strict: c_int);
    pub fn numa_alloc_onnode(size: usize, node: c_int) -> *mut c_void;
    pub fn numa_free(start: *mut c_void, size: usize);
    pub fn numa_tonode_memory(start: *mut c_void, size: usize, node: c_int);
}

/// A contiguous memory region that is dynamically allocated on the specified
/// NUMA node.
#[derive(Debug)]
pub struct NumaMemory<T> {
    pointer: *mut T,
    len: usize,
    base_pointer: *mut T,
    alignment: usize,
    node: u16,
    is_page_locked: bool,
}

impl<T> NumaMemory<T> {
    /// Allocates a new memory region with the specified capacity on the
    /// specified NUMA node.
    pub fn alloc_on_node(len: usize, node: u16) -> Self {
        // Get page alignment
        let alignment = ProcessorCache::page_size();

        let size = len * size_of::<T>();
        let base_pointer = unsafe { numa_alloc_onnode(size + alignment, node.into()) } as *mut T;
        if base_pointer.is_null() {
            panic!("Couldn't allocate memory on NUMA node {}", node);
        }

        // Align to page size
        // Note: use std::pointer::align_offset() when in stable branch
        let offset = (base_pointer as usize) & alignment;
        let pointer = unsafe { (base_pointer as *mut u8).add(offset) as *mut T };

        assert!((pointer as usize) & alignment == 0);

        Self {
            pointer,
            len,
            base_pointer,
            alignment,
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
            host_register(self.as_mut_slice()).chain_err(|| {
                ErrorKind::RuntimeError("Failed to page-lock NUMA memory region".to_string())
            })?
        };
        self.is_page_locked = true;
        Ok(())
    }

    fn page_unlock(&mut self) -> Result<()> {
        if self.is_page_locked {
            unsafe {
                host_unregister(self.as_mut_slice())?;
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
                host_unregister(self.as_mut_slice()).unwrap();
            }
        }

        let size = self.len * size_of::<T>();
        unsafe { numa_free(self.base_pointer as *mut c_void, size + self.alignment) };
    }
}

unsafe impl<T> Send for NumaMemory<T> {}
unsafe impl<T> Sync for NumaMemory<T> {}

/// NUMA allocations will fail if the memory cannot be allocated on the target
/// NUMA node. The default behavior is to fall back on other nodes.
pub fn set_strict(strict: bool) {
    unsafe { numa_set_strict(strict.into()) };
}

/// Run the current thread on the specified NUMA node.
pub fn run_on_node(node: u16) -> Result<()> {
    let ret = unsafe { numa_run_on_node(node.into()) };
    if ret == -1 {
        Err(Error::with_chain(
            io::Error::last_os_error(),
            "Couldn't bind thread to CPU node",
        ))?;
    }
    Ok(())
}

/// Put memory on a specific node.
///
/// ```
/// # use numa_gpu::runtime::backend::tonode_memory;
/// let data = vec!(1; 1024);
/// tonode_memory(&data, 0);
/// ```
pub fn tonode_memory<T>(mem: &[T], node: u16) {
    unsafe {
        numa_tonode_memory(
            mem.as_ptr() as *mut T as *mut c_void,
            mem.len() * size_of::<T>(),
            node.into(),
        )
    };
}
