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

use bitflags::bitflags;

use libc::{mmap, munmap};

use std::io;
use std::io::Error as IoError;
use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::os::raw::{c_int, c_long, c_uint, c_ulong, c_void};
use std::ptr;
use std::slice;

use crate::error::{Error, ErrorKind, Result, ResultExt};
use crate::runtime::cuda_wrapper::{host_register, host_unregister};
use crate::runtime::hw_info::ProcessorCache;
use crate::runtime::memory::PageLock;

use num_rational::Ratio;

#[link(name = "numa")]
extern "C" {
    pub fn numa_run_on_node(node: c_int) -> c_int;
    pub fn numa_set_strict(strict: c_int);
    pub fn numa_alloc_onnode(size: usize, node: c_int) -> *mut c_void;
    pub fn numa_free(start: *mut c_void, size: usize);
    pub fn numa_tonode_memory(start: *mut c_void, size: usize, node: c_int);
    pub fn mbind(
        addr: *mut c_void,
        len: c_ulong,
        mode: c_int,
        nodemask: *const c_ulong,
        maxnode: c_ulong,
        flags: c_uint,
    ) -> c_long;
}

bitflags! {
/// Flags for set_mempolicy
///
/// See `linux/mempolicy.h` for definition.
pub struct MemPolicyFlags: c_uint {
    const DEFAULT = 0x0;
}
}

bitflags! {
/// Flags for `mbind`
///
/// See `numaif.h` for definition.
pub struct MemBindFlags: c_uint {
    /// Default
    const DEFAULT = 0x0;

    /// Verify existing pages in the mapping
    const STRICT = 0x1;

    /// Move pages owned by this process to conform to mapping
    const MOVE = 0x2;

    /// Move every page to conform to mapping
    const MOVE_ALL = 0x4;
}
}

bitflags! {
/// Memory policies
///
/// See `numaif.h` and `linux/mempolicy.h` for definition.
pub struct MemPolicyModes: c_int {
    /// Restores default behavior by removing any previously requested policy.
    const DEFAULT = 0x0;

    /// Specifies that pages should first try to be allocated on a node in the
    /// node mask. If free memory is low on these nodes, the pages will be
    /// allocated on another node.
    const PREFERRED = 0x1;

    /// Specifies a strict policy that restricts memory allocation to the nodes
    /// specified in the node mask. Pages will not be allocated from any node
    /// not in the node mask.
    const BIND = 0x2;

    /// Specifies that page allocations should be interleaved over the nodes in
    /// the node mask. This often leads to higher bandwidth at the cost of higher
    /// latency.
    const INTERLEAVE = 0x3;

    /// Specifies that pages should be allocated locally on the node that
    /// triggers the allocation. If free memory is low on the local node, pages
    /// will be allocated on another node.
    const LOCAL = 0x4;

    /// Specifies that the nodes should be interpreted as physical node IDs.
    /// Thus, the operating system does not reinterpret the node set when the
    /// thread moves to a different cpuset context.
    ///
    /// Optional and cannot be combined with RELATIVE_NODES.
    const STATIC_NODES = (1 << 15);

    /// Speficies that the nodes should be interpreted relative to the thread's
    /// cpuset context.
    ///
    /// Optional and cannot be combined with STATIC_NODES.
    const RELATIVE_NODES = (1 << 14);
}
}

/// CPU set to create CPU and NUMA node masks.
///
/// Inspired by Linux's `cpu_set_t`, see the `cpu_set` manual page.
///
/// Limitations
/// ===========
///
/// The set is currently restricted to a single 64-bit integer. Therefore, IDs
/// must be smaller or equal to 63.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct CpuSet {
    mask: u64,
}

impl CpuSet {
    /// Create an empty CPU set.
    pub fn new() -> Self {
        Self { mask: 0 }
    }

    /// Add an ID to the set.
    pub fn add(&mut self, id: u16) {
        assert!(id <= 63);
        self.mask = self.mask | (1 << id);
    }

    /// Remove an ID from the set.
    pub fn remove(&mut self, id: u16) {
        assert!(id <= 63);
        self.mask = self.mask & !(1 << id);
    }

    /// Query if an ID is included in the set.
    pub fn is_set(&self, id: u16) -> bool {
        assert!(id <= 63);
        (self.mask & (1 << id)) != 0
    }

    /// Returns the number of IDs in the set
    pub fn count(&self) -> usize {
        self.mask.count_ones() as usize
    }

    /// Reset the set to zero.
    pub fn zero(&mut self) {
        self.mask = 0;
    }

    /// Query the maximum amount of nodes currently in the set.
    pub fn max_node(&self) -> u16 {
        64 - self.mask.leading_zeros() as u16
    }

    /// Get the set as a slice.
    pub fn as_slice(&self) -> &[u64] {
        slice::from_ref(&self.mask)
    }
}

impl std::ops::BitAnd for CpuSet {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            mask: self.mask & rhs.mask,
        }
    }
}

impl std::ops::BitOr for CpuSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            mask: self.mask | rhs.mask,
        }
    }
}

impl std::ops::BitXor for CpuSet {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self {
            mask: self.mask ^ rhs.mask,
        }
    }
}

pub fn rust_mbind<T>(
    data: &[T],
    mode: MemPolicyModes,
    nodes: CpuSet,
    flags: MemBindFlags,
) -> Result<()> {
    unsafe {
        if mbind(
            data.as_ptr() as *mut T as *mut c_void,
            (data.len() * size_of::<T>()) as u64,
            mode.bits(),
            &nodes.mask,
            64,
            // nodes.max_node().into(),
            flags.bits(),
        ) == -1
        {
            Err(ErrorKind::Io(IoError::last_os_error()))?;
        }

        Ok(())
    }
}

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
        let pointer = unsafe { numa_alloc_onnode(size, node.into()) } as *mut T;
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
        unsafe { numa_free(self.pointer as *mut c_void, size) };
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
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
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

                    rust_mbind(
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
/// # use numa_gpu::runtime::numa::tonode_memory;
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

#[cfg(test)]
mod test {
    use super::*;
    use libc::{mmap, munmap};
    use std::alloc::{alloc, dealloc, Layout};
    use std::mem::size_of;
    use std::ptr;
    use std::slice;

    #[test]
    fn test_mbind() -> Result<()> {
        let bytes = 2_usize.pow(20);
        let aligned = unsafe {
            let ptr = mmap(
                ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                0,
                0,
            );
            if ptr == libc::MAP_FAILED {
                Err(ErrorKind::Io(IoError::last_os_error()))?;
            }
            slice::from_raw_parts_mut(ptr as *mut usize, bytes / size_of::<usize>())
        };

        let mut nodes = CpuSet::new();
        nodes.add(0);
        rust_mbind(
            aligned,
            MemPolicyModes::PREFERRED,
            nodes,
            MemBindFlags::STRICT,
        )?;

        aligned.iter_mut().enumerate().for_each(|(id, val)| {
            *val = id;
        });

        unsafe {
            if munmap(
                aligned.as_ptr() as *mut usize as *mut libc::c_void,
                aligned.len() * size_of::<usize>(),
            ) == -1
            {
                Err(ErrorKind::Io(IoError::last_os_error()))?;
            }
        }

        Ok(())
    }
}
