/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use crate::error::{Error, ErrorKind, Result};
use bitflags::bitflags;
use std::io;
use std::io::Error as IoError;
use std::mem::size_of;
use std::os::raw::{c_int, c_long, c_uint, c_ulong, c_void};

mod bindings {
    use super::*;

    #[link(name = "numa")]
    extern "C" {
        pub fn mbind(
            addr: *mut c_void,
            len: c_ulong,
            mode: c_int,
            nodemask: *const c_ulong,
            maxnode: c_ulong,
            flags: c_uint,
        ) -> c_long;
        pub fn numa_run_on_node(node: c_int) -> c_int;
        pub fn numa_set_strict(strict: c_int);
        pub fn numa_tonode_memory(start: *mut c_void, size: usize, node: c_int);
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
    mask: [u64; 4],
}

impl CpuSet {
    const MAX_LEN: u16 = 256;
    const ENTRY_LEN: u16 = 64;

    /// Create an empty CPU set.
    pub fn new() -> Self {
        Self { mask: [0; 4] }
    }

    /// Add an ID to the set.
    pub fn add(&mut self, id: u16) {
        assert!(id < Self::MAX_LEN);

        let entry = &mut self.mask[(id / Self::ENTRY_LEN) as usize];
        let pos = id % Self::ENTRY_LEN;
        *entry = *entry | (1 << pos);
    }

    /// Remove an ID from the set.
    pub fn remove(&mut self, id: u16) {
        assert!(id < Self::MAX_LEN);

        let entry = &mut self.mask[(id / Self::ENTRY_LEN) as usize];
        let pos = id % Self::ENTRY_LEN;
        *entry = *entry & !(1 << pos);
    }

    /// Query if an ID is included in the set.
    pub fn is_set(&self, id: u16) -> bool {
        assert!(id < Self::MAX_LEN);

        let entry = &self.mask[(id / Self::ENTRY_LEN) as usize];
        let pos = id % Self::ENTRY_LEN;
        (*entry & (1 << pos)) != 0
    }

    /// Returns the number of IDs in the set
    pub fn count(&self) -> usize {
        self.mask.iter().map(|e| e.count_ones() as usize).sum()
    }

    /// Reset the set to zero.
    pub fn zero(&mut self) {
        self.mask = [0; 4];
    }

    /// Query the maximum possible number of IDs currently in the set.
    pub fn max_id(&self) -> u16 {
        let leading_zeros: u16 = self
            .mask
            .iter()
            .rev()
            .scan(true, |take_next, e| {
                if *take_next {
                    let lzs = e.leading_zeros() as u16;
                    if lzs != Self::ENTRY_LEN {
                        *take_next = false;
                    }
                    Some(lzs)
                } else {
                    None
                }
            })
            .sum();

        Self::MAX_LEN - leading_zeros + 1
    }

    /// Get the set as a slice.
    pub fn as_slice(&self) -> &[u64] {
        &self.mask
    }
}

impl std::ops::BitAnd for CpuSet {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        let mut mask = self.mask.clone();
        mask.iter_mut()
            .zip(rhs.mask.iter())
            .for_each(|(l, r)| *l = *l & *r);

        Self { mask }
    }
}

impl std::ops::BitOr for CpuSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        let mut mask = self.mask.clone();
        mask.iter_mut()
            .zip(rhs.mask.iter())
            .for_each(|(l, r)| *l = *l | *r);

        Self { mask }
    }
}

impl std::ops::BitXor for CpuSet {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut mask = self.mask.clone();
        mask.iter_mut()
            .zip(rhs.mask.iter())
            .for_each(|(l, r)| *l = *l ^ *r);

        Self { mask }
    }
}

pub fn mbind<T>(
    data: &[T],
    mode: MemPolicyModes,
    nodes: CpuSet,
    flags: MemBindFlags,
) -> Result<()> {
    unsafe {
        if bindings::mbind(
            data.as_ptr() as *mut T as *mut c_void,
            (data.len() * size_of::<T>()) as u64,
            mode.bits(),
            nodes.mask.as_ptr(),
            nodes.max_id().into(),
            // nodes.max_node().into(),
            flags.bits(),
        ) == -1
        {
            Err(ErrorKind::Io(IoError::last_os_error()))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use libc::{mmap, munmap};
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
        mbind(
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

/// NUMA allocations will fail if the memory cannot be allocated on the target
/// NUMA node. The default behavior is to fall back on other nodes.
pub fn numa_set_strict(strict: bool) {
    unsafe { bindings::numa_set_strict(strict.into()) };
}

/// Run the current thread on the specified NUMA node.
pub fn numa_run_on_node(node: u16) -> Result<()> {
    let ret = unsafe { bindings::numa_run_on_node(node.into()) };
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
pub fn numa_tonode_memory<T>(mem: &[T], node: u16) {
    unsafe {
        bindings::numa_tonode_memory(
            mem.as_ptr() as *mut T as *mut c_void,
            mem.len() * size_of::<T>(),
            node.into(),
        )
    };
}
