// Copyright 2019-2022 Clemens Lutz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::hw_info::ProcessorCache;
use crate::error::{Error, ErrorKind, Result};
use bitflags::bitflags;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::Error as IoError;
use std::io::{BufRead, BufReader};
use std::mem::{size_of, size_of_val};
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
        pub fn numa_set_preferred(node: c_int);
        pub fn numa_run_on_node(node: c_int) -> c_int;
        pub fn numa_set_strict(strict: c_int);
        pub fn numa_tonode_memory(start: *mut c_void, size: usize, node: c_int);
        pub fn numa_node_of_cpu(cpu: i32) -> i32;
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

bitflags::bitflags! {
pub struct MemProtectFlags: c_int {
    const NONE = libc::PROT_NONE;
    const READ = libc::PROT_READ;
    const WRITE = libc::PROT_WRITE;
    const EXEC = libc::PROT_EXEC;
    const GROWSDOWN = libc::PROT_GROWSDOWN;
    const GROWSUP = libc::PROT_GROWSUP;
}
}

/// CPU set to create CPU and NUMA node masks.
///
/// Inspired by Linux's `cpu_set_t`, see the `cpu_set` manual page.
///
/// Limitations
/// ===========
///
/// The set is currently restricted to a 16 64-bit integers. Therefore, IDs
/// must be smaller or equal to 1023.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct CpuSet {
    mask: [u64; Self::MASK_LEN as usize],
}

impl CpuSet {
    const MASK_LEN: u16 = 16;
    const ENTRY_LEN: u16 = 64;
    const MAX_LEN: u16 = Self::MASK_LEN * Self::ENTRY_LEN;

    /// Create an empty CPU set.
    pub fn new() -> Self {
        Self {
            mask: [0; Self::MASK_LEN as usize],
        }
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

    /// Returns the number of IDs in the set.
    pub fn count(&self) -> usize {
        self.mask.iter().map(|e| e.count_ones() as usize).sum()
    }

    /// Returns the size of the set in bytes.
    pub fn bytes(&self) -> usize {
        size_of_val(&self.mask)
    }

    /// Reset the set to zero.
    pub fn zero(&mut self) {
        self.mask.fill(0);
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

    /// Get the set as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        &mut self.mask
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

/// Defines the `mprotect` system call for the type
pub trait MemProtect {
    /// Sets the protection flags of a memory region
    ///
    /// Refer to `man mprotect` for details.
    fn mprotect(&self, flags: MemProtectFlags) -> Result<()>;
}

impl<T> MemProtect for [T] {
    fn mprotect(&self, flags: MemProtectFlags) -> Result<()> {
        mprotect(self, ProcessorCache::page_size(), flags)
    }
}

pub(crate) fn mprotect<T>(data: &[T], page_size: usize, flags: MemProtectFlags) -> Result<()> {
    let page_mask = !(page_size - 1);

    // Round pointer down to page start, and round length up to page size
    let std::ops::Range { start, end } = data.as_ptr_range();
    let start_page = start as usize & page_mask;
    let end_page = (end as usize + page_size - 1) & page_mask;

    let bytes = end_page - start_page;

    unsafe {
        if libc::mprotect(start_page as *mut std::ffi::c_void, bytes, flags.bits()) == -1 {
            Err(ErrorKind::Io(IoError::last_os_error()))?;
        }
    }

    Ok(())
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

/// Try to allocate memory on the sepecified NUMA node.
///
/// Falls back to other NUMA nodes if no memory is available on the preferred
/// node.
pub fn numa_set_preferred(node: u16) {
    unsafe { bindings::numa_set_preferred(node.into()) };
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

/// Find the NUMA node that the CPU core belongs to.
pub fn numa_node_of_cpu(cpu_id: u16) -> Result<u16> {
    let ret = unsafe { bindings::numa_node_of_cpu(cpu_id as i32) };
    if ret == -1 {
        Err(Error::with_chain(
            io::Error::last_os_error(),
            "Couldn't find NUMA node of the given CPU ID",
        ))
    } else {
        Ok(ret as u16)
    }
}

/// NUMA node memory information
pub struct NumaMemInfo {
    /// Total bytes
    pub total: usize,

    /// Free bytes
    pub free: usize,

    /// Used bytes
    pub used: usize,
}

/// Returns the memory information of a NUMA node
///
/// Read the memory information provided by Linux at
/// `/sys/devices/system/node/nodeX/meminfo`.
///
/// Example
/// ```
/// # use numa_gpu::runtime::linux_wrapper::{self, NumaMemInfo};
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let node_id = 0;
/// let NumaMemInfo{total, free, ..} = linux_wrapper::numa_mem_info(node_id)?;
/// println!("total: {} free: {}", total, free);
/// # Ok(())
/// # }
/// ```
pub fn numa_mem_info(node_id: u16) -> Result<NumaMemInfo> {
    let meminfo_path = format!("/sys/devices/system/node/node{}/meminfo", node_id);
    let meminfo_file = File::open(meminfo_path)?;
    let meminfo_map = read_sysfs_file(BufReader::new(&meminfo_file))?;

    // Line format: "Node 255 MemTotal:       16515072 kB"
    let parse_item = |item| -> usize {
        meminfo_map
            .get(&format!("Node {} {}", node_id, item))
            .expect("Failed to get item")
            .strip_suffix(" kB")
            .expect("Failed to strip unit")
            .trim()
            .parse()
            .expect("Failed to parse item")
    };

    let total = parse_item("MemTotal") * 1024;
    let free = parse_item("MemFree") * 1024;
    let used = parse_item("MemUsed") * 1024;

    Ok(NumaMemInfo { total, free, used })
}

/// Reads a Linux sysfs file and returns its contents as a hash map
pub(crate) fn read_sysfs_file<R: BufRead>(reader: R) -> Result<HashMap<String, String>> {
    let mut map = HashMap::new();

    for line in reader.lines() {
        let line = line?;

        if line.is_empty() {
            continue;
        }

        let mut fields = line.splitn(2, ':').map(|s| s.trim());
        if let (Some(key), Some(val)) = (fields.next(), fields.next()) {
            map.insert(key.to_string(), val.to_string());
        } else {
            return Err(
                ErrorKind::RuntimeError("Failed to parse the line from /proc".to_string()).into(),
            );
        }
    }

    Ok(map)
}
