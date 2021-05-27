/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

//! Rust bindings to Linux's 'numa' library.

use super::cuda_wrapper::{host_register, host_unregister};
use super::hw_info::ProcessorCache;
use super::linux_wrapper::{mbind, CpuSet, MemBindFlags, MemPolicyModes};
use super::memory::{MemLock, PageLock};
use crate::error::{ErrorKind, Result, ResultExt};

use libc::{madvise, mlock, mmap, munlock, munmap};

use std::io::Error as IoError;
use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::slice;

use num_rational::Ratio;

/// Re-export Linux's NUMA bindings
pub use super::linux_wrapper::{
    numa_node_of_cpu as node_of_cpu, numa_run_on_node as run_on_node,
    numa_set_strict as set_strict, numa_tonode_memory as tonode_memory,
};

/// Specifies the allocation page type
///
/// Linux supports multiple page sizes, as well as transparent huge pages.
/// `PageType` configures type of pages (default, THP, HugeTLB) and page size
/// used for a memory allocation.
///
/// # Additional steps required
///
/// Before huge pages can be allocated, they first must be enabled by the system
/// administrator. For example, this can be done by running:
///
/// ```ignore
/// echo 20 > /proc/sys/vm/nr_overcommit_hugepages
/// ```
///
/// # Documentation
///
/// See the Linux kernel documentation on [transparent huge pages][thp_docs] and
/// [HugeTLB][hugetlb_docs] for details.
///
/// [thp_docs]: https://www.kernel.org/doc/Documentation/vm/transhuge.txt
/// [hugetlb_docs]: https://www.kernel.org/doc/Documentation/vm/hugetlbpage.txt
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PageType {
    /// The default page type
    ///
    /// As configured in the operating system. Modern Linux systems usually have
    /// transparent huge pages enabled.
    Default,

    /// Small pages
    ///
    /// Forces the use of small pages with `madvise`. Thus, small pages are
    /// allocated even if transparent huge pages are enabled. The small page
    /// size depends on the processor architecture. For x86_64 it's 4 KiB, for
    /// ppc64le it's 64 KiB.
    Small,

    /// Transparent huge pages
    ///
    /// Forces the use of transparent huge pages with `madvise`, even if huge
    /// pages are disabled.
    TransparentHuge,

    /// 2 MiB huge pages
    ///
    /// Allocates 2 MiB huge pages using HugeTLB with `mmap`.
    Huge2MB,

    /// 16 MiB huge pages
    ///
    /// Allocates 16 MiB huge pages using HugeTLB with `mmap`.
    Huge16MB,

    /// 1 GiB huge pages
    ///
    /// Allocates 1 GiB huge pages using HugeTLB with `mmap`.
    Huge1GB,

    /// 16 GiB huge pages
    ///
    /// Allocates 16 GiB huge pages using HugeTLB with `mmap`.
    Huge16GB,
}

impl PageType {
    /// Returns the page size in bytes
    fn page_size(&self) -> Result<usize> {
        match self {
            PageType::Default | PageType::Small => Ok(ProcessorCache::page_size()),
            // THP use small page size instead of ProcessorCache::huge_page_size(),
            // because alignment for mbind and munmap is on small page size.
            PageType::TransparentHuge => Ok(ProcessorCache::page_size()),
            PageType::Huge2MB => Ok(1 << 21),
            PageType::Huge16MB => Ok(1 << 24),
            PageType::Huge1GB => Ok(1 << 30),
            PageType::Huge16GB => Ok(1 << 34),
        }
    }
}

/// Returns `x` rounded up to the page size
fn round_to_next_page(x: usize, page_size: usize) -> usize {
    let align_mask = !(page_size - 1);
    (x + page_size - 1) & align_mask
}

/// A contiguous memory region that is dynamically allocated on the specified
/// NUMA node.
#[derive(Debug)]
pub struct NumaMemory<T> {
    pointer: *mut T,
    len: usize,
    node: u16,
    page_type: PageType,
    is_memory_locked: bool,
    is_page_locked: bool,
}

impl<T> NumaMemory<T> {
    /// Allocates a new memory region with the specified capacity on the specified NUMA node.
    ///
    /// == Transparent Huge Pages ==
    ///
    /// Small pages are advised using `madvise` when the `huge_pages` flag is set to `false`. Huge
    /// pages are advised when the flag is set to `true`. However, the actual behavior depends on
    /// OS configuration options. Specifying `None` uses the default OS setting.
    ///
    /// - `/sys/kernel/mm/transparent_hugepage/enabled` controls the default page size. It can be
    /// set to `never`, `madvise`, or `always`. The settings `madvise` and `always` set the default
    /// page size to small pages and huge pages, respectively. Both of these options allow
    /// `madvise` to override the default. In constrast, the setting `none` specifies small pages
    /// without an override option.
    ///
    /// - `/sys/kernel/mm/transparent_hugepage/defrag` controls huge page reclaimation when not
    /// enough huge pages are available.  The options `always`, `defer+madvise`, and `madvise`
    /// stall on the `madvise` syscall until enough huge pages are available. In contrast, `defer`
    /// and `never` allow small page allocation despite requesting huge pages.
    ///
    /// See the [Linux kernel documentation](https://www.kernel.org/doc/Documentation/vm/transhuge.txt)
    /// for more details.
    ///
    /// == Memory Alignment ==
    ///
    /// `mmap` with `MMAP_ANONYMOUS` allocates pages. Separate alignment for cacheline alignment is
    /// not necessary.
    pub fn new(len: usize, node: u16, page_type: PageType) -> Self {
        assert_ne!(len, 0);

        let hugetlb_flags = match page_type {
            PageType::Huge2MB => libc::MAP_HUGETLB | libc::MAP_HUGE_2MB,
            PageType::Huge16MB => libc::MAP_HUGETLB | libc::MAP_HUGE_16MB,
            PageType::Huge1GB => libc::MAP_HUGETLB | libc::MAP_HUGE_1GB,
            PageType::Huge16GB => libc::MAP_HUGETLB | libc::MAP_HUGE_16GB,
            PageType::Default | PageType::Small | PageType::TransparentHuge => 0,
        };

        // Allocate memory with mmap
        let size = len * size_of::<T>();
        let pointer = unsafe {
            mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | hugetlb_flags,
                0,
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            std::result::Result::Err::<(), _>(IoError::last_os_error())
                .expect("Failed to mmap memory");
        }

        // Enable or disable transparent transparent huge pages
        let advice = match page_type {
            PageType::Small => Some(libc::MADV_NOHUGEPAGE),
            PageType::TransparentHuge => Some(libc::MADV_HUGEPAGE),
            PageType::Default
            | PageType::Huge2MB
            | PageType::Huge16MB
            | PageType::Huge1GB
            | PageType::Huge16GB => None,
        };

        if let Some(advice_flag) = advice {
            unsafe {
                if madvise(pointer, size, advice_flag) == -1 {
                    let err = IoError::last_os_error();
                    std::result::Result::Err::<(), _>(err).expect("Failed to madvise memory");
                }
            }
        }

        // Set up the NUMA node set for mbind
        let mut node_set = CpuSet::new();
        node_set.add(node);

        // mbind fails with `EINVAL` for HugeTLB mappings if `size` isn't a
        // multiple of the page size
        let page_size = page_type.page_size().expect("Failed to get the page size");
        let aligned_size = round_to_next_page(size, page_size);

        // Bind to the NUMA node
        unsafe {
            let slice = slice::from_raw_parts(pointer, aligned_size);

            mbind(slice, MemPolicyModes::BIND, node_set, MemBindFlags::STRICT)
                .expect("Failed to bind memory to NUMA node.");
        }

        Self {
            pointer: pointer as *mut T,
            len,
            node,
            page_type,
            is_memory_locked: false,
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

impl<T> MemLock for NumaMemory<T> {
    fn mlock(&mut self) -> Result<()> {
        if !self.is_memory_locked {
            let size = self.len * size_of::<T>();
            unsafe {
                if mlock(self.pointer as *mut libc::c_void, size) == -1 {
                    let err = IoError::last_os_error();
                    if let Some(code) = err.raw_os_error() {
                        if code == libc::ENOMEM {
                            eprintln!("mlock() failed with ENOMEM; try setting 'memlock' to 'unlimited' in /etc/security/limits.conf");
                        }
                    }

                    std::result::Result::Err::<(), _>(err).expect("Failed to mlock memory");
                }
            }
            self.is_memory_locked = true;
        }

        Ok(())
    }

    fn munlock(&mut self) -> Result<()> {
        if self.is_memory_locked {
            let size = self.len * size_of::<T>();
            unsafe {
                if munlock(self.pointer as *mut libc::c_void, size) == -1 {
                    std::result::Result::Err::<(), _>(IoError::last_os_error())
                        .expect("Failed to munlock memory");
                }
            }
        }

        Ok(())
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

        // munmap fails with `EINVAL` for HugeTLB mappings if `size` isn't a
        // multiple of the page size. This error is documented in `man munmap`.
        let size = self.len * size_of::<T>();
        let page_size = self
            .page_type
            .page_size()
            .expect("Failed to get the page size");
        let aligned_size = round_to_next_page(size, page_size);

        unsafe {
            if munmap(self.pointer as *mut libc::c_void, aligned_size) == -1 {
                std::result::Result::Err::<(), _>(IoError::last_os_error())
                    .expect("Failed to munmap memory");
            }
        }
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

/// Specifies the requested memory allocation size on the NUMA node
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeLen {
    /// The NUMA node
    pub node: u16,

    /// The allocation size in bytes
    pub len: usize,
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
    page_type: PageType,
    is_memory_locked: bool,
    is_page_locked: bool,
}

impl<T> DistributedNumaMemory<T> {
    /// Allocates a new memory region.
    ///
    /// Memory is allocated on the specified NUMA nodes according to
    /// `node_lengths`. The actual allocations can slightly differ from the
    /// specified lengths, because the allocation granularity is a page and the
    /// lengths are rounded to a page.
    ///
    /// Note that the sum of all node lengths must equal `len`.
    pub fn new_with_len(len: usize, node_lengths: Box<[NodeLen]>, page_type: PageType) -> Self {
        assert_ne!(len, 0);
        {
            let total: usize = node_lengths.iter().map(|n| n.len).sum();
            assert_eq!(total, len);
        }

        let size = len * size_of::<T>();
        let page_size = page_type.page_size().expect("Failed to get the page size");
        let total_pages = (size + page_size - 1) / page_size;

        // Round number of pages up
        let node_pages: Box<[NodeLen]> = node_lengths
            .iter()
            .enumerate()
            .scan(0, |pages_seen, (i, &NodeLen { node, len })| {
                let pages = if i + 1 == node_lengths.len() {
                    // The last node gets the remainder
                    total_pages - *pages_seen
                } else {
                    // Round number of pages down
                    let pages = (len * size_of::<T>()) / page_size;
                    *pages_seen += pages;
                    pages
                };

                Some(NodeLen { node, len: pages })
            })
            .collect();

        Self::new_with_pages(len, node_pages, page_type)
    }

    /// Allocates a new memory region.
    ///
    /// Memory is allocated proportionally on the specified NUMA nodes according
    /// to their ratios. The actual ratios can slightly differ from the specified
    /// ratios, because the allocation granularity is a page.
    ///
    /// Note that the sum of all ratios must equal 1.
    pub fn new_with_ratio(len: usize, node_ratios: Box<[NodeRatio]>, page_type: PageType) -> Self {
        assert_ne!(len, 0);
        {
            let total: Ratio<usize> = node_ratios.iter().map(|n| n.ratio).sum();
            assert_eq!(total, 1.into());
        }

        // Calculate number of pages, rounded up
        let size = len * size_of::<T>();
        let page_size = page_type.page_size().expect("Failed to get the page size");
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

        let node_pages: Box<[NodeLen]> = scaled_ratios
            .iter()
            .map(|&NodeRatio { node, ratio }| NodeLen {
                node,
                len: ratio.to_integer(),
            })
            .collect();

        Self::new_with_pages(len, node_pages, page_type)
    }

    fn new_with_pages(len: usize, node_pages: Box<[NodeLen]>, page_type: PageType) -> Self {
        let hugetlb_flags = match page_type {
            PageType::Huge2MB => libc::MAP_HUGETLB | libc::MAP_HUGE_2MB,
            PageType::Huge16MB => libc::MAP_HUGETLB | libc::MAP_HUGE_16MB,
            PageType::Huge1GB => libc::MAP_HUGETLB | libc::MAP_HUGE_1GB,
            PageType::Huge16GB => libc::MAP_HUGETLB | libc::MAP_HUGE_16GB,
            PageType::Default | PageType::Small | PageType::TransparentHuge => 0,
        };

        // Allocate memory with mmap
        let size = len * size_of::<T>();
        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | hugetlb_flags,
                0,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            std::result::Result::Err::<(), _>(IoError::last_os_error())
                .expect("Failed to mmap memory");
        }

        // Enable or disable transparent transparent huge pages
        let advice = match page_type {
            PageType::Small => Some(libc::MADV_NOHUGEPAGE),
            PageType::TransparentHuge => Some(libc::MADV_HUGEPAGE),
            PageType::Default
            | PageType::Huge2MB
            | PageType::Huge16MB
            | PageType::Huge1GB
            | PageType::Huge16GB => None,
        };

        if let Some(advice_flag) = advice {
            unsafe {
                if madvise(ptr, size, advice_flag) == -1 {
                    let err = IoError::last_os_error();
                    std::result::Result::Err::<(), _>(err).expect("Failed to madvise memory");
                }
            }
        }

        // Calculate number of pages, rounded up
        let page_size = page_type.page_size().expect("Failed to get the page size");
        let aligned_size = round_to_next_page(size, page_size);
        let pages = aligned_size / page_size;

        {
            let pages_sum: usize = node_pages.iter().map(|n| n.len).sum();
            assert_eq!(pages, pages_sum);
        }

        // Bind all pages to their NUMA nodes, and calculate the actual ratios
        let final_node_ratios = node_pages
            .iter()
            .scan(
                0,
                |page_offset,
                 &NodeLen {
                     node,
                     len: page_len,
                 }| {
                    let old = *page_offset;
                    *page_offset = *page_offset + page_len;
                    Some((node, old, page_len))
                },
            )
            .map(|(node, page_offset, page_len)| {
                let mut node_set = CpuSet::new();
                node_set.add(node);

                unsafe {
                    let slice = slice::from_raw_parts(
                        ptr.add(page_offset as usize * page_size),
                        page_len as usize * page_size,
                    );

                    // Note that mbind fails with `EINVAL` for HugeTLB mappings if `size`
                    // isn't a multiple of the page size
                    mbind(slice, MemPolicyModes::BIND, node_set, MemBindFlags::STRICT)?;
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
            page_type,
            is_memory_locked: false,
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

        // munmap fails with `EINVAL` for HugeTLB mappings if `size` isn't a
        // multiple of the page size. This error is documented in `man munmap`.
        let size = self.len * size_of::<T>();
        let page_size = self
            .page_type
            .page_size()
            .expect("Failed to get the page size");
        let aligned_size = round_to_next_page(size, page_size);

        unsafe {
            if munmap(self.ptr as *mut libc::c_void, aligned_size) == -1 {
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

impl<T> MemLock for DistributedNumaMemory<T> {
    fn mlock(&mut self) -> Result<()> {
        if !self.is_memory_locked {
            let size = self.len * size_of::<T>();
            unsafe {
                if mlock(self.ptr as *mut libc::c_void, size) == -1 {
                    let err = IoError::last_os_error();
                    if let Some(code) = err.raw_os_error() {
                        if code == libc::ENOMEM {
                            eprintln!("mlock() failed with ENOMEM; try setting 'memlock' to 'unlimited' in /etc/security/limits.conf");
                        }
                    }

                    std::result::Result::Err::<(), _>(err).expect("Failed to mlock memory");
                }
            }
            self.is_memory_locked = true;
        }

        Ok(())
    }

    fn munlock(&mut self) -> Result<()> {
        if self.is_memory_locked {
            let size = self.len * size_of::<T>();
            unsafe {
                if munlock(self.ptr as *mut libc::c_void, size) == -1 {
                    std::result::Result::Err::<(), _>(IoError::last_os_error())
                        .expect("Failed to munlock memory");
                }
            }
        }

        Ok(())
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
