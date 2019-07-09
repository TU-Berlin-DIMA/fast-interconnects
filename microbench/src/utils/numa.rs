use std::mem::size_of;
use std::os::raw::{c_int, c_void};
use std::slice;
use std::u8;

use super::hw_info::ProcessorCache;

#[link(name = "numa")]
extern "C" {
    pub fn numa_run_on_node(node: c_int) -> c_int;
    pub fn numa_set_strict(strict: c_int);
    pub fn numa_alloc_onnode(size: usize, node: c_int) -> *mut c_void;
    pub fn numa_free(start: *mut c_void, size: usize);
    pub fn numa_tonode_memory(start: *mut c_void, size: usize, node: c_int);
}

#[derive(Debug)]
pub struct NumaMemory<T> {
    pointer: *mut T,
    len: usize,
    base_pointer: *mut T,
    alignment: usize,
    node: u16,
}

impl<'a, T> NumaMemory<T> {
    pub fn alloc_on_node(len: usize, node: u16) -> Self {
        // Get page alignment
        let alignment = ProcessorCache::page_size();

        let size = len * size_of::<T>();
        let c_node = node as c_int;
        let base_pointer = unsafe { numa_alloc_onnode(size + alignment, c_node) } as *mut T;
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
        }
    }

    pub fn as_slice(&self) -> &'a [T] {
        unsafe { slice::from_raw_parts::<'a>(self.pointer, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &'a mut [T] {
        unsafe { slice::from_raw_parts_mut::<'a>(self.pointer, self.len) }
    }

    // TODO: implement deref instead
    pub fn as_ptr(&self) -> *const T {
        self.pointer
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.pointer
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn node(&self) -> u16 {
        self.node
    }

    pub fn set_strict() {
        let c_true: c_int = 1;
        unsafe { numa_set_strict(c_true) };
    }
}

impl<T> Drop for NumaMemory<T> {
    fn drop(&mut self) {
        let size = self.len * size_of::<T>();
        unsafe { numa_free(self.base_pointer as *mut c_void, size + self.alignment) };
    }
}

unsafe impl<T> Send for NumaMemory<T> {}
unsafe impl<T> Sync for NumaMemory<T> {}

pub fn run_on_node(node: u16) {
    let c_node = node as c_int;
    let ret = unsafe { numa_run_on_node(c_node) };
    if ret == -1 {
        panic!("Couldn't bind thread to node {}", node);
    }
}

/// Put memory on a specific node.
///
/// ```
/// let data = vec!(1; 1024);
/// tonode_memory(&data, 0);
/// ```
pub fn tonode_memory<T>(mem: &[T], node: u16) {
    unsafe {
        numa_tonode_memory(
            mem.as_ptr() as *mut T as *mut c_void,
            mem.len() * size_of::<T>(),
            node as c_int,
        )
    };
}
