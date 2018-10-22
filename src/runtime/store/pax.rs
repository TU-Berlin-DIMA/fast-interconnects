/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use super::AttrVec;

use std::mem::size_of;
use std::num::NonZeroU32;

const PAGE_ALIGN: usize = 256;
const PAGE_MAX_ATTRS: u32 = 8;
const PAGE_HEADER_SIZE: usize =
    ((size_of::<HeaderContents>() + PAGE_ALIGN - 1) / PAGE_ALIGN) * PAGE_ALIGN;

/// A page representing the PAX storage format
///
#[repr(C, packed)]
pub struct Page {
    header: Header,
    payload: [u8],
}

#[repr(C, packed)]
struct Header {
    h: HeaderContents,
    _pad: [u8; PAGE_HEADER_SIZE - size_of::<HeaderContents>()],
}

/// Actual contents of the header
///
/// attrs: Number of attributes stored in the page
/// rcrds: Number of records stored in the page
/// offset: Offset of each minipage; offsets are an exclusive prefix sum of
///         minipage sizes, with the first offset implicitly assumed as 0
/// attr_size: Size in bytes of fixed-size attributes, NULL if variable size
///
#[repr(C, packed)]
struct HeaderContents {
    attrs: u32,
    rcrds: u32,
    offset: [u32; PAGE_MAX_ATTRS as usize],
    attr_size: [AttrSize; PAGE_MAX_ATTRS as usize],
}

#[repr(C)]
enum AttrSize {
    Fixed(NonZeroU32),
    Variable,
}

#[repr(C)]
pub enum Minipage<'a> {
    Fixed(FixedMinipage<'a>),
    Variable(VariableMinipage<'a>),
}

/// A minipage with fixed-size values
///
/// Each value has a presence bit. Values are stored at the start of the minipage,
/// while presence bits are stored at the end of the minipage. Presence bits are
/// stored in the same order as their corresponding values.
///
#[repr(C)]
pub struct FixedMinipage<'a> {
    payload: &'a [u8],
    presence: &'a [u8],
    attr_size: NonZeroU32,
}

/// A minipage with variable-sized values
///
/// Each value has a length. Lengths are stored in reverse order of the values.
/// The offset position of a value is given by the sum of all lengths before it.
///
#[repr(C)]
pub struct VariableMinipage<'a> {
    payload: &'a [u8],
    length: &'a [u32],
}

/// A builder to create a Page
///
/// Use new() to start builing, add features, and finally call the build
/// method.
///
///     # extern crate numa_gpu;
///     # use numa_gpu::runtime::store::*;
///     # use std::mem::size_of;
///     # let mut array: [u8; 1024] = [0; 1024];
///     # let mut mem = &mut array[..];
///     let builder = PageBuilder::new(mem)
///         .add_attr(AttrType::Fixed(size_of::<u32>()));
///     let _page = builder.build();
///
pub struct PageBuilder<'p> {
    mem: &'p mut [u8],
    attrs: Vec<AttrType>,
}

pub enum AttrType {
    Fixed(usize),
    Variable(usize),
}

impl Page {
    /// Extracts a slice containing the entire minipage.
    ///
    pub fn minipage_slice<'s>(&'s self, attr: u32) -> &'s [u8] {
        let header = &self.header.h;

        assert!(attr < header.attrs);

        let mp_range = if attr == 0 {
            0
        } else {
            header.offset[attr as usize - 1] as usize
        }..(header.offset[attr as usize] as usize);

        &self.payload[mp_range]
    }

    /// Extracts a mutable slice containing the entire minipage.
    ///
    pub fn minipage_mut_slice<'s>(&'s mut self, attr: u32) -> &'s mut [u8] {
        let header = &self.header.h;

        assert!(attr < header.attrs);

        let mp_range = if attr == 0 {
            0
        } else {
            header.offset[attr as usize - 1] as usize
        }..(header.offset[attr as usize] as usize);

        &mut self.payload[mp_range]
    }

    /// Returns a Minipage containing a reference to data
    ///
    pub fn minipage(&self, attr: u32) -> Minipage {
        let header = &self.header.h;

        assert!(attr < header.attrs);

        match header.attr_size[attr as usize] {
            AttrSize::Fixed(attr_size) => {
                let mp_range = if attr == 0 {
                    0
                } else {
                    header.offset[attr as usize - 1] as usize
                }..(header.offset[attr as usize] as usize);

                let mp_size = mp_range.end - mp_range.start;

                // Each attribute requires a present bit
                let attr_bits = attr_size.get() as u64 * 8;
                let mp_bits = mp_size as u64 * 8;
                let max_records = ((mp_bits / (attr_bits + 1) + 7) / 8) as usize;
                let presence_size = ((max_records as u64 + 7) / 8) as usize;

                let v_range = mp_range.start..(mp_range.start + max_records);
                let p_range = (mp_range.end - presence_size)..mp_range.end;

                Minipage::Fixed(FixedMinipage {
                    payload: &self.payload[v_range],
                    presence: &self.payload[p_range],
                    attr_size: attr_size,
                })
            }
            AttrSize::Variable => {
                unimplemented!();
            }
        }
    }
}

impl<'a> AttrVec for FixedMinipage<'a> {}

impl<'a> ::std::ops::Index<usize> for FixedMinipage<'a> {
    type Output = [u8];

    fn index(&self, pos: usize) -> &[u8] {
        let size = self.attr_size.get() as usize;
        &self.payload[(pos * size)..((pos + 1) * size)]
    }
}

impl<'a> AttrVec for VariableMinipage<'a> {}

impl<'a> ::std::ops::Index<usize> for VariableMinipage<'a> {
    type Output = [u8];

    fn index(&self, _pos: usize) -> &[u8] {
        unimplemented!();
    }
}

impl<'p> PageBuilder<'p> {
    pub fn new(mem: &mut [u8]) -> PageBuilder {
        PageBuilder {
            mem,
            attrs: Vec::new(),
        }
    }

    pub fn add_attr(mut self, attr_type: AttrType) -> Self {
        self.attrs.push(attr_type);
        self
    }

    pub fn build(self) -> &'p mut Page {
        let page = unsafe { &mut *(self.mem as *mut [u8] as *mut Page) };

        // TODO: check if attributes <= PAGE_MAX_ATTRS and return Result

        page.header.h.attrs = self.attrs.len() as u32;
        page.header.h.rcrds = 0;
        page.header.h.offset[0] = 0;

        let record_size: usize = self
            .attrs
            .iter()
            .map(|t| match t {
                AttrType::Fixed(size) => *size as usize,
                AttrType::Variable(size) => *size as usize,
            }).sum();
        let max_records = (page.payload.len() / record_size) as u32;

        let mut sum = 0;
        for (i, t) in self.attrs.iter().enumerate() {
            sum += max_records * match t {
                AttrType::Fixed(size) => *size as u32,
                AttrType::Variable(size) => *size as u32,
            };

            page.header.h.offset[i] = sum;
        }

        for (i, t) in self.attrs.iter().enumerate() {
            page.header.h.attr_size[i] = match t {
                AttrType::Fixed(size) => {
                    AttrSize::Fixed(NonZeroU32::new(*size as u32).expect("Attribute has zero size"))
                }
                AttrType::Variable(size) => {
                    assert_ne!(*size, 0);
                    AttrSize::Variable
                }
            };
        }

        page
    }
}
