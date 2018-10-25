/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

extern crate accel;

use self::accel::mvec::MVec;
use self::accel::uvec::UVec;

use std::any::Any;

pub use self::Mem::*;
pub enum Mem<T> {
    SysMem(Vec<T>),
    CudaDevMem(MVec<T>),
    CudaUniMem(UVec<T>),
}

impl<T: Any + Copy> Mem<T> {
    pub fn len(&self) -> usize {
        match self {
            SysMem(ref m) => m.len(),
            CudaDevMem(ref m) => m.len(),
            CudaUniMem(ref m) => m.len(),
        }
    }

    pub fn as_any(&self) -> &Any {
        match self {
            SysMem(ref m) => &m[0] as &Any,
            CudaDevMem(ref m) => m as &Any,
            CudaUniMem(ref m) => m as &Any,
        }
    }
}
