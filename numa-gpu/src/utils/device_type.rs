/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2021, Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

/// A device type specifier
///
/// The device type is sometimes useful to specify, e.g., on which device type
/// a function should be executed. The per-device generic payload stores
/// additional metadata, such as a closure or function parameters.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DeviceType<T, U> {
    /// CPU device type
    Cpu(T),

    /// GPU device type
    Gpu(U),
}

impl<T, U> DeviceType<T, U> {
    /// Returns `true` if the device is a CPU
    pub fn is_cpu(&self) -> bool {
        match self {
            Self::Cpu(_) => true,
            Self::Gpu(_) => false,
        }
    }

    /// Returns `true` if the device is a GPU
    pub fn is_gpu(&self) -> bool {
        !self.is_cpu()
    }

    /// Returns `Some(T)` if device is a CPU or otherwise `None`
    pub fn cpu(self) -> Option<T> {
        match self {
            Self::Cpu(x) => Some(x),
            Self::Gpu(_) => None,
        }
    }

    /// Returns `Some(U)` if device is a GPU or otherwise `None`
    pub fn gpu(self) -> Option<U> {
        match self {
            Self::Cpu(_) => None,
            Self::Gpu(y) => Some(y),
        }
    }

    /// Returns the contained `T` for a CPU, or computes it from the closure
    pub fn cpu_or_else<F: FnOnce(U) -> T>(self, f: F) -> T {
        match self {
            Self::Cpu(x) => x,
            Self::Gpu(y) => f(y),
        }
    }

    /// Returns the contained `U` for a GPU, or computes it from the closure
    pub fn gpu_or_else<F: FnOnce(T) -> U>(self, f: F) -> U {
        match self {
            Self::Cpu(x) => f(x),
            Self::Gpu(y) => y,
        }
    }

    /// Maps the contained `T` or `U` into the same `M` type using the respective closure
    pub fn either<F, G, M>(self, f: F, g: G) -> M
    where
        F: FnOnce(T) -> M,
        G: FnOnce(U) -> M,
    {
        match self {
            Self::Cpu(x) => f(x),
            Self::Gpu(y) => g(y),
        }
    }
}
