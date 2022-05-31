// Copyright 2021-2022 Clemens Lutz
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
