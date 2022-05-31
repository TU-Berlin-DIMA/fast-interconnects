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

//! Rust-ified wrapper of NVIDIA Tools Extension (NVTX) library.

use crate::error::{ErrorKind, Result};
use nvtx_sys::{
    nvtxMarkA, nvtxRangeEnd, nvtxRangeId_t, nvtxRangePop, nvtxRangePushA, nvtxRangeStartA,
};
use serde::Serialize;
use std::ffi::CStr;
use std::fmt;

const NVTX_NO_PUSH_POP_TRACKING: i32 = -2;

/// A range denoting a time span.
///
/// A range denotes an arbitrary time span in a process. Each range can contain
/// a text message or specify additional attributes.
///
/// A unique (opaque) correlation ID is created for each range. The correlation
/// ID is annotated to the nvprof profiler output.
///
/// The official documentation is available in [Nvidia's Profiler User
/// Guide](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx).
///
/// # Thread-safety
///
/// A range may be ended by a different thread than it is started by. Thus, a
/// range is `Send` but not `Sync`, `Copy` or `Clone`. However, a pop must occur
/// on the same thread as the push.
///
/// # Examples
///
/// A range can be started and stopped at arbitrary points.
///
/// ```
/// # use numa_gpu::runtime::nvtx::Range;
/// # use std::ffi::CStr;
/// #
/// let message = unsafe { CStr::from_bytes_with_nul_unchecked(b"Hello World!\0") };
/// let range = Range::new(message);
/// // ...
/// let range_id = range.end();
/// println!("Range {} ended", range_id);
/// ```
///
/// Alternatively, a ranges can be nested by pushing and popping them.
///
/// ```no_run
/// # use numa_gpu::runtime::nvtx::Range;
/// # use std::ffi::CStr;
/// #
/// # // Note: push and pop return error when NVTX is not initialized. However,
/// # // linking nvtxInitialize gives an error.
/// let message = unsafe { CStr::from_bytes_with_nul_unchecked(b"Hello World!\0") };
/// Range::push(&message).unwrap();
/// // ...
/// Range::pop().unwrap();
/// ```
#[derive(Debug)]
pub struct Range {
    id: nvtxRangeId_t,
}

/// A range identifier.
///
/// Each range ID is associated with a range, and can be matched to the output
/// of a profiler. For example, nvprof lists the range ID in its output.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct RangeId(nvtxRangeId_t);

impl Range {
    /// Start a process range.
    ///
    /// The `message` is associated to this range event.
    pub fn new(message: &CStr) -> Self {
        let id = unsafe { nvtxRangeStartA(message.as_ptr()) };

        Self { id }
    }

    /// End a process range.
    pub fn end(self) -> RangeId {
        unsafe {
            nvtxRangeEnd(self.id);
        }

        RangeId(self.id)
    }

    /// Get the ID of a range.
    pub fn id(&self) -> RangeId {
        RangeId(self.id)
    }

    /// Start a new nested process range.
    ///
    /// The `message` is associated to this range event. The push returns the
    /// zero-based depth of the range being started.
    pub fn push(message: &CStr) -> Result<u16> {
        let depth = unsafe { nvtxRangePushA(message.as_ptr()) };

        Self::push_pop_error(depth)
    }

    /// Stop a nested range.
    ///
    /// The pop returns the zero-based depth of the range being ended. A runtime
    /// error is returned if no range was pushed before the pop.
    pub fn pop() -> Result<u16> {
        let depth = unsafe { nvtxRangePop() };

        Self::push_pop_error(depth)
    }

    fn push_pop_error(depth: i32) -> Result<u16> {
        match depth {
            d if d >= 0 => Ok(depth as u16),
            NVTX_NO_PUSH_POP_TRACKING => {
                Err(ErrorKind::RuntimeError("No NVTX push-pop tracking.".to_string()).into())
            }
            _ => Err(ErrorKind::RuntimeError(
                "A NVTX range must be pushed before it can be popped.".to_string(),
            )
            .into()),
        }
    }
}

impl fmt::Display for RangeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Set a marker.
///
/// A marker describes an instanteneous event.
///
/// # Example
///
/// ```
/// # use numa_gpu::runtime::nvtx;
/// # use std::ffi::CStr;
/// #
/// let message = unsafe { CStr::from_bytes_with_nul_unchecked(b"Hello World!\0") };
/// nvtx::mark(&message);
/// ```
pub fn mark(message: &CStr) {
    unsafe { nvtxMarkA(message.as_ptr()) }
}
