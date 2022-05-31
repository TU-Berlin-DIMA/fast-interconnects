// Copyright 2018-2022 Clemens Lutz
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

use std::convert::From;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum ErrorKind {
    CudaError(rustacuda::error::CudaError),
    IntegerOverflow(String),
    InvalidArgument(String),
    LikwidError(likwid::error::LikwidError),
    Msg(String),
    NulCharError(String),
    NumaGpuError(numa_gpu::error::Error),
    RuntimeError(String),
}

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
}

impl Error {
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.kind, f)
    }
}

impl From<rustacuda::error::CudaError> for Error {
    fn from(error: rustacuda::error::CudaError) -> Self {
        Self {
            kind: ErrorKind::CudaError(error),
        }
    }
}

impl From<likwid::error::LikwidError> for Error {
    fn from(error: likwid::error::LikwidError) -> Self {
        Self {
            kind: ErrorKind::LikwidError(error),
        }
    }
}

impl From<numa_gpu::error::Error> for Error {
    fn from(error: numa_gpu::error::Error) -> Self {
        Self {
            kind: ErrorKind::NumaGpuError(error),
        }
    }
}

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Self {
        Self { kind }
    }
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorKind::CudaError(ref e) => e.fmt(f),
            ErrorKind::IntegerOverflow(ref s) => write!(f, "IntegerOverflow: {}", s),
            ErrorKind::InvalidArgument(ref s) => write!(f, "InvalidArgument: {}", s),
            ErrorKind::LikwidError(ref e) => e.fmt(f),
            ErrorKind::NulCharError(ref s) => write!(f, "NulCharError: {}", s),
            ErrorKind::NumaGpuError(ref e) => e.fmt(f),
            ErrorKind::Msg(ref s) => write!(f, "Msg: {}", s),
            ErrorKind::RuntimeError(ref s) => write!(f, "Runtime: {}", s),
        }
    }
}

impl From<String> for ErrorKind {
    fn from(s: String) -> Self {
        ErrorKind::Msg(s)
    }
}

impl<'a> From<&'a str> for ErrorKind {
    fn from(s: &'a str) -> Self {
        ErrorKind::Msg(s.to_string())
    }
}
