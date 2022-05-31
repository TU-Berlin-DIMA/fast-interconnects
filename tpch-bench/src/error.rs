// Copyright 2020-2022 Clemens Lutz
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

use csv::Error as CsvError;
use datagen::error::{Error as DataGenError, ErrorKind as DataGenErrorKind};
use numa_gpu::error::Error as NumaGpuError;
use rustacuda::error::CudaError;
use std::convert::From;
use std::io::Error as IoError;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum ErrorKind {
    CsvError(CsvError),
    CudaError(CudaError),
    IntegerOverflow(String),
    InvalidArgument(String),
    IoError(IoError),
    Msg(String),
    NulCharError(String),
    NumaGpuError(NumaGpuError),
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

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Self {
        Self { kind }
    }
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorKind::CsvError(ref e) => e.fmt(f),
            ErrorKind::CudaError(ref e) => e.fmt(f),
            ErrorKind::IntegerOverflow(ref s) => write!(f, "IntegerOverflow: {}", s),
            ErrorKind::InvalidArgument(ref s) => write!(f, "InvalidArgument: {}", s),
            ErrorKind::IoError(ref e) => e.fmt(f),
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

impl From<CsvError> for Error {
    fn from(e: CsvError) -> Self {
        Self {
            kind: ErrorKind::CsvError(e),
        }
    }
}

impl From<CudaError> for Error {
    fn from(e: CudaError) -> Self {
        Self {
            kind: ErrorKind::CudaError(e),
        }
    }
}

impl From<IoError> for Error {
    fn from(e: IoError) -> Self {
        Self {
            kind: ErrorKind::IoError(e),
        }
    }
}

impl From<NumaGpuError> for Error {
    fn from(e: NumaGpuError) -> Self {
        Self {
            kind: ErrorKind::NumaGpuError(e),
        }
    }
}

impl From<DataGenError> for Error {
    fn from(e: DataGenError) -> Self {
        let kind = match e.kind {
            DataGenErrorKind::Msg(s) => ErrorKind::Msg(s),
            DataGenErrorKind::IntegerOverflow(o) => ErrorKind::IntegerOverflow(o),
            DataGenErrorKind::InvalidArgument(a) => ErrorKind::InvalidArgument(a),
        };

        Self { kind }
    }
}
