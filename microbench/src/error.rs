/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2020, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use csv::Error as CsvError;
use numa_gpu::error::Error as NumaGpuError;
use rustacuda::error::CudaError;
use std::convert::From;
use std::io::Error as IoError;

#[cfg(not(target_arch = "aarch64"))]
use nvml_wrapper::error::Error as NvmlError;

#[cfg(target_arch = "aarch64")]
type NvmlError = ();

pub type Result<T> = std::result::Result<T, Error>;

#[allow(dead_code)]
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
    NvmlError(NvmlError),
    RuntimeError(String),
}

#[derive(Debug)]
pub struct Error {
    pub kind: ErrorKind,
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
            ErrorKind::Msg(ref s) => write!(f, "Msg: {}", s),
            ErrorKind::NulCharError(ref s) => write!(f, "NulCharError: {}", s),
            ErrorKind::NumaGpuError(ref e) => e.fmt(f),
            ErrorKind::RuntimeError(ref s) => write!(f, "Runtime: {}", s),

            #[cfg(not(target_arch = "aarch64"))]
            ErrorKind::NvmlError(ref e) => e.fmt(f),

            #[cfg(target_arch = "aarch64")]
            ErrorKind::NvmlError(_) => unreachable!(),
        }
    }
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Self {
            kind: ErrorKind::Msg(s),
        }
    }
}

impl<'a> From<&'a str> for Error {
    fn from(s: &'a str) -> Self {
        Self {
            kind: ErrorKind::Msg(s.to_string()),
        }
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

#[cfg(not(target_arch = "aarch64"))]
impl From<NvmlError> for Error {
    fn from(e: NvmlError) -> Self {
        Self {
            kind: ErrorKind::NvmlError(e),
        }
    }
}
