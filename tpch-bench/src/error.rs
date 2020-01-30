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
    IoError(IoError),
    Msg(String),
    NumaGpuError(NumaGpuError),
    IntegerOverflow(String),
    InvalidArgument(String),
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

impl std::error::Error for Error {
    fn description(&self) -> &str {
        match self.kind {
            ErrorKind::CsvError(ref e) => e.description(),
            ErrorKind::CudaError(ref e) => e.description(),
            ErrorKind::IoError(ref e) => e.description(),
            ErrorKind::NumaGpuError(ref e) => e.description(),
            ErrorKind::IntegerOverflow(ref s) => s.as_str(),
            ErrorKind::InvalidArgument(ref s) => s.as_str(),
            ErrorKind::Msg(ref s) => s.as_str(),
            ErrorKind::RuntimeError(ref s) => s.as_str(),
        }
    }

    fn cause(&self) -> Option<&dyn std::error::Error> {
        None
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.kind, f)
    }
}

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Self {
        Self { kind }
    }
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ErrorKind::CsvError(ref e) => e.fmt(f),
            ErrorKind::CudaError(ref e) => e.fmt(f),
            ErrorKind::IoError(ref e) => e.fmt(f),
            ErrorKind::NumaGpuError(ref e) => e.fmt(f),
            ErrorKind::IntegerOverflow(ref s) => write!(f, "IntegerOverflow: {}", s),
            ErrorKind::InvalidArgument(ref s) => write!(f, "InvalidArgument: {}", s),
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
