/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019-2020, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use datagen::error::Error as DatagenError;
use numa_gpu::error::Error as NumaGpuError;
use rayon::ThreadPoolBuildError;
use rustacuda::error::CudaError;
use sql_ops::error::Error as SqlOpsError;
use std::convert::From;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
#[allow(dead_code)]
pub enum ErrorKind {
    CsvError(csv::Error),
    CudaError(CudaError),
    DatagenError(DatagenError),
    IntegerOverflow(String),
    InvalidArgument(String),
    InvalidConversion(String),
    IoError(::std::io::Error),
    LogicError(String),
    NumaGpuError(NumaGpuError),
    RuntimeError(String),
    SqlOpsError(SqlOpsError),
    RayonThreadPoolBuildError(ThreadPoolBuildError),
}

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.kind, f)
    }
}

impl From<csv::Error> for Error {
    fn from(error: csv::Error) -> Self {
        Self {
            kind: ErrorKind::CsvError(error),
        }
    }
}

impl From<CudaError> for Error {
    fn from(error: CudaError) -> Self {
        Self {
            kind: ErrorKind::CudaError(error),
        }
    }
}

impl From<DatagenError> for Error {
    fn from(error: DatagenError) -> Self {
        Self {
            kind: ErrorKind::DatagenError(error),
        }
    }
}

impl From<::std::io::Error> for Error {
    fn from(error: ::std::io::Error) -> Self {
        Self {
            kind: ErrorKind::IoError(error),
        }
    }
}

impl From<NumaGpuError> for Error {
    fn from(error: NumaGpuError) -> Self {
        Self {
            kind: ErrorKind::NumaGpuError(error),
        }
    }
}

impl From<SqlOpsError> for Error {
    fn from(error: SqlOpsError) -> Self {
        Self {
            kind: ErrorKind::SqlOpsError(error),
        }
    }
}

impl From<ThreadPoolBuildError> for Error {
    fn from(error: ThreadPoolBuildError) -> Self {
        Self {
            kind: ErrorKind::RayonThreadPoolBuildError(error),
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
            ErrorKind::CsvError(ref e) => e.fmt(f),
            ErrorKind::CudaError(ref e) => e.fmt(f),
            ErrorKind::DatagenError(ref e) => e.fmt(f),
            ErrorKind::IntegerOverflow(ref s) => write!(f, "Integer overflow: {}", s),
            ErrorKind::InvalidArgument(ref s) => write!(f, "Invalid argument: {}", s),
            ErrorKind::InvalidConversion(ref s) => write!(f, "Invalid conversion: {}", s),
            ErrorKind::IoError(ref e) => e.fmt(f),
            ErrorKind::LogicError(ref s) => write!(f, "Logic error: {}", s),
            ErrorKind::NumaGpuError(ref e) => e.fmt(f),
            ErrorKind::RuntimeError(ref s) => write!(f, "Runtime error: {}", s),
            ErrorKind::SqlOpsError(ref e) => e.fmt(f),
            ErrorKind::RayonThreadPoolBuildError(ref e) => e.fmt(f),
        }
    }
}
