// Copyright 2019-2022 Clemens Lutz
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
    Msg(String),
    IntegerOverflow(String),
    InvalidArgument(String),
}

#[derive(Debug)]
pub struct Error {
    pub kind: ErrorKind,
}

impl Error {
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }
}

impl std::error::Error for Error {
    fn description(&self) -> &str {
        match self.kind {
            ErrorKind::IntegerOverflow(ref s) => s.as_str(),
            ErrorKind::InvalidArgument(ref s) => s.as_str(),
            ErrorKind::Msg(ref s) => s.as_str(),
        }
    }

    fn cause(&self) -> Option<&dyn std::error::Error> {
        None
    }
}

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
            ErrorKind::IntegerOverflow(ref s) => write!(f, "IntegerOverflow: {}", s),
            ErrorKind::InvalidArgument(ref s) => write!(f, "InvalidArgument: {}", s),
            ErrorKind::Msg(ref s) => write!(f, "Msg: {}", s),
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
