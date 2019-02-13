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

use self::accel::error::Error as AccelError;

error_chain! {
    errors {
        Cuda(msg: String) {
            description("A CUDA error occured")
            display("Aborted with: {}", msg)
        }
        CudaRuntime(msg: String) {
            description("A CUDA error occured")
            display("Aborted with: {}", msg)
        }
        CuBlas(msg: String) {
            description("A CUDA error occured")
            display("Aborted with: {}", msg)
        }
    }

    foreign_links {
        Io(::std::io::Error);
    }
}

impl From<AccelError> for Error {
    fn from(error: AccelError) -> Self {
        match &error {
            AccelError::cudaError(e) => Self::from_kind(ErrorKind::Cuda(format!("{:?}", e))),
            AccelError::cudaRuntimeError(e) => {
                Self::from_kind(ErrorKind::CudaRuntime(format!("{:?}", e)))
            }
            AccelError::cublasError(e) => Self::from_kind(ErrorKind::CuBlas(format!("{:?}", e))),
        }
    }
}
