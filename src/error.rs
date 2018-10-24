extern crate accel;

// use std::os::raw::c_int;
// use std::ffi::CStr;
use self::accel::error::Error as AccelError;

error_chain!{
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
            AccelError::cudaError(e) => {
                Self::from_kind(ErrorKind::Cuda(format!("{:?}", e)))
            },
            AccelError::cudaRuntimeError(e) => {
                Self::from_kind(ErrorKind::CudaRuntime(format!("{:?}", e)))
            },
            AccelError::cublasError(e) => {
                Self::from_kind(ErrorKind::CuBlas(format!("{:?}", e)))
            },
            // _ => panic!("Unexpected Accel error!"),
        }

    }
}
