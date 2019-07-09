extern crate accel;

use self::accel::device::Device as CudaDevice;

use crate::types::{DeviceId, Warp, SM};

pub struct DeviceProp {}

impl DeviceProp {
    /// Number of threads per warp
    ///
    /// Returns None for non-CUDA devices.
    pub fn warp_size(dev: DeviceId) -> Option<Warp> {
        match dev {
            DeviceId::Cpu(_) => None,
            DeviceId::Gpu(id) => {
                let ws = CudaDevice::set(id)
                    .expect("Couldn't set device")
                    .get_property()
                    .expect("Couldn't get CUDA warp size")
                    .warpSize;
                Some(Warp(ws))
            }
        }
    }

    /// Amount of shared multiprocessors
    ///
    /// Returns None for non-CUDA devices.
    pub fn multi_processor_count(dev: DeviceId) -> Option<SM> {
        match dev {
            DeviceId::Cpu(_) => None,
            DeviceId::Gpu(id) => {
                let mpc = CudaDevice::set(id)
                    .expect("Couldn't set device")
                    .get_property()
                    .expect("Couldn't get CUDA multi-processor count")
                    .multiProcessorCount;
                Some(SM(mpc))
            }
        }
    }
}
