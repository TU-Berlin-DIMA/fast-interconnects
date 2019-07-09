/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use rustacuda::device::{Device, DeviceAttribute};

use std::fmt;

use crate::error::Result;

pub struct ProcessorCache {}

impl ProcessorCache {
    #[allow(non_snake_case)]
    pub fn L1D_size() -> usize {
        let size = unsafe { libc::sysconf(libc::_SC_LEVEL1_DCACHE_SIZE) };
        if size == -1 {
            // TODO: std::Option
        }
        size as usize
    }

    #[allow(non_snake_case)]
    pub fn L2_size() -> usize {
        let size = unsafe { libc::sysconf(libc::_SC_LEVEL2_CACHE_SIZE) };
        size as usize
    }

    #[allow(non_snake_case)]
    pub fn L3_size() -> usize {
        let size = unsafe { libc::sysconf(libc::_SC_LEVEL3_CACHE_SIZE) };
        size as usize
    }

    pub fn page_size() -> usize {
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        size as usize
    }
}

impl fmt::Display for ProcessorCache {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "L1 cache size: {}\nL2 cache size: {}\nL3 cache size: {}\npage size: {}",
            Self::L1D_size(),
            Self::L2_size(),
            Self::L3_size(),
            Self::page_size()
        )
    }
}

#[cfg(target_arch = "x86_64")]
pub fn cpu_codename() -> String {
    extern crate raw_cpuid;
    let cpuid = raw_cpuid::CpuId::new();
    cpuid
        .get_extended_function_info()
        .as_ref()
        .and_then(|i| i.processor_brand_string())
        .map_or_else(|| String::from("unknown x86-64"), |s| String::from(s))
}

#[cfg(target_arch = "powerpc64")]
pub fn cpu_codename() -> String {
    String::from("POWER9")
}

/// Extends Rustacuda's Device with methods that provide additional hardware
/// information.
pub trait CudaDeviceInfo {
    /// Returns the number of cores per streaming multiprocessor
    fn sm_cores(&self) -> Result<u32>;

    /// Returns the total number of cores that are in the device
    fn cores(&self) -> Result<u32>;

    /// Returns `true` if concurrent managed access is supported by the device
    fn concurrent_managed_access(&self) -> Result<bool>;
}

impl CudaDeviceInfo for Device {
    fn sm_cores(&self) -> Result<u32> {
        let major = self.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
        let minor = self.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;

        Ok(match (major, minor) {
            (3, 0) => 192, // Kepler Generation (SM 3.0) GK10x class
            (3, 2) => 192, // Kepler Generation (SM 3.2) GK10x class
            (3, 5) => 192, // Kepler Generation (SM 3.5) GK11x class
            (3, 7) => 192, // Kepler Generation (SM 3.7) GK21x class
            (5, 0) => 128, // Maxwell Generation (SM 5.0) GM10x class
            (5, 2) => 128, // Maxwell Generation (SM 5.2) GM20x class
            (5, 3) => 128, // Maxwell Generation (SM 5.3) GM20x class
            (6, 0) => 64,  // Pascal Generation (SM 6.0) GP100 class
            (6, 1) => 128, // Pascal Generation (SM 6.1) GP10x class
            (6, 2) => 128, // Pascal Generation (SM 6.2) GP10x class
            (7, 0) => 64,  // Volta Generation (SM 7.0) GV100 class
            _ => unreachable!("Unsupported Core"),
        })
    }

    fn cores(&self) -> Result<u32> {
        Ok(self.sm_cores()? * self.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32)
    }

    fn concurrent_managed_access(&self) -> Result<bool> {
        let is_supported = self.get_attribute(DeviceAttribute::ConcurrentManagedAccess)?;

        Ok(match is_supported {
            1 => true,
            0 => false,
            _ => unreachable!("Concurrent menaged access should return 0 or 1"),
        })
    }
}
