/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018-2021 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use crate::error::{ErrorKind, Result};
use crate::runtime::linux_wrapper;
use procfs::CpuInfo;
use rustacuda::device::{Device, DeviceAttribute};
use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::mem;
use std::os::raw::c_int;
use std::path::PathBuf;
use std::str::FromStr;

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

    /// Returns the transparent huge page size
    ///
    /// Example
    /// ```
    /// # use numa_gpu::runtime::hw_info::ProcessorCache;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// ProcessorCache::huge_page_size()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn huge_page_size() -> Result<usize> {
        let mut file = File::open("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size")?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let huge_page_size: usize = contents.trim().parse().map_err(|_| {
            ErrorKind::RuntimeError("Failed to parse Linux huge page size".to_string())
        })?;

        Ok(huge_page_size)
    }
}

impl fmt::Display for ProcessorCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

/// Returns the codename of the current CPU.
///
/// For example: `Intel(R) Core(TM) i7-5600U CPU @ 2.60GHz`
#[cfg(not(target_arch = "powerpc64"))]
pub fn cpu_codename() -> Result<String> {
    let cpu_id = 0;
    Ok(CpuInfo::new()?
        .model_name(cpu_id)
        .expect("Failed to get CPU codename")
        .to_string())
}

/// Returns the codename of the current CPU.
///
/// For example: `POWER9, altivec supported`
#[cfg(target_arch = "powerpc64")]
pub fn cpu_codename() -> Result<String> {
    let cpu_id = 0;
    Ok(CpuInfo::new()?
        .get_info(cpu_id)
        .and_then(|mut m| m.remove("cpu"))
        .expect("Failed to get CPU codename")
        .to_string())
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

    /// Returns the default clock rate of the streaming multiprocessor in megahertz
    fn clock_rate(&self) -> Result<u32>;

    /// Returns the default memory clock rate of the GPU in megahertz
    fn memory_clock_rate(&self) -> Result<u32>;
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

    fn clock_rate(&self) -> Result<u32> {
        Ok(self.get_attribute(DeviceAttribute::ClockRate)? as u32 / 1000)
    }

    fn memory_clock_rate(&self) -> Result<u32> {
        Ok(self.get_attribute(DeviceAttribute::MemoryClockRate)? as u32 / 1000)
    }
}

/// Extends Rustacuda's Device with hardware information obtained from the Nvidia
/// device driver
///
/// Specifically, `NvidiaDriverInfo` maps the GPU device to the NUMA node on
/// IBM POWER systems with NVLink.
pub trait NvidiaDriverInfo {
    /// Returns the NUMA node associated with this GPU device
    ///
    /// # Example
    /// ```
    /// # use numa_gpu::runtime::hw_info::NvidiaDriverInfo;
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # init(CudaFlags::empty())?;
    /// use rustacuda::device::Device;
    /// let device = Device::get_device(0)?;
    /// if let Ok(numa_node) = device.numa_node() {
    ///   println!("NUMA node: {}", numa_node);
    /// }
    /// else {
    ///   println!("GPU isn't a NUMA node");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    fn numa_node(&self) -> Result<u16>;

    /// Returns if the GPU memory is online as a NUMA node
    ///
    /// # Example
    /// ```
    /// # use numa_gpu::runtime::hw_info::NvidiaDriverInfo;
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # init(CudaFlags::empty())?;
    /// use rustacuda::device::Device;
    /// let device = Device::get_device(0)?;
    /// if let Ok(is_numa_mem_online) = device.is_numa_mem_online() {
    ///   println!("Is memory online: {}", is_numa_mem_online);
    /// }
    /// else {
    ///   println!("GPU isn't a NUMA node");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    fn is_numa_mem_online(&self) -> Result<bool>;

    /// Returns the NUMA memory size in bytes as seen by the Linux driver
    ///
    /// # Example
    /// ```
    /// # use numa_gpu::runtime::hw_info::NvidiaDriverInfo;
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # init(CudaFlags::empty())?;
    /// use rustacuda::device::Device;
    /// let device = Device::get_device(0)?;
    /// if let Ok(numa_mem_size) = device.numa_mem_size() {
    ///   println!("Memory size: {}", numa_mem_size);
    /// }
    /// else {
    ///   println!("GPU isn't a NUMA node");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    fn numa_mem_size(&self) -> Result<usize>;

    /// Returns the NUMA node of the CPU socket associated with the GPU
    ///
    /// # Example
    /// ```
    /// # use numa_gpu::runtime::hw_info::NvidiaDriverInfo;
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # init(CudaFlags::empty())?;
    /// use rustacuda::device::Device;
    /// let device = Device::get_device(0)?;
    /// if let Ok(numa_memory_affinity) = device.numa_memory_affinity() {
    ///   println!("NUMA node: {}", numa_memory_affinity);
    /// }
    /// else {
    ///   println!("NUMA memory affinity is unknown");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    fn numa_memory_affinity(&self) -> Result<u16>;
}

impl NvidiaDriverInfo for Device {
    fn numa_node(&self) -> Result<u16> {
        let nvidia_info = NvidiaDriverInternal::from_device(self)?;
        Ok(nvidia_info.numa_node)
    }

    fn is_numa_mem_online(&self) -> Result<bool> {
        let nvidia_info = NvidiaDriverInternal::from_device(self)?;
        Ok(nvidia_info.is_mem_online)
    }

    fn numa_mem_size(&self) -> Result<usize> {
        let nvidia_info = NvidiaDriverInternal::from_device(self)?;
        Ok(nvidia_info.mem_size)
    }

    fn numa_memory_affinity(&self) -> Result<u16> {
        NvidiaDriverInternal::numa_memory_affinity(self)
    }
}

/// A private helper struct to load the GPU driver information
struct NvidiaDriverInternal {
    numa_node: u16,
    is_mem_online: bool,
    mem_size: usize,
}

impl NvidiaDriverInternal {
    fn from_device(device: &Device) -> Result<Self> {
        let device_id = unsafe { mem::transmute_copy::<Device, c_int>(device) };
        let pci_id = Self::pci_id(device)?;

        let mut device_path = PathBuf::from_str("/proc/driver/nvidia/gpus")
            .expect("Failed to convert string to a path");
        device_path.push(pci_id);
        let device_path = device_path;

        let mut information_path = device_path.clone();
        information_path.push("information");
        let information_file = File::open(information_path)?;
        let information_map = linux_wrapper::read_sysfs_file(BufReader::new(&information_file))?;

        let driver_device_id: c_int = information_map
            .get("Device Minor")
            .expect("Failed to get device ID")
            .parse()
            .expect("Failed to parse device ID");
        if driver_device_id != device_id {
            return Err(ErrorKind::RuntimeError(
                "CUDA device ID does not match driver device ID".to_string(),
            )
            .into());
        }

        let mut numa_status_path = device_path.clone();
        numa_status_path.push("numa_status");
        let numa_status_file = File::open(numa_status_path);

        if let Ok(numa_status_file) = numa_status_file {
            let numa_status_map =
                linux_wrapper::read_sysfs_file(BufReader::new(&numa_status_file))?;

            let numa_node = numa_status_map
                .get("Node")
                .expect("Failed to get NUMA node")
                .parse()
                .expect("Failed to parse NUMA node");
            let is_mem_online = numa_status_map
                .get("Status")
                .expect("Failed to get memory status")
                .eq_ignore_ascii_case("online");
            let mem_size_str = numa_status_map
                .get("Size")
                .expect("Failed to get memory size");
            let mem_size =
                usize::from_str_radix(&mem_size_str, 16).expect("Failed to parse memory size");

            Ok(Self {
                numa_node,
                is_mem_online,
                mem_size,
            })
        } else {
            Err(ErrorKind::InvalidArgument(
                "GPU doesn't have a NUMA node; probably does not support NVLink".to_string(),
            )
            .into())
        }
    }

    fn numa_memory_affinity(device: &Device) -> Result<u16> {
        let mut device_path =
            PathBuf::from_str("/sys/bus/pci/devices").expect("Failed to convert string to a path");
        let pci_id = Self::pci_id(device)?;
        device_path.push(pci_id);
        let device_path = device_path;

        let mut numa_node_path = device_path.clone();
        numa_node_path.push("numa_node");
        let mut numa_node_file = File::open(numa_node_path)?;

        let mut contents = String::new();
        numa_node_file.read_to_string(&mut contents)?;

        let numa_node: i32 = contents
            .trim()
            .parse()
            .expect("Failed to parse NUMA node affinity");

        if numa_node < 0 {
            Err(
                ErrorKind::RuntimeError("NUMA memory affinity of GPU not available".to_string())
                    .into(),
            )
        } else {
            Ok(numa_node as u16)
        }
    }

    fn pci_id(device: &Device) -> Result<String> {
        let pci_domain_id = device.get_attribute(DeviceAttribute::PciDomainId)?;
        let pci_bus_id = device.get_attribute(DeviceAttribute::PciBusId)?;
        let pci_device_id = device.get_attribute(DeviceAttribute::PciDeviceId)?;
        let pci_function_id: u32 = 0;

        let pci_id = format!(
            "{:04x}:{:02x}:{:02x}.{:1x}",
            pci_domain_id, pci_bus_id, pci_device_id, pci_function_id
        );

        Ok(pci_id)
    }
}
