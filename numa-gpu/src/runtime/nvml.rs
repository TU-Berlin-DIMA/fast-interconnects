/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2020 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

pub use nvml_impl::*;

#[cfg(target_arch = "aarch64")]
mod nvml_impl {
    use std::fmt;

    pub struct ThrottleReasons;

    impl fmt::Display for ThrottleReasons {
        fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
            Ok(())
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod nvml_impl {
    use crate::error::{ErrorKind, Result};
    use crate::runtime::linux_wrapper::{numa_node_of_cpu, CpuSet};
    use nvml_wrapper::bitmasks::device::ThrottleReasons as NvmlTR;
    use nvml_wrapper::device::Device;
    use nvml_wrapper::enum_wrappers::device::Clock as GpuClock;
    use nvml_wrapper::error::NvmlError;
    use std::convert::From;
    use std::fmt;
    use std::mem;
    use std::os::raw::c_ulong;

    pub struct ThrottleReasons(NvmlTR);

    impl fmt::Display for ThrottleReasons {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let possible_reasons = [
                (NvmlTR::GPU_IDLE, "GPU_Idle|"),
                (
                    NvmlTR::APPLICATIONS_CLOCKS_SETTING,
                    "Applications_Clocks_Settings|",
                ),
                (NvmlTR::SW_POWER_CAP, "SW_Power_Cap|"),
                (NvmlTR::HW_SLOWDOWN, "HW_Slowdown|"),
                (NvmlTR::SYNC_BOOST, "Sync_Boost|"),
                (NvmlTR::SW_THERMAL_SLOWDOWN, "SW_Thermal_Slowdown|"),
                (NvmlTR::HW_THERMAL_SLOWDOWN, "HW_Thermal_Slowdown|"),
                (NvmlTR::HW_POWER_BRAKE_SLOWDOWN, "HW_Power_Brake_Slowdown|"),
                (NvmlTR::DISPLAY_CLOCK_SETTING, "Display_Clock_Setting|"),
                (NvmlTR::NONE, "None|"),
            ];

            let mut reasons: String = possible_reasons
                .iter()
                .filter_map(|&(reason, reason_str)| {
                    if self.0.contains(reason) {
                        Some(reason_str)
                    } else {
                        None
                    }
                })
                .collect();
            reasons.pop();

            write!(f, "{}", reasons)
        }
    }

    impl From<NvmlTR> for ThrottleReasons {
        fn from(other: NvmlTR) -> Self {
            Self(other)
        }
    }

    /// Extra features for GPU devices with NVML
    pub trait NvmlDeviceExtra {
        /// Returns the NUMA memory affinity of the GPU device
        fn numa_mem_affinity(&self) -> Result<u16>;
    }

    impl NvmlDeviceExtra for Device<'_> {
        fn numa_mem_affinity(&self) -> Result<u16> {
            let mut cpu_set = CpuSet::new();
            let capacity = cpu_set.bytes() / mem::size_of::<c_ulong>();

            // Supporting only 64-bit systems, 32-bit will require bit-shifting
            assert_eq!(mem::size_of::<c_ulong>(), mem::size_of::<u64>());

            let cpu_affinity: Vec<_> = self
                .cpu_affinity(capacity)
                .map_err(|e| ErrorKind::RuntimeError(e.to_string()))?
                .iter()
                .map(|&ulong| ulong as u64)
                .collect();
            cpu_set
                .as_mut_slice()
                .iter_mut()
                .zip(cpu_affinity.iter())
                .for_each(|(set_item, nvml_item)| *set_item = *nvml_item);
            let first_cpu = (0..=cpu_set.max_id())
                .find(|&id| cpu_set.is_set(id))
                .ok_or_else(|| ErrorKind::RuntimeError("GPU has no CPU affinity".to_string()))?;
            let memory_affinity = numa_node_of_cpu(first_cpu)?;

            Ok(memory_affinity)
        }
    }

    pub trait DeviceClocks {
        fn set_max_gpu_clocks(&mut self) -> Result<()>;
        fn set_default_gpu_clocks(&mut self) -> Result<()>;
    }

    impl DeviceClocks for Device<'_> {
        fn set_max_gpu_clocks(&mut self) -> Result<()> {
            let max_graphics_mhz = self
                .max_clock_info(GpuClock::Graphics)
                .map_err(|e| ErrorKind::RuntimeError(e.to_string()))?;
            let max_memory_mhz = self
                .max_clock_info(GpuClock::Memory)
                .map_err(|e| ErrorKind::RuntimeError(e.to_string()))?;

            if let Err(error) = self.set_applications_clocks(max_memory_mhz, max_graphics_mhz) {
                match error {
                    NvmlError::NotSupported => eprintln!(
                        "WARNING: Your GPU doesn't support setting the clock \
                        rate. Measurements may be inaccurate."
                    ),
                    NvmlError::NoPermission => {
                        return Err(ErrorKind::RuntimeError(
                            "Failed to set GPU clock rate. Setting changes \
                            might be restricted to root. Try running: sudo \
                            nvidia-smi --persistence-mode=ENABLED; sudo \
                            nvidia-smi \
                            --applications-clocks-permission=UNRESTRICTED"
                                .to_string(),
                        )
                        .into())
                    }
                    _ => return Err(ErrorKind::RuntimeError(error.to_string()).into()),
                }
            }

            Ok(())
        }

        fn set_default_gpu_clocks(&mut self) -> Result<()> {
            if let Err(error) = self.reset_applications_clocks() {
                match error {
                    NvmlError::NotSupported => { /* ignore */ }
                    NvmlError::NoPermission => { /* ignore */ }
                    _ => return Err(ErrorKind::RuntimeError(error.to_string()).into()),
                }
            }

            Ok(())
        }
    }
}
