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
    use nvml_wrapper::bitmasks::device::ThrottleReasons as NvmlTR;
    use nvml_wrapper::device::Device;
    use nvml_wrapper::enum_wrappers::device::Clock as GpuClock;
    use nvml_wrapper::error::NvmlError;
    use std::convert::From;
    use std::fmt;

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
