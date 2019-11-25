/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019, Clemens Lutz <lutzcle@cml.li>
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
    use nvml_wrapper::bitmasks::device::ThrottleReasons as NvmlTR;
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
}
