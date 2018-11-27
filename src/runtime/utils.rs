/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

extern crate raw_cpuid;

#[cfg(target_arch = "x86_64")]
pub fn cpu_codename() -> String {
    let cpuid = self::raw_cpuid::CpuId::new();
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
