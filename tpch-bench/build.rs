/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2020 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

fn main() {
    // Add CPP utils
    cc::Build::new()
        .compiler("gcc-8")
        .cpp(true)
        // Note: -march not supported by GCC-7 on Power9, use -mcpu instead
        .flag("-std=c++11")
        .debug(true)
        .flag_if_supported("-mcpu=native")
        .flag_if_supported("-march=native")
        .flag("-mtune=native")
        // Note: Enables x86 intrinsic translations on POWER9
        // See also "Linux on Power Porting Guide - Vector Intrinsics"
        .define("NO_WARN_X86_INTRINSICS", None)
        .pic(true)
        .file("cpputils/queries.cpp")
        .compile("libcpputils.a");
}
