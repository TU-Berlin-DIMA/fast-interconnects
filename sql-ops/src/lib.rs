/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019-2021, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

pub mod error;
pub mod join;
pub mod partition;
pub mod prefix_scan;

#[allow(dead_code)]
pub(crate) mod constants {
    include!(concat!(env!("OUT_DIR"), "/constants.rs"));
}
