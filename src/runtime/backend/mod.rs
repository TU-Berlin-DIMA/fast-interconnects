/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

mod cuda;
mod hw_info;
mod numa;

// Re-exports
pub use self::cuda::*;
pub use self::hw_info::*;
pub use self::numa::*;

pub trait Backend {}
