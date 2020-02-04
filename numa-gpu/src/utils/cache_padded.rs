/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2020, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

/// Cache pad a value to avoid false sharing between threads.
///
/// Pads the value to 128 bytes, because Intel Sandy Bridge and later pre-fetch
/// two 64-byte cache lines, and IBM POWER processors have 128-byte cache lines.
#[derive(Clone, Copy, Default, Hash, PartialEq, Eq)]
#[repr(align(128))]
pub struct CachePadded<T> {
    pub value: T,
}
