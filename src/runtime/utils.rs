/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

/// Ensure that memory is backed by physical pages
///
/// Operating systems (such as Linux) sometimes do not back memory filled with
/// identical data with physical pages. Instead, the OS allocates only a single
/// page and replicates it in virtual memory. Thus, in benchmarks the page is
/// cached in L1, which leads to unrealistically high performance. This effect
/// can be observed by profiling, e.g., with Linux perf or Intel VTune.
///
/// This function forces the OS to back all pages with physical memory by
/// writing non-uniform data into each page.
#[inline(never)]
pub fn ensure_physically_backed<T: std::convert::From<u16>>(data: &mut [T]) {
    data.iter_mut()
        .by_ref()
        .enumerate()
        .for_each(|(i, x)| *x = (i as u16).into());
}
