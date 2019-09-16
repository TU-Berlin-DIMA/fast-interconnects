/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use std::cmp;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) struct MorselDispatcher {
    offset: AtomicUsize,
    data_len: usize,
    morsel_len: usize,
}

impl MorselDispatcher {
    pub(super) fn new(data_len: usize, morsel_len: usize) -> Self {
        let offset = AtomicUsize::new(0);

        Self {
            offset,
            data_len,
            morsel_len,
        }
    }

    pub(super) fn dispatch(&self) -> Option<Range<usize>> {
        let morsel = self.offset.fetch_add(self.morsel_len, Ordering::SeqCst);
        if morsel >= self.data_len {
            return None;
        }
        let morsel_len = cmp::min(self.data_len - morsel, self.morsel_len);

        Some(Range {
            start: morsel,
            end: morsel + morsel_len,
        })
    }
}
