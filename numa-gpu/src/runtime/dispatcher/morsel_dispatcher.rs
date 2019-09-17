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
use std::iter::Iterator;
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

    pub(super) fn iter(&self) -> MorselDispatcherIter<'_> {
        MorselDispatcherIter { dispatcher: &self }
    }
}

pub(super) struct MorselDispatcherIter<'a> {
    dispatcher: &'a MorselDispatcher,
}

impl<'a> Iterator for MorselDispatcherIter<'a> {
    type Item = Range<usize>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let morsel = self
            .dispatcher
            .offset
            .fetch_add(self.dispatcher.morsel_len, Ordering::SeqCst);
        if morsel >= self.dispatcher.data_len {
            return None;
        }
        let morsel_len = cmp::min(
            self.dispatcher.data_len - morsel,
            self.dispatcher.morsel_len,
        );

        Some(Range {
            start: morsel,
            end: morsel + morsel_len,
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let offset = self.dispatcher.offset.load(Ordering::Relaxed);
        let remaining_morsels = (self.dispatcher.data_len - offset) / self.dispatcher.morsel_len;

        // Lower bound is zero, because we don't know how many morsels the
        // calling worker will receive
        (0, Some(remaining_morsels))
    }
}
