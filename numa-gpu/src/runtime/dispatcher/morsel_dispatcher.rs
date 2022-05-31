// Copyright 2019-2022 Clemens Lutz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::cmp;
use std::iter::Iterator;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) struct MorselDispatcher {
    offset: AtomicUsize,
    data_len: usize,
    cpu_morsel_len: usize,
    gpu_morsel_len: usize,
}

impl MorselDispatcher {
    pub(super) fn new(data_len: usize, cpu_morsel_len: usize, gpu_morsel_len: usize) -> Self {
        let offset = AtomicUsize::new(0);

        Self {
            offset,
            data_len,
            cpu_morsel_len,
            gpu_morsel_len,
        }
    }

    pub(super) fn cpu_iter(&self) -> MorselDispatcherIter<'_> {
        MorselDispatcherIter {
            dispatcher: &self,
            morsel_len: self.cpu_morsel_len,
        }
    }

    pub(super) fn gpu_iter(&self) -> MorselDispatcherIter<'_> {
        MorselDispatcherIter {
            dispatcher: &self,
            morsel_len: self.gpu_morsel_len,
        }
    }
}

pub(super) struct MorselDispatcherIter<'a> {
    dispatcher: &'a MorselDispatcher,
    morsel_len: usize,
}

impl<'a> Iterator for MorselDispatcherIter<'a> {
    type Item = Range<usize>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let morsel = self
            .dispatcher
            .offset
            .fetch_add(self.morsel_len, Ordering::SeqCst);
        if morsel >= self.dispatcher.data_len {
            return None;
        }
        let morsel_len = cmp::min(self.dispatcher.data_len - morsel, self.morsel_len);

        Some(Range {
            start: morsel,
            end: morsel + morsel_len,
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let offset = self.dispatcher.offset.load(Ordering::Relaxed);
        let remaining_morsels = (self.dispatcher.data_len - offset) / self.morsel_len;

        // Lower bound is zero, because we don't know how many morsels the
        // calling worker will receive
        (0, Some(remaining_morsels))
    }
}
