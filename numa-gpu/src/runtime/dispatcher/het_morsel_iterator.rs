/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use super::morsel_dispatcher::MorselDispatcher;
use super::HetMorselExecutor;
use crate::error::*;
use crate::runtime::memory::LaunchableMem;
use crate::runtime::memory::LaunchableSlice;
use rustacuda::memory::DeviceCopy;
use rustacuda::stream::{Stream, StreamFlags};
use std::sync::Arc;

pub trait IntoHetMorselIterator<'a> {
    /// The type of the iterator to produce.
    type Iter;

    /// Create an iterator from a value.
    ///
    /// See module-level documentation for details.
    fn into_het_morsel_iter<'h: 'a>(&'a mut self, executor: &'h HetMorselExecutor) -> Self::Iter;
}

impl<'a, 'r, 's, R, S> IntoHetMorselIterator<'a> for (&'r mut [R], &'s mut [S])
where
    'r: 'a,
    's: 'a,
    R: 'a + Copy + DeviceCopy + Send + Sync,
    S: 'a + Copy + DeviceCopy + Send + Sync,
{
    type Iter = HetMorselIterator2<'a, R, S>;

    fn into_het_morsel_iter<'h: 'a>(&'a mut self, executor: &'h HetMorselExecutor) -> Self::Iter {
        assert_eq!(self.0.len(), self.1.len());

        Self::Iter {
            data: (self.0, self.1),
            executor,
        }
    }
}

pub struct HetMorselIterator2<'a, R, S> {
    data: (&'a mut [R], &'a mut [S]),
    executor: &'a HetMorselExecutor,
}

impl<'a, R, S> HetMorselIterator2<'a, R, S> {
    pub fn fold<CpuF, GpuF>(&mut self, cpu_f: CpuF, gpu_f: GpuF) -> Result<()>
    where
        R: Copy + DeviceCopy + Send + Sync,
        S: Copy + DeviceCopy + Send + Sync,
        CpuF: Fn((&[R], &[S])) -> Result<()> + Send + Sync,
        GpuF: Fn((LaunchableSlice<R>, LaunchableSlice<S>), &Stream) -> Result<()> + Send + Sync,
    {
        let dispatcher = MorselDispatcher::new(self.data.0.len(), self.executor.morsel_len);
        let dispatcher_ref = &dispatcher;
        let executor = &self.executor;

        let cpu_workers = self.executor.cpu_workers;
        let gpu_workers = self.executor.gpu_workers;

        let ro_data = (self.data.0.as_ref(), self.data.1.as_ref());

        executor.cpu_thread_pool.scope(move |cpu_scope| {
            executor.gpu_thread_pool.scope(move |gpu_scope| {
                let cpu_af = Arc::new(cpu_f);
                let gpu_af = Arc::new(gpu_f);

                for _ in 0..cpu_workers {
                    let af = cpu_af.clone();

                    cpu_scope.spawn(move |_| {
                        Self::cpu_worker(dispatcher_ref, ro_data, af)
                            .expect("Failed to run CPU worker");
                    });
                }

                for _ in 0..gpu_workers {
                    let af = gpu_af.clone();

                    gpu_scope.spawn(move |_| {
                        Self::gpu_worker(dispatcher_ref, ro_data, af)
                            .expect("Failed to run GPU worker");
                    });
                }
            });
        });

        Ok(())
    }

    fn cpu_worker<F>(dispatcher: &MorselDispatcher, data: (&[R], &[S]), f: Arc<F>) -> Result<()>
    where
        F: Fn((&[R], &[S])) -> Result<()>,
    {
        for morsel in dispatcher.iter() {
            f((&data.0[morsel.clone()], &data.1[morsel.clone()]))?;
        }

        Ok(())
    }

    fn gpu_worker<F>(dispatcher: &MorselDispatcher, data: (&[R], &[S]), f: Arc<F>) -> Result<()>
    where
        F: Fn((LaunchableSlice<R>, LaunchableSlice<S>), &Stream) -> Result<()>,
    {
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        for morsel in dispatcher.iter() {
            f(
                (
                    data.0[morsel.clone()].as_launchable_slice(),
                    data.1[morsel.clone()].as_launchable_slice(),
                ),
                &stream,
            )?
        }

        stream.synchronize()?;

        Ok(())
    }
}
