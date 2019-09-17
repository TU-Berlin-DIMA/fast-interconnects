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
use crate::error::*;
use crate::runtime::memory::LaunchableMem;
use crate::runtime::memory::LaunchableSlice;
use rayon::{ThreadPool, ThreadPoolBuilder};
use rustacuda::context::CurrentContext;
use rustacuda::memory::DeviceCopy;
use rustacuda::stream::{Stream, StreamFlags};
use std::mem::size_of;
use std::sync::Arc;

pub trait IntoHetMorselIterator<'a> {
    /// The type of the iterator to produce.
    type Iter;

    /// Create an iterator from a value.
    ///
    /// See module-level documentation for details.
    fn into_het_morsel_iter(&'a mut self) -> Result<Self::Iter>;
}

impl<'a, 'r, 's, R, S> IntoHetMorselIterator<'a> for (&'r mut [R], &'s mut [S])
where
    'r: 'a,
    's: 'a,
    R: 'a + Copy + DeviceCopy + Send + Sync,
    S: 'a + Copy + DeviceCopy + Send + Sync,
{
    type Iter = HetMorselIterator2<'a, R, S>;

    fn into_het_morsel_iter(&'a mut self) -> Result<Self::Iter> {
        assert_eq!(self.0.len(), self.1.len());

        let morsel_bytes = 16_usize * 2_usize.pow(20);
        let morsel_len = morsel_bytes / (size_of::<R>() + size_of::<S>());

        let cpu_workers = 2;
        let gpu_workers = 1;

        let dispatcher = MorselDispatcher::new(self.0.len(), morsel_len);
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(cpu_workers + gpu_workers)
            .start_handler(|_thread_index| {
                // FIXME: pin each thread to a core
            })
            .build()?;

        Ok(Self::Iter {
            data: (self.0, self.1),
            thread_pool,
            dispatcher,
            cpu_workers,
            gpu_workers,
        })
    }
}

pub struct HetMorselIterator2<'a, R, S> {
    data: (&'a mut [R], &'a mut [S]),
    thread_pool: ThreadPool,
    dispatcher: MorselDispatcher,
    cpu_workers: usize,
    gpu_workers: usize,
}

impl<'a, R, S> HetMorselIterator2<'a, R, S> {
    pub fn fold<CpuF, GpuF>(&mut self, cpu_f: CpuF, gpu_f: GpuF) -> Result<()>
    where
        R: Copy + DeviceCopy + Send + Sync,
        S: Copy + DeviceCopy + Send + Sync,
        CpuF: Fn((&[R], &[S])) -> Result<()> + Send + Sync,
        GpuF: Fn((LaunchableSlice<R>, LaunchableSlice<S>), &Stream) -> Result<()> + Send + Sync,
    {
        let dispatcher = &self.dispatcher;
        let cpu_workers = self.cpu_workers;
        let gpu_workers = self.gpu_workers;

        let ro_data = (self.data.0.as_ref(), self.data.1.as_ref());
        let unowned_context = CurrentContext::get_current()?;

        self.thread_pool.scope(move |scope| {
            let cpu_af = Arc::new(cpu_f);
            let gpu_af = Arc::new(gpu_f);

            for _ in 0..cpu_workers {
                let af = cpu_af.clone();

                scope.spawn(move |_| {
                    cpu_worker(dispatcher, ro_data, af).expect("Failed to run CPU worker");
                });
            }

            for _ in 0..gpu_workers {
                let af = gpu_af.clone();
                let thread_context = unowned_context.clone();

                scope.spawn(move |_| {
                    CurrentContext::set_current(&thread_context)
                        .expect("Failed to set CUDA context in GPU worker thread");

                    gpu_worker(dispatcher, ro_data, af).expect("Failed to run GPU worker");
                });
            }
        });

        Ok(())
    }
}

fn cpu_worker<R, S, F>(dispatcher: &MorselDispatcher, data: (&[R], &[S]), f: Arc<F>) -> Result<()>
where
    R: Copy,
    S: Copy,
    F: Fn((&[R], &[S])) -> Result<()>,
{
    for morsel in dispatcher.iter() {
        f((&data.0[morsel.clone()], &data.1[morsel.clone()]))?;
    }

    Ok(())
}

fn gpu_worker<R, S, F>(dispatcher: &MorselDispatcher, data: (&[R], &[S]), f: Arc<F>) -> Result<()>
where
    R: DeviceCopy,
    S: DeviceCopy,
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
