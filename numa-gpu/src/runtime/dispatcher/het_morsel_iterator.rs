/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019-2021 Clemens Lutz
 * Author: Clemens Lutz <lutzcle@cml.li>
 */

use super::morsel_dispatcher::MorselDispatcher;
use super::HetMorselExecutor;
use crate::error::*;
use crate::runtime::memory::LaunchableMem;
use crate::runtime::memory::LaunchableSlice;
use rustacuda::memory::DeviceCopy;
use rustacuda::stream::Stream;
use std::mem::{self, size_of, MaybeUninit};
use std::sync::Arc;

pub trait IntoHetMorselIterator<'a> {
    /// The type of the iterator to produce.
    type Iter;

    /// Create an iterator from a value.
    ///
    /// See module-level documentation for details.
    fn into_het_morsel_iter<'h: 'a>(
        &'a mut self,
        executor: &'h mut HetMorselExecutor,
    ) -> Self::Iter;
}

impl<'a, 'r, 's, R, S> IntoHetMorselIterator<'a> for (&'r mut [R], &'s mut [S])
where
    'r: 'a,
    's: 'a,
    R: 'a + Copy + DeviceCopy + Send + Sync,
    S: 'a + Copy + DeviceCopy + Send + Sync,
{
    type Iter = HetMorselIterator2<'a, R, S>;

    fn into_het_morsel_iter<'h: 'a>(
        &'a mut self,
        executor: &'h mut HetMorselExecutor,
    ) -> Self::Iter {
        assert_eq!(self.0.len(), self.1.len());

        Self::Iter {
            data: (self.0, self.1),
            executor,
        }
    }
}

pub struct HetMorselIterator2<'a, R, S> {
    data: (&'a mut [R], &'a mut [S]),
    executor: &'a mut HetMorselExecutor,
}

pub struct StatefulHetMorselIterator2<'a, R, S, CWS: Send, GWS: Send> {
    data: (&'a mut [R], &'a mut [S]),
    executor: &'a mut HetMorselExecutor,
    cpu_worker_states: Vec<CWS>,
    gpu_worker_states: Vec<GWS>,
}

impl<'a, R, S> HetMorselIterator2<'a, R, S> {
    pub fn with_state<CpuF, GpuF, CWS, GWS>(
        self,
        cpu_init: CpuF,
        gpu_init: GpuF,
    ) -> Result<StatefulHetMorselIterator2<'a, R, S, CWS, GWS>>
    where
        CpuF: Fn(u16) -> Result<CWS> + Send + Sync,
        GpuF: Fn(u16, &Stream) -> Result<GWS> + Send + Sync,
        CWS: Send,
        GWS: Send,
    {
        let cpu_thread_pool = &mut self.executor.cpu_thread_pool;
        let gpu_thread_pool = &mut self.executor.gpu_thread_pool;

        let cpu_workers = self.executor.cpu_workers;
        let gpu_workers = self.executor.gpu_workers;
        let streams = &mut self.executor.streams;

        let mut cpu_uninit_states: Vec<MaybeUninit<CWS>> =
            (0..cpu_workers).map(|_| MaybeUninit::uninit()).collect();
        let mut gpu_uninit_states: Vec<MaybeUninit<GWS>> =
            (0..gpu_workers).map(|_| MaybeUninit::uninit()).collect();

        let cus = &mut cpu_uninit_states;
        let gus = &mut gpu_uninit_states;

        cpu_thread_pool.scope(move |cpu_scope| {
            gpu_thread_pool.scope(move |gpu_scope| {
                let cpu_af = Arc::new(cpu_init);
                let gpu_af = Arc::new(gpu_init);

                for (thread_id, uninit_state) in (0..).zip(cus.iter_mut()) {
                    let af = cpu_af.clone();

                    cpu_scope.spawn(move |_| {
                        let state: CWS = af(thread_id).expect("Failed to run CPU worker");
                        *uninit_state = MaybeUninit::new(state);
                    });
                }

                for ((thread_id, uninit_state), stream) in
                    (0..).zip(gus.iter_mut()).zip(streams.iter_mut())
                {
                    let af = gpu_af.clone();

                    gpu_scope.spawn(move |_| {
                        let state: GWS = af(thread_id, stream).expect("Failed to run GPU worker");
                        *uninit_state = MaybeUninit::new(state);
                    });
                }
            });
        });

        let cpu_worker_states: Vec<CWS> = unsafe {
            cpu_uninit_states
                .into_iter()
                .map(|state| state.assume_init())
                .collect()
        };
        let gpu_worker_states: Vec<GWS> = unsafe {
            gpu_uninit_states
                .into_iter()
                .map(|state| state.assume_init())
                .collect()
        };

        Ok(StatefulHetMorselIterator2 {
            data: self.data,
            executor: self.executor,
            cpu_worker_states,
            gpu_worker_states,
        })
    }

    pub fn fold<CpuF, GpuF>(&mut self, cpu_f: CpuF, gpu_f: GpuF) -> Result<()>
    where
        R: Copy + DeviceCopy + Send + Sync,
        S: Copy + DeviceCopy + Send + Sync,
        CpuF: Fn((&[R], &[S])) -> Result<()> + Send + Sync,
        GpuF: Fn((LaunchableSlice<'_, R>, LaunchableSlice<'_, S>), &Stream) -> Result<()>
            + Send
            + Sync,
    {
        // Create dummy state
        let cpu_worker_states = vec![(); self.executor.cpu_workers];
        let gpu_worker_states = vec![(); self.executor.gpu_workers];

        let r: &mut [R] = self.data.0;
        let s: &mut [S] = self.data.1;
        let data = (r, s);

        let mut iter = StatefulHetMorselIterator2 {
            data,
            executor: self.executor,
            cpu_worker_states,
            gpu_worker_states,
        };

        // Wrap StatefulHetMorselIterator2::fold with dummy state
        iter.fold(|data, _| cpu_f(data), |data, _, stream| gpu_f(data, stream))
    }
}

impl<'a, R, S, CWS: Send, GWS: Send> StatefulHetMorselIterator2<'a, R, S, CWS, GWS> {
    pub fn fold<CpuF, GpuF>(&mut self, cpu_f: CpuF, gpu_f: GpuF) -> Result<()>
    where
        R: Copy + DeviceCopy + Send + Sync,
        S: Copy + DeviceCopy + Send + Sync,
        CpuF: Fn((&[R], &[S]), &mut CWS) -> Result<()> + Send + Sync,
        GpuF: Fn((LaunchableSlice<'_, R>, LaunchableSlice<'_, S>), &mut GWS, &Stream) -> Result<()>
            + Send
            + Sync,
    {
        let cpu_morsel_len =
            self.executor.morsel_spec.cpu_morsel_bytes / (size_of::<R>() + size_of::<S>());
        let gpu_morsel_len =
            self.executor.morsel_spec.gpu_morsel_bytes / (size_of::<R>() + size_of::<S>());
        let dispatcher = MorselDispatcher::new(self.data.0.len(), cpu_morsel_len, gpu_morsel_len);
        let dispatcher_ref = &dispatcher;

        let cpu_thread_pool = &mut self.executor.cpu_thread_pool;
        let gpu_thread_pool = &mut self.executor.gpu_thread_pool;

        let cpu_worker_states = &mut self.cpu_worker_states;
        let gpu_worker_states = &mut self.gpu_worker_states;
        let streams = &mut self.executor.streams;

        let ro_data = (self.data.0.as_ref(), self.data.1.as_ref());

        cpu_thread_pool.scope(move |cpu_scope| {
            gpu_thread_pool.scope(move |gpu_scope| {
                let cpu_af = Arc::new(cpu_f);
                let gpu_af = Arc::new(gpu_f);

                for state in cpu_worker_states.iter_mut() {
                    let af = cpu_af.clone();

                    cpu_scope.spawn(move |_| {
                        Self::cpu_worker(dispatcher_ref, ro_data, af, state)
                            .expect("Failed to run CPU worker");
                    });
                }

                for (state, stream) in gpu_worker_states.iter_mut().zip(streams.iter_mut()) {
                    let af = gpu_af.clone();

                    gpu_scope.spawn(move |_| {
                        Self::gpu_worker(dispatcher_ref, ro_data, af, state, stream)
                            .expect("Failed to run GPU worker");
                    });
                }
            });
        });

        Ok(())
    }

    fn cpu_worker<F>(
        dispatcher: &MorselDispatcher,
        data: (&[R], &[S]),
        f: Arc<F>,
        state: &mut CWS,
    ) -> Result<()>
    where
        F: Fn((&[R], &[S]), &mut CWS) -> Result<()>,
    {
        for morsel in dispatcher.cpu_iter() {
            f((&data.0[morsel.clone()], &data.1[morsel.clone()]), state)?;
        }

        Ok(())
    }

    fn gpu_worker<F>(
        dispatcher: &MorselDispatcher,
        data: (&[R], &[S]),
        f: Arc<F>,
        state: &mut GWS,
        stream: &Stream,
    ) -> Result<()>
    where
        F: Fn((LaunchableSlice<'_, R>, LaunchableSlice<'_, S>), &mut GWS, &Stream) -> Result<()>,
    {
        for morsel in dispatcher.gpu_iter() {
            f(
                (
                    data.0[morsel.clone()].as_launchable_slice(),
                    data.1[morsel.clone()].as_launchable_slice(),
                ),
                state,
                stream,
            )?
        }

        // synchronize to not queue up _all_ morsels on the stream
        stream.synchronize()?;

        Ok(())
    }
}

impl<'a, R, S, CWS: Send, GWS: Send> Drop for StatefulHetMorselIterator2<'a, R, S, CWS, GWS> {
    fn drop(&mut self) {
        let executor = &mut self.executor;

        let cpu_worker_states = &mut self.cpu_worker_states;
        let gpu_worker_states = &mut self.gpu_worker_states;
        let streams = &mut executor.streams;

        executor.cpu_thread_pool.scope(move |cpu_scope| {
            for state in cpu_worker_states.iter_mut() {
                cpu_scope.spawn(move |_| {
                    mem::drop(state);
                });
            }
        });

        executor.gpu_thread_pool.scope(move |gpu_scope| {
            for (state, stream) in gpu_worker_states.iter_mut().zip(streams.iter_mut()) {
                gpu_scope.spawn(move |_| {
                    mem::drop(state);
                    mem::drop(stream);
                });
            }
        });
    }
}
