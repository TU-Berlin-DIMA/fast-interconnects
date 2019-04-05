/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use crossbeam_utils::thread::scope;

use rustacuda::context::CurrentContext;
use rustacuda::error::CudaResult;
use rustacuda::memory::{
    AsyncCopyDestination, DeviceBuffer, DeviceCopy, LockedBuffer, UnifiedBuffer,
};
use rustacuda::stream::{Stream, StreamFlags};

use std::cmp::min;
use std::ops::Range;

use crate::error::{ErrorKind, Result, ResultExt};
use crate::runtime::cuda_wrapper::{host_register, host_unregister, prefetch_async};
use crate::runtime::memory::{LaunchableMem, LaunchableSlice};

#[derive(Clone, Copy, Debug)]
pub enum CudaTransferStrategy {
    PageableCopy,
    PinnedCopy,
    LazyPinnedCopy,
    Coherence,
}

pub trait IntoCudaIterator<'a> {
    type Iter;

    fn into_cuda_iter(&'a mut self, chunk_len: usize) -> Result<Self::Iter>;
}

pub trait IntoCudaIteratorWithStrategy<'a> {
    type Iter;

    fn into_cuda_iter_with_strategy(
        &'a mut self,
        strategy: CudaTransferStrategy,
        chunk_len: usize,
    ) -> Result<Self::Iter>;
}

impl<'i, 'r, 's, R, S> IntoCudaIteratorWithStrategy<'i> for (&'r mut [R], &'s mut [S])
where
    'r: 'i,
    's: 'i,
    R: Copy + DeviceCopy + Send + 'i,
    S: Copy + DeviceCopy + Send + 'i,
{
    type Iter = CudaIterator2<'i, R, S>;

    fn into_cuda_iter_with_strategy(
        &'i mut self,
        strategy: CudaTransferStrategy,
        chunk_len: usize,
    ) -> Result<CudaIterator2<'i, R, S>> {
        assert_eq!(self.0.len(), self.1.len());

        let num_partitions = 4;
        let data_len = self.0.len();
        let part_len = (data_len + num_partitions - 1) / num_partitions;

        let fst = &mut self.0;
        let snd = &mut self.1;
        let partitions = fst
            .chunks_mut(part_len)
            .zip(snd.chunks_mut(part_len))
            .collect::<Vec<_>>();

        let strategy_impls = (0..num_partitions)
            .map(|_| {
                Ok((
                    new_strategy_impl::<R>(strategy, chunk_len)?,
                    new_strategy_impl::<S>(strategy, chunk_len)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(CudaIterator2::<'i> {
            chunk_len,
            partitions,
            strategy_impls,
        })
    }
}

impl<'i, 'r, 's, R, S> IntoCudaIterator<'i> for (&'r mut UnifiedBuffer<R>, &'s mut UnifiedBuffer<S>)
where
    'r: 'i,
    's: 'i,
    R: Copy + DeviceCopy,
    S: Copy + DeviceCopy,
{
    type Iter = CudaUnifiedIterator2<'i, R, S>;

    fn into_cuda_iter(&'i mut self, chunk_len: usize) -> Result<CudaUnifiedIterator2<'i, R, S>> {
        assert_eq!(self.0.len(), self.1.len());

        let streams = std::iter::repeat_with(|| Stream::new(StreamFlags::NON_BLOCKING, None))
            .take(2)
            .collect::<CudaResult<Vec<Stream>>>()?;

        Ok(CudaUnifiedIterator2::<'i> {
            data: (self.0, self.1),
            chunk_len,
            streams,
        })
    }
}

fn new_strategy_impl<'a, T: Copy + DeviceCopy + Send + 'a>(
    strategy: CudaTransferStrategy,
    chunk_len: usize,
) -> Result<Box<CudaTransferStrategyImpl<Item = T> + 'a>> {
    let wrapper: Box<CudaTransferStrategyImpl<Item = T>> = match strategy {
        CudaTransferStrategy::PageableCopy => Box::new(CudaPageableCopyStrategy::new(chunk_len)?),
        CudaTransferStrategy::PinnedCopy => Box::new(CudaPinnedCopyStrategy::new(chunk_len)?),
        CudaTransferStrategy::LazyPinnedCopy => {
            Box::new(CudaLazyPinnedCopyStrategy::new(chunk_len)?)
        }
        CudaTransferStrategy::Coherence => Box::new(CudaCoherenceStrategy::new()),
    };

    Ok(wrapper)
}

trait CudaTransferStrategyImpl: Send {
    type Item: DeviceCopy;

    fn warm_up(&mut self, _chunk: &[Self::Item]) -> Result<()> {
        Ok(())
    }

    fn copy_to_device<'s: 'l, 'c: 'l, 'l>(
        &'s mut self,
        chunk: &'c [Self::Item],
        stream: &Stream,
    ) -> Result<LaunchableSlice<'l, Self::Item>>;

    fn cool_down(&mut self, _chunk: &[Self::Item]) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
struct CudaPageableCopyStrategy<T: DeviceCopy> {
    buffer: DeviceBuffer<T>,
}

impl<T: DeviceCopy> CudaPageableCopyStrategy<T> {
    fn new(chunk_len: usize) -> Result<Self> {
        let buffer = unsafe { DeviceBuffer::<T>::zeroed(chunk_len)? };

        Ok(Self { buffer })
    }
}

impl<T: DeviceCopy> CudaTransferStrategyImpl for CudaPageableCopyStrategy<T> {
    type Item = T;

    fn copy_to_device<'s: 'l, 'c: 'l, 'l>(
        &'s mut self,
        chunk: &'c [T],
        stream: &Stream,
    ) -> Result<LaunchableSlice<'l, T>> {
        let buffer_slice = &mut self.buffer[0..chunk.len()];
        unsafe {
            buffer_slice.async_copy_from(chunk, stream)?;
        }

        Ok(buffer_slice.as_launchable_slice())
    }
}

#[derive(Debug)]
struct CudaPinnedCopyStrategy<T: Copy + DeviceCopy> {
    devc_buffer: DeviceBuffer<T>,
    host_buffer: LockedBuffer<T>,
}

impl<T: Copy + DeviceCopy> CudaPinnedCopyStrategy<T> {
    fn new(chunk_len: usize) -> Result<Self> {
        let devc_buffer = unsafe { DeviceBuffer::<T>::zeroed(chunk_len)? };

        let host_buffer = unsafe { LockedBuffer::<T>::uninitialized(chunk_len)? };

        Ok(Self {
            devc_buffer,
            host_buffer,
        })
    }
}

impl<T: Copy + DeviceCopy> CudaTransferStrategyImpl for CudaPinnedCopyStrategy<T> {
    type Item = T;

    fn warm_up(&mut self, chunk: &[T]) -> Result<()> {
        self.host_buffer[0..chunk.len()].copy_from_slice(chunk);
        Ok(())
    }

    fn copy_to_device<'s: 'l, 'c: 'l, 'l>(
        &'s mut self,
        chunk: &'c [T],
        stream: &Stream,
    ) -> Result<LaunchableSlice<'l, T>> {
        let d_slice = &mut self.devc_buffer[0..chunk.len()];
        unsafe {
            d_slice.async_copy_from(&self.host_buffer[0..chunk.len()], stream)?;
        }

        Ok(d_slice.as_launchable_slice())
    }
}

#[derive(Debug)]
struct CudaLazyPinnedCopyStrategy<T: DeviceCopy> {
    buffer: DeviceBuffer<T>,
}

impl<T: DeviceCopy> CudaLazyPinnedCopyStrategy<T> {
    fn new(chunk_len: usize) -> Result<Self> {
        let buffer = unsafe { DeviceBuffer::<T>::zeroed(chunk_len)? };

        Ok(Self { buffer })
    }
}

impl<T: DeviceCopy> CudaTransferStrategyImpl for CudaLazyPinnedCopyStrategy<T> {
    type Item = T;

    fn warm_up(&mut self, chunk: &[T]) -> Result<()> {
        unsafe {
            host_register(chunk).chain_err(|| {
                ErrorKind::RuntimeError("Failed to page-lock NUMA memory region".to_string())
            })?;
        };
        Ok(())
    }

    fn copy_to_device<'s: 'l, 'c: 'l, 'l>(
        &'s mut self,
        chunk: &'c [T],
        stream: &Stream,
    ) -> Result<LaunchableSlice<'l, T>> {
        let buffer_slice = &mut self.buffer[0..chunk.len()];
        unsafe {
            buffer_slice.async_copy_from(chunk, stream)?;
        }

        Ok(buffer_slice.as_launchable_slice())
    }

    fn cool_down(&mut self, chunk: &[T]) -> Result<()> {
        unsafe {
            host_unregister(chunk)?;
        };
        Ok(())
    }
}

#[derive(Debug)]
struct CudaCoherenceStrategy<T> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: DeviceCopy> CudaCoherenceStrategy<T> {
    fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DeviceCopy + Send> CudaTransferStrategyImpl for CudaCoherenceStrategy<T> {
    type Item = T;

    fn copy_to_device<'s: 'l, 'c: 'l, 'l>(
        &'s mut self,
        chunk: &'c [T],
        _stream: &Stream,
    ) -> Result<LaunchableSlice<'l, T>> {
        Ok(chunk.as_launchable_slice())
    }
}

pub struct CudaIterator2<'a, R: Copy + DeviceCopy, S: Copy + DeviceCopy> {
    chunk_len: usize,
    partitions: Vec<(&'a mut [R], &'a mut [S])>,
    strategy_impls: Vec<(
        Box<CudaTransferStrategyImpl<Item = R> + 'a>,
        Box<CudaTransferStrategyImpl<Item = S> + 'a>,
    )>,
}

impl<'a, R: Copy + DeviceCopy + Send, S: Copy + DeviceCopy + Send> CudaIterator2<'a, R, S> {
    pub fn fold<F>(&mut self, f: F) -> Result<()>
    where
        F: Fn((LaunchableSlice<R>, LaunchableSlice<S>), &Range<usize>, &Stream) -> Result<()>
            + Send
            + Sync,
    {
        let partitions = &mut self.partitions;
        let strategy_impls = &mut self.strategy_impls;
        let chunk_len = self.chunk_len;
        let af = std::sync::Arc::new(f);

        scope(|scope| {
            partitions
                .iter_mut()
                .zip(strategy_impls.iter_mut())
                .map(
                    |((partition_fst, partition_snd), (strategy_fst, strategy_snd))| {
                        let pf = af.clone();
                        let unowned_context = CurrentContext::get_current().unwrap();
                        scope.spawn(move |_| {
                            CurrentContext::set_current(&unowned_context).unwrap();
                            let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
                            partition_fst
                                .chunks_mut(chunk_len)
                                .zip(partition_snd.chunks_mut(chunk_len))
                                .map(|(fst, snd)| {
                                    strategy_fst.warm_up(&fst).unwrap();
                                    strategy_snd.warm_up(&snd).unwrap();

                                    let fst_chunk =
                                        strategy_fst.copy_to_device(&fst, &stream).unwrap();
                                    let snd_chunk =
                                        strategy_snd.copy_to_device(&snd, &stream).unwrap();

                                    let range = 0..fst.len();
                                    let result = pf((fst_chunk, snd_chunk), &range, &stream);

                                    strategy_fst.cool_down(&fst).unwrap();
                                    strategy_snd.cool_down(&snd).unwrap();

                                    result.unwrap();
                                    ()
                                })
                                .for_each(drop);
                            stream.synchronize().unwrap();
                        });
                    },
                )
                .for_each(drop);
        })
        .unwrap();

        Ok(())
    }
}

#[derive(Debug)]
pub struct CudaUnifiedIterator2<'a, R: Copy + DeviceCopy, S: Copy + DeviceCopy> {
    data: (&'a mut UnifiedBuffer<R>, &'a mut UnifiedBuffer<S>),
    chunk_len: usize,
    streams: Vec<Stream>,
}

impl<'a, R: Copy + DeviceCopy, S: Copy + DeviceCopy> CudaUnifiedIterator2<'a, R, S> {
    pub fn fold<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut((LaunchableSlice<R>, LaunchableSlice<S>), &Range<usize>, &Stream) -> Result<()>,
    {
        let data_len = self.data.0.len();
        let chunk_len = self.chunk_len;
        let fst = &mut self.data.0;
        let snd = &mut self.data.1;
        let streams = &self.streams;

        (0..data_len)
            .step_by(chunk_len)
            .map(|start| {
                let end = min(start + chunk_len, data_len);
                start..end
            })
            .zip(streams.iter().cycle())
            .map(|(range, stream)| {
                let fst_ptr = unsafe { fst.as_unified_ptr().add(range.start) };
                let snd_ptr = unsafe { snd.as_unified_ptr().add(range.start) };

                prefetch_async(fst_ptr, range.len(), stream)?;
                prefetch_async(snd_ptr, range.len(), stream)?;

                let fst_chunk = fst.as_slice()[range.clone()].as_launchable_slice();
                let snd_chunk = snd.as_slice()[range.clone()].as_launchable_slice();

                f((fst_chunk, snd_chunk), &range, stream)?;

                Ok(())
            })
            .collect::<Result<_>>()?;

        streams
            .iter()
            .map(|stream| stream.synchronize())
            .collect::<CudaResult<()>>()?;

        Ok(())
    }
}
