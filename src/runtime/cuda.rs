/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

//! CUDA runtime for data transfer and kernel execution.
//!
//! There exist multiple methods to transfer data from main-memory to device
//! memory. Also, data transfer and execution should overlap for the best
//! performance. This module provides a collection of transfer method
//! implementations, and efficient iterators for executing GPU kernels.

use crossbeam_utils::thread::{scope, ScopedJoinHandle};

use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPoolBuilder;

use rustacuda::context::CurrentContext;
use rustacuda::error::CudaResult;
use rustacuda::event::{Event, EventFlags};
use rustacuda::memory::{
    AsyncCopyDestination, DeviceBuffer, DeviceCopy, LockedBuffer, UnifiedBuffer,
};
use rustacuda::stream::{Stream, StreamFlags};

use std::cmp::min;
use std::collections::LinkedList;
use std::default::Default;
use std::thread::Result as ThreadResult;
use std::time::Instant;

use crate::error::{ErrorKind, Result, ResultExt};
use crate::runtime::cuda_wrapper::{
    current_device_id, host_register, host_unregister, prefetch_async,
};
use crate::runtime::memory::{LaunchableMem, LaunchableSlice};

/// Timer based on CUDA events.
///
/// Times the duration of operations scheduled on a CUDA stream.
///
/// # Example:
///
/// ```
/// # use numa_gpu::runtime::cuda::EventTimer;
///
/// # use rustacuda::quick_init;
/// # use rustacuda::stream::{Stream, StreamFlags};
/// #
/// # let _ctx = quick_init().unwrap();
/// # let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
/// let timer = EventTimer::record_start(&stream).unwrap();
/// // ... schedule some work on the queue ...
/// timer.record_stop(&stream).unwrap();
/// let time_in_ms = timer.synchronize_and_time().unwrap();
/// ```
pub struct EventTimer {
    start: Event,
    end: Event,
}

impl EventTimer {
    /// Starts recording time.
    pub fn record_start(stream: &Stream) -> CudaResult<Self> {
        let start = Event::new(EventFlags::DEFAULT)?;
        let end = Event::new(EventFlags::DEFAULT)?;

        start.record(&stream)?;

        Ok(Self { start, end })
    }

    /// Stops recording time.
    pub fn record_stop(&self, stream: &Stream) -> CudaResult<()> {
        self.end.record(stream)?;
        Ok(())
    }

    /// Waits for the timer to finish and returns the duration in milliseconds.
    pub fn synchronize_and_time(&self) -> CudaResult<f32> {
        self.end.synchronize()?;
        self.end.elapsed_time_f32(&self.start)
    }
}

/// Specify the CUDA transfer strategy.
///
/// Defines which strategy with which to transfer data from main-memory to
/// device memory.
///
/// A `Prefetch` strategy is not defined, because `cuda_prefetch_async()`
/// requires unified memory allocated with `cuda_malloc_managed()`. Thus,
/// regular memory that is allocated with the system allocator cannot be used in
/// conjunction with prefetching. As unified memory is more specific than the
/// general system memory, prefetching is handled separately using
/// `IntoCudaIterator`.
#[derive(Clone, Copy, Debug)]
pub enum CudaTransferStrategy {
    /// Copy directly from pageable memory.
    ///
    /// Transfers data using `cuda_memcopy()` directly from the specified memory
    /// location. No intermediate steps are performed.
    ///
    /// This strategy can also transfer memory that is pinned by the user before
    /// the transfer.
    PageableCopy,

    /// Copy using an intermediate, pinned buffer.
    ///
    /// The transfer first copies data from its original location into a pinned
    /// buffer. Then data is transferred using `cuda_memcpy()`.
    ///
    /// In principle, this strategy can also tranfer from a pinned memory
    /// location. However, the `PageableCopy` strategy would avoid the additional
    /// overhead incurred by the intermediate buffer.
    PinnedCopy,

    /// Pin the memory in-place and the copy.
    ///
    /// The transfer first pins the memory in-place using `cuda_host_register()`.
    /// Then the data is transferred using `cuda_memcpy()`.
    ///
    /// Don't use this strategy with a pinned memory location, as calling
    /// `cuda_host_register()` on a pinned location is probably undefined
    /// behavior.
    LazyPinnedCopy,

    /// Access the memory in-place without any copies.
    ///
    /// Requires that the GPU has cache-coherent access to main-memory.
    /// E.g., POWER9 and Tesla V100 with NVLink 2.0.
    Coherence,
}

/// Conversion into a CUDA iterator.
///
/// By implementing `IntoCudaIterator` for a type, you define how the type is
/// converted into an iterator capable of executing CUDA functions on a GPU.
///
/// The iterator must define a transfer strategy.
///
/// The `chunk_len` parameter specifies the granularity of each data transfer
/// from main-memory to device memory. The same granularity is used when
/// passing input parameters to the GPU kernel.
pub trait IntoCudaIterator<'a> {
    /// The type of the iterator to produce.
    type Iter;

    /// Creates an iterator from a value.
    ///
    /// See the module-level documentation for details.
    fn into_cuda_iter(&'a mut self, chunk_len: usize) -> Result<Self::Iter>;
}

/// Conversion into a CUDA iterator with a specified transfer strategy.
///
/// By implementing `IntoCudaIteratorWithStrategy` for a type, you define how
/// the type is converted into an iterator capable of executing CUDA functions
/// on a GPU.
///
/// `strategy` specifies which transfer strategy to use. See the
/// `CudaTransferStrategy` documation for details. The iterator must implement
/// all strategies.
///
/// The `chunk_len` parameter specifies the granularity of each data transfer
/// from main-memory to device memory. The same granularity is used when
/// passing input parameters to the GPU kernel.
pub trait IntoCudaIteratorWithStrategy<'a> {
    /// The type of the iterator to produce.
    type Iter;

    /// Creates an iterator from a value.
    ///
    /// See the module-level documentation for details.
    fn into_cuda_iter_with_strategy(
        &'a mut self,
        strategy: CudaTransferStrategy,
        chunk_len: usize,
    ) -> Result<Self::Iter>;
}

/// Timings of the `CudaTransferStrategy` phases
#[derive(Debug, Default)]
pub struct CudaTransferStrategyMeasurement {
    /// Warm up phase in nanoseconds
    pub warm_up_ns: Option<f64>,

    /// Copy phase in nanoseconds
    pub copy_ns: Option<f64>,

    /// Compute phase (i.e., kernel execution) in nanoseconds
    pub compute_ns: Option<f64>,

    /// Cool down phase in nanoseconds
    pub cool_down_ns: Option<f64>,
}

/// Converts a tuple of two mutable slices into a CUDA iterator.
///
/// The slices must be mutable (and cannot be read-only) because transfer
/// strategies can be implemented using a parallel pipeline. Parallelism
/// requires exclusive access to data for Rust to successfully type-check.
/// In Rust, holding a mutable reference guarantees exclusive access, because
/// only a single mutable reference can exist at any point in time.
///
/// The slices can have mutually distinct lifetimes. However, the lifetime
/// of the resulting iterator must be shorter than that of the shortest slice
/// lifetime.
///
/// This implementation should be used as a basis for future implementations
/// for other tuples of slices (e.g., one slice, three slices, etc.).
impl<'i, 'r, 's, R, S> IntoCudaIteratorWithStrategy<'i> for (&'r mut [R], &'s mut [S])
where
    'r: 'i,
    's: 'i,
    R: Copy + DeviceCopy + Send + Sync + 'i,
    S: Copy + DeviceCopy + Send + Sync + 'i,
{
    type Iter = CudaIterator2<'i, R, S>;

    fn into_cuda_iter_with_strategy(
        &'i mut self,
        strategy: CudaTransferStrategy,
        chunk_len: usize,
    ) -> Result<CudaIterator2<'i, R, S>> {
        assert_eq!(self.0.len(), self.1.len());

        let memcpy_threads = 8;
        let num_partitions = 4;

        let streams = std::iter::repeat_with(|| Stream::new(StreamFlags::NON_BLOCKING, None))
            .take(3)
            .collect::<CudaResult<Vec<Stream>>>()?;

        let strategy_impls = (0..num_partitions)
            .map(|_| {
                Ok((
                    new_strategy_impl::<R>(strategy, chunk_len)?,
                    new_strategy_impl::<S>(strategy, chunk_len)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        // Error occurs after first call, because pool is already built
        let _ignore_error = ThreadPoolBuilder::new()
            .num_threads(memcpy_threads)
            .build_global();

        Ok(CudaIterator2::<'i> {
            data: (&mut self.0, &mut self.1),
            chunk_len,
            streams,
            strategy,
            strategy_impls,
        })
    }
}

/// Converts a tuple of two mutable unified buffer references into a CUDA
/// iterator.
///
/// The references must be mutable because the buffer should not be modified
/// during the prefetching operation (as required by
/// UnifiedBuffer::as_unified_ptr).
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

/// Instantiates the concrete transfer strategy implementation as specified by
/// the strategy enum.
fn new_strategy_impl<'a, T: Copy + DeviceCopy + Send + Sync + 'a>(
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

/// Implements a CUDA transfer strategy.
///
/// By implementing `CudaTransferStrategyImpl` for a type, you define a concrete
/// strategy to transfer data from main-memory to device memory.
///
/// # Contract
///
/// A caller of the methods specified by `CudaTransferStrategyImpl` must adhere
/// to the following contract.
///
/// The methods `warm_up()`, `copy_to_device`, and `cool_down` must be called
/// in this order. In addition, they must be called on the same chunk, exactly
/// once, in sequence. Chunks may not be interleaved.
///
/// A common use-case for interleaving chunks is transferring multiple input
/// parameters to the device. Instead of iterleaving chunks originating from
/// different inputs, each input should be seen as a separate stream of chunks.
/// Thus, each chunk stream should have its own `CudaTransferStrategyImpl`
/// instance.
///
/// # Parallelism
///
/// All functions shall execute asynchronously. To wait for it to complete, call
/// `Stream::synchronize()` or other CUDA stream functions.
trait CudaTransferStrategyImpl: Send {
    /// The type of elements being iterator over.
    type Item: DeviceCopy;

    /// Prepare the chunk for copying.
    ///
    /// For example, copy to an itermediate buffer.
    fn warm_up(&mut self, _chunk: &[Self::Item], _stream: &Stream) -> Result<()> {
        Ok(())
    }

    /// Transfer the chunk from main-memory to device memory.
    ///
    /// For example, by using `cuda_memcpy()`.
    ///
    /// # Reference lifetimes
    ///
    /// The resulting `LaunchableSlice` must have a shorter lifetime than the data
    /// it references. However, we want to be able to either reference data that
    /// belongs to `self` (e.g., an intermediate pinned buffer), or to reference
    /// the chunk directly. Thus, `self` and the chunk must both outlive
    /// `LaunchableSlice`.
    ///
    /// See [the Rust book's chapter on advanced lifetimes](https://doc.rust-lang.org/book/ch19-02-advanced-lifetimes.html)
    /// for syntax details.
    fn copy_to_device<'s: 'l, 'c: 'l, 'l>(
        &'s mut self,
        chunk: &'c [Self::Item],
        stream: &Stream,
    ) -> Result<LaunchableSlice<'l, Self::Item>>;

    /// Tear down any resources created in `warm_up`.
    ///
    /// For example, unpin a lazily pinned buffer.
    fn cool_down(&mut self, _chunk: &[Self::Item], _stream: &Stream) -> Result<()> {
        Ok(())
    }
}

/// CUDA pageable copy strategy.
///
/// See the `CudaTransferStrategy` documentation for details.
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

/// CUDA pinned copy strategy.
///
/// See the `CudaTransferStrategy` documentation for details.
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

impl<T: Copy + DeviceCopy + Send + Sync> CudaTransferStrategyImpl for CudaPinnedCopyStrategy<T> {
    type Item = T;

    fn warm_up(&mut self, chunk: &[T], stream: &Stream) -> Result<()> {
        stream.synchronize()?;

        let memcpy_threads = rayon::current_num_threads();
        let par_chunk_len = (chunk.len() + memcpy_threads - 1) / memcpy_threads;

        self.host_buffer[0..chunk.len()]
            .par_chunks_mut(par_chunk_len)
            .zip(chunk.par_chunks(par_chunk_len))
            .for_each(|(dst, src)| {
                dst.copy_from_slice(src);
            });

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

/// CUDA lazy pinned copy strategy.
///
/// See the `CudaTransferStrategy` documentation for details.
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

    fn warm_up(&mut self, chunk: &[T], stream: &Stream) -> Result<()> {
        stream.synchronize()?;
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

    fn cool_down(&mut self, chunk: &[T], stream: &Stream) -> Result<()> {
        stream.synchronize()?;
        unsafe {
            host_unregister(chunk)?;
        };
        Ok(())
    }
}

/// CUDA coherence strategy.
///
/// See the `CudaTransferStrategy` documentation for details.
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

/// CUDA iterator for two mutable inputs.
///
/// Transfers data from main-memory to device memory on a chunk-sized
/// granularity.
///
/// # Preconditions
///
/// All inputs are required to have the same length.
///
/// # Thread safety
///
/// Transfers involve multiple stages (currently 4 stages). These
/// stages are performed in a parallel pipeline using threads. For this reason,
/// thread-safety is implemented through mutable references and the `Send`
/// marker trait.
///
/// See the `fold()` documentation for details.
///
/// # Notes for future reference
///
/// Currently, `CudaIterator2` does not copy back data from device memory to
/// main-memory, but this functionality could be implemented in future.
///
/// `CudaIterator2` could be used as a template for iterators over less or more
/// inputs, e.g. a `CudaIterator1` or `CudaIterator3`.
pub struct CudaIterator2<'a, R: Copy + DeviceCopy, S: Copy + DeviceCopy> {
    data: (&'a mut [R], &'a mut [S]),
    chunk_len: usize,
    streams: Vec<Stream>,
    // partitions: Vec<(&'a mut [R], &'a mut [S])>,
    strategy: CudaTransferStrategy,
    strategy_impls: Vec<(
        Box<CudaTransferStrategyImpl<Item = R> + 'a>,
        Box<CudaTransferStrategyImpl<Item = S> + 'a>,
    )>,
}

impl<'a, R: Copy + DeviceCopy + Send, S: Copy + DeviceCopy + Send> CudaIterator2<'a, R, S> {
    /// Apply a GPU function that produces a single, final value.
    ///
    /// `fold()` takes two arguments: a data value, and a CUDA stream. In the
    /// case of `CudaIterator2`, the data value is specified as a two-tuple of
    /// launchable slices. The slices are guaranteed to have the same length.
    ///
    /// The function passed to `fold()` is meant to launch a CUDA kernel function
    /// on the given CUDA stream.
    ///
    /// In contrast to Rust's standard library `fold()` iterator, the state in
    /// this iterator is implicit in GPU memory.
    ///
    /// # Thread safety
    ///
    /// As the transfer is performed as a parallel pipeline, i.e., transfer and
    /// execution overlap. Therefore, the function may be called by multiple
    /// threads at the same time, and must be thread-safe. Thread-safety is
    /// specified through the `Send` marker trait.
    ///
    /// Furthermore, the CUDA kernel is executed on two or more CUDA streams,
    /// and must therefore be thread-safe. However, Rust cannot guarantee
    /// thread-safety of CUDA kernels. Thus, the user must ensure that the CUDA
    /// kernels are safe to execute on multiple CUDA streams, e.g. by using
    /// atomic operations when accessing device memory.
    ///
    /// # Internals
    ///
    /// The current implementation calls `fold_par()` if the `LazyPinnedCopy`
    /// strategy is selected. For all other strategies, `fold_async()` is called.
    /// The reason is that `LazyPinnedCopy` performs blocking calls, therefore
    /// transfer-compute-overlapping requires multi-threading. In constrast, the
    /// other strategies can be executed completely asynchronously by CUDA.
    ///
    /// # Example
    ///
    /// ```
    /// # use numa_gpu::runtime::cuda::{
    /// #     CudaTransferStrategy, IntoCudaIterator, IntoCudaIteratorWithStrategy,
    /// # };
    /// #
    /// # use rustacuda::launch;
    /// # use rustacuda::memory::{CopyDestination, DeviceBox, UnifiedBuffer};
    /// # use rustacuda::module::Module;
    /// # use rustacuda::quick_init;
    /// #
    /// # use std::ffi::CString;
    /// #
    /// # let _ctx = quick_init().unwrap();
    /// #
    /// # let module_data =
    /// #     CString::new(include_str!("../../resources/dot.ptx")).unwrap();
    /// # let module = Module::load_from_string(&module_data).unwrap();
    /// # let cuda_dot = module.get_function(&CString::new("dot").unwrap()).unwrap();
    /// #
    /// # let data_len = 2_usize.pow(20);
    /// let chunk_len = 1024_usize;
    /// # assert_eq!(data_len % chunk_len, 0);
    ///
    /// let mut data_0 = vec![1.0_f32; data_len];
    /// let mut data_1 = vec![1.0_f32; data_len];
    /// let mut result = DeviceBox::new(&0.0f32).unwrap();
    /// let result_ptr = result.as_device_ptr();
    ///
    /// (data_0.as_mut_slice(), data_1.as_mut_slice())
    ///     .into_cuda_iter_with_strategy(CudaTransferStrategy::PageableCopy, chunk_len)
    ///     .unwrap()
    ///     .fold(|(x, y), stream| {
    ///         unsafe {
    ///             launch!(cuda_dot<<<1, 1, 0, stream>>>(
    ///             x.len(),
    ///             x.as_launchable_ptr(),
    ///             y.as_launchable_ptr(),
    ///             result_ptr
    ///             ))?;
    ///         }
    ///
    ///         Ok(())
    ///     }).unwrap();
    ///
    /// let mut result_host = 0.0f32;
    /// result.copy_to(&mut result_host).unwrap();
    /// ```
    pub fn fold<F>(&mut self, f: F) -> Result<CudaTransferStrategyMeasurement>
    where
        // FIXME: should be using a mutable LaunchableSlice type
        F: Fn((LaunchableSlice<R>, LaunchableSlice<S>), &Stream) -> Result<()> + Send + Sync,
    {
        match self.strategy {
            CudaTransferStrategy::LazyPinnedCopy | CudaTransferStrategy::PageableCopy => {
                self.fold_par(f)
            }
            _ => self.fold_async(f),
        }
    }

    /// A parallel implementation of `fold()`.
    ///
    /// Transfer-compute-overlapping is performed by multi-threading the pipeline
    /// stages parallelism. As we configure at least as many threads as there
    /// are pipeline stages, each thread may execute the complete pipeline
    /// synchronously.
    pub fn fold_par<F>(&mut self, f: F) -> Result<CudaTransferStrategyMeasurement>
    where
        // FIXME: should be using a mutable LaunchableSlice type
        F: Fn((LaunchableSlice<R>, LaunchableSlice<S>), &Stream) -> Result<()> + Send + Sync,
    {
        let num_partitions = self.strategy_impls.len();
        let data_fst = &mut self.data.0;
        let data_snd = &mut self.data.1;
        let data_len = data_fst.len();
        let part_len = (data_len + num_partitions - 1) / num_partitions;

        let mut partitions = data_fst
            .chunks_mut(part_len)
            .zip(data_snd.chunks_mut(part_len))
            .collect::<Vec<_>>();

        let strategy_impls = &mut self.strategy_impls;
        let chunk_len = self.chunk_len;
        let af = std::sync::Arc::new(f);

        let summed_times: ThreadResult<Result<_>> = scope(|scope| {
            let mut thread_handles = partitions
                .iter_mut()
                .zip(strategy_impls.iter_mut())
                .map(
                    |((partition_fst, partition_snd), (strategy_fst, strategy_snd))| {
                        let pf = af.clone();
                        let unowned_context = CurrentContext::get_current()?;
                        let handle: ScopedJoinHandle<Result<_>> = scope.spawn(move |_| {
                            CurrentContext::set_current(&unowned_context)?;
                            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
                            let times = partition_fst
                                .chunks_mut(chunk_len)
                                .zip(partition_snd.chunks_mut(chunk_len))
                                .map(|(fst, snd)| {
                                    let warm_up_timer = Instant::now();
                                    strategy_fst.warm_up(&fst, &stream)?;
                                    strategy_snd.warm_up(&snd, &stream)?;
                                    let warm_up_ns = warm_up_timer.elapsed().as_nanos() as f64;

                                    let begin_copy_event = Event::new(EventFlags::DEFAULT)?;
                                    begin_copy_event.record(&stream)?;

                                    let fst_chunk = strategy_fst.copy_to_device(&fst, &stream)?;
                                    let snd_chunk = strategy_snd.copy_to_device(&snd, &stream)?;

                                    let end_copy_event = Event::new(EventFlags::DEFAULT)?;
                                    end_copy_event.record(&stream)?;

                                    let begin_comp_event = Event::new(EventFlags::DEFAULT)?;
                                    begin_comp_event.record(&stream)?;

                                    pf((fst_chunk, snd_chunk), &stream)?;

                                    let end_comp_event = Event::new(EventFlags::DEFAULT)?;
                                    end_comp_event.record(&stream)?;

                                    // Wait for the copy to complete before
                                    // cooling down resources created in the warm_up
                                    end_copy_event.synchronize()?;
                                    let copy_ns =
                                        end_copy_event.elapsed_time_f32(&begin_copy_event)? as f64
                                            * 10_f64.powf(6.0);

                                    let cool_down_timer = Instant::now();
                                    strategy_fst.cool_down(&fst, &stream)?;
                                    strategy_snd.cool_down(&snd, &stream)?;
                                    let cool_down_ns = cool_down_timer.elapsed().as_nanos() as f64;

                                    end_comp_event.synchronize()?;
                                    let comp_ns =
                                        end_comp_event.elapsed_time_f32(&begin_comp_event)? as f64
                                            * 10_f64.powf(6.0);

                                    Ok((warm_up_ns, copy_ns, comp_ns, cool_down_ns))
                                })
                                .collect::<Result<Vec<(f64, f64, f64, f64)>>>()?;
                            stream.synchronize()?;

                            Ok(times)
                        });

                        Ok(handle)
                    },
                )
                .collect::<Result<
                    Vec<
                        crossbeam_utils::thread::ScopedJoinHandle<
                            '_,
                            Result<Vec<(f64, f64, f64, f64)>>,
                        >,
                    >,
                >>()?;

            let thread_results = thread_handles
                .drain(..)
                .map(|handle| {
                    let times = handle.join().expect("Failed to join thread");
                    times
                })
                .collect::<Result<LinkedList<_>>>()?;

            let summed_times = thread_results.iter().flatten().fold(
                CudaTransferStrategyMeasurement {
                    warm_up_ns: Some(0.0),
                    copy_ns: Some(0.0),
                    compute_ns: Some(0.0),
                    cool_down_ns: Some(0.0),
                },
                |m, (warm_up, copy, comp, cool_down)| CudaTransferStrategyMeasurement {
                    warm_up_ns: m.warm_up_ns.map(|ns| ns + warm_up),
                    copy_ns: m.copy_ns.map(|ns| ns + copy),
                    compute_ns: m.compute_ns.map(|ns| ns + comp),
                    cool_down_ns: m.cool_down_ns.map(|ns| ns + cool_down),
                    ..Default::default()
                },
            );

            Ok(summed_times)
        });

        summed_times.expect("Failure inside thread scope")
    }
}

impl<'a, R: Copy + DeviceCopy, S: Copy + DeviceCopy> CudaIterator2<'a, R, S> {
    /// An asynchronous implementation of `fold()`.
    ///
    /// Transfer-compute-overlapping is achieved by calling asynchronous CUDA
    /// functions. If not all functions are asynchronous, the pipeline is
    /// executed synchronously.
    ///
    /// # Correctness
    ///
    /// If a blocking function is scheduled as part of a strategy, then that
    /// function must enforce synchronous execution, e.g. by calling
    /// `stream.synchronize()`.
    pub fn fold_async<F>(&mut self, mut f: F) -> Result<CudaTransferStrategyMeasurement>
    where
        F: FnMut((LaunchableSlice<R>, LaunchableSlice<S>), &Stream) -> Result<()>,
    {
        let chunk_len = self.chunk_len;
        let fst = &mut self.data.0;
        let snd = &mut self.data.1;
        let streams = &self.streams;
        let (strategy_fst, strategy_snd) = &mut self.strategy_impls[0];

        let timers = fst
            .chunks_mut(chunk_len)
            .zip(snd.chunks_mut(chunk_len))
            .zip(streams.iter().cycle())
            .map(|((fst, snd), stream)| {
                // Warm-up
                let warm_up_timer = EventTimer::record_start(&stream)?;
                strategy_fst.warm_up(&fst, &stream)?;
                strategy_snd.warm_up(&snd, &stream)?;
                warm_up_timer.record_stop(&stream)?;

                // Copy to device
                let copy_timer = EventTimer::record_start(&stream)?;
                let fst_chunk = strategy_fst.copy_to_device(&fst, &stream)?;
                let snd_chunk = strategy_snd.copy_to_device(&snd, &stream)?;
                copy_timer.record_stop(&stream)?;

                // Launch kernel
                let comp_timer = EventTimer::record_start(&stream)?;
                f((fst_chunk, snd_chunk), stream)?;
                comp_timer.record_stop(&stream)?;

                // Cool down
                let cool_down_timer = EventTimer::record_start(&stream)?;
                strategy_fst.cool_down(&fst, &stream)?;
                strategy_snd.cool_down(&snd, &stream)?;
                cool_down_timer.record_stop(&stream)?;

                // Collect timers
                Ok((warm_up_timer, copy_timer, comp_timer, cool_down_timer))
            })
            .collect::<Result<Vec<_>>>()?;

        let summed_times = timers
            .iter()
            .map(|(warm_up, copy, comp, cool_down)| {
                Ok((
                    warm_up.synchronize_and_time()? as f64 * 10_f64.powf(6.0),
                    copy.synchronize_and_time()? as f64 * 10_f64.powf(6.0),
                    comp.synchronize_and_time()? as f64 * 10_f64.powf(6.0),
                    cool_down.synchronize_and_time()? as f64 * 10_f64.powf(6.0),
                ))
            })
            .fold(
                Ok(CudaTransferStrategyMeasurement {
                    warm_up_ns: Some(0.0),
                    copy_ns: Some(0.0),
                    compute_ns: Some(0.0),
                    cool_down_ns: Some(0.0),
                }),
                |accum: Result<_>, times: Result<_>| {
                    accum.and_then(|m| {
                        times.and_then(|(warm_up, copy, comp, cool_down)| {
                            Ok(CudaTransferStrategyMeasurement {
                                warm_up_ns: m.warm_up_ns.map(|ns| ns + warm_up),
                                copy_ns: m.copy_ns.map(|ns| ns + copy),
                                compute_ns: m.compute_ns.map(|ns| ns + comp),
                                cool_down_ns: m.cool_down_ns.map(|ns| ns + cool_down),
                            })
                        })
                    })
                },
            )?;

        streams
            .iter()
            .map(|stream| stream.synchronize())
            .collect::<CudaResult<()>>()?;

        Ok(summed_times)
    }
}

/// CUDA iterator for two mutable unified memory inputs.
///
/// Prefetches data from main-memory to device memory on a chunk-sized
/// granularity.
///
/// # Preconditions
///
/// All inputs are required to have the same length.
///
/// # Thread safety
///
/// Only one CPU thread is used within the iterator, thus thread-safety only
/// applies to the CUDA kernel. See the `fold()` documentation for details.
#[derive(Debug)]
pub struct CudaUnifiedIterator2<'a, R: Copy + DeviceCopy, S: Copy + DeviceCopy> {
    data: (&'a mut UnifiedBuffer<R>, &'a mut UnifiedBuffer<S>),
    chunk_len: usize,
    streams: Vec<Stream>,
}

impl<'a, R: Copy + DeviceCopy, S: Copy + DeviceCopy> CudaUnifiedIterator2<'a, R, S> {
    /// Apply a GPU function that produces a single, final value.
    ///
    /// `fold()` takes two arguments: a data value, and a CUDA stream. In the
    /// case of `CudaUnifiedIterator2`, the data value is specified as a
    /// two-tuple of launchable slices. The slices are guaranteed to have the
    /// same length.
    ///
    /// The function passed to `fold()` is meant to launch a CUDA kernel function
    /// on the given CUDA stream.
    ///
    /// In contrast to Rust's standard library `fold()` iterator, the state in
    /// this iterator is implicit in GPU memory.
    ///
    /// # Thread safety
    ///
    /// Prefetching and kernel execution are asynchronous operations. They are
    /// performed on two or more CUDA streams to achieve parallelism, i.e..
    /// prefetching and execution overlap.
    ///
    /// However, Rust cannot guarantee thread-safety of CUDA kernels. Thus, the
    /// user must ensure that the CUDA kernels are safe to execute on multiple
    /// CUDA streams, e.g. by using atomic operations when accessing device
    /// memory.
    pub fn fold<F>(&mut self, mut f: F) -> Result<CudaTransferStrategyMeasurement>
    where
        F: FnMut((LaunchableSlice<R>, LaunchableSlice<S>), &Stream) -> Result<()>,
    {
        let data_len = self.data.0.len();
        let chunk_len = self.chunk_len;
        let fst = &mut self.data.0;
        let snd = &mut self.data.1;
        let streams = &self.streams;
        let device_id = current_device_id()?;

        let timers = (0..data_len)
            .step_by(chunk_len)
            .map(|start| {
                let end = min(start + chunk_len, data_len);
                start..end
            })
            .zip(streams.iter().cycle())
            .map(|(range, stream)| {
                let fst_ptr = unsafe { fst.as_unified_ptr().add(range.start) };
                let snd_ptr = unsafe { snd.as_unified_ptr().add(range.start) };

                // Prefetch chunk to device
                let prefetch_timer = EventTimer::record_start(&stream)?;
                prefetch_async(fst_ptr, range.len(), device_id, stream)?;
                prefetch_async(snd_ptr, range.len(), device_id, stream)?;
                prefetch_timer.record_stop(&stream)?;

                let fst_chunk = fst.as_slice()[range.clone()].as_launchable_slice();
                let snd_chunk = snd.as_slice()[range.clone()].as_launchable_slice();

                // Launch kernel
                let comp_timer = EventTimer::record_start(&stream)?;
                f((fst_chunk, snd_chunk), stream)?;
                comp_timer.record_stop(&stream)?;

                Ok((prefetch_timer, comp_timer))
            })
            .collect::<Result<LinkedList<_>>>()?;

        let summed_times = timers
            .iter()
            .map(|(prefetch, comp)| {
                Ok((
                    prefetch.synchronize_and_time()? as f64 * 10_f64.powf(6.0),
                    comp.synchronize_and_time()? as f64 * 10_f64.powf(6.0),
                ))
            })
            .fold(
                Ok(CudaTransferStrategyMeasurement {
                    warm_up_ns: None,
                    copy_ns: Some(0.0),
                    compute_ns: Some(0.0),
                    cool_down_ns: None,
                }),
                |accum: Result<_>, times: Result<_>| {
                    accum.and_then(|m| {
                        times.and_then(|(prefetch, comp)| {
                            Ok(CudaTransferStrategyMeasurement {
                                copy_ns: m.copy_ns.map(|ns| ns + prefetch),
                                compute_ns: m.compute_ns.map(|ns| ns + comp),
                                ..Default::default()
                            })
                        })
                    })
                },
            )?;

        streams
            .iter()
            .map(|stream| stream.synchronize())
            .collect::<CudaResult<()>>()?;

        Ok(summed_times)
    }
}
