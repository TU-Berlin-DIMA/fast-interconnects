// Copyright 2020-2022 Clemens Lutz
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

use super::DataPoint;
use crate::error::{ErrorKind, Result};
use numa_gpu::runtime::allocator::{Allocator, DerefMemType, MemType};
use numa_gpu::runtime::linux_wrapper::{MemProtect, MemProtectFlags};
use numa_gpu::runtime::memory::{DerefMem, Mem, MemLock};
use rustacuda::context::{Context, ContextFlags, CurrentContext};
use rustacuda::device::{Device, DeviceAttribute};
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::{CopyDestination, DeviceBox};
use rustacuda::module::Module;
use rustacuda::stream::{Stream, StreamFlags};
use rustacuda::{launch, CudaFlags};
use std::cmp;
use std::convert::TryInto;
use std::ffi::{CStr, CString};
use std::iter;
use std::mem;
use std::ops::RangeInclusive;

#[cfg(not(target_arch = "aarch64"))]
use numa_gpu::runtime::nvml::{DeviceClocks, ThrottleReasons};
#[cfg(not(target_arch = "aarch64"))]
use nvml_wrapper::{enum_wrappers::device::Clock, NVML};

#[cfg(target_arch = "aarch64")]
use numa_gpu::runtime::hw_info::CudaDeviceInfo;

type Position = u64;

#[derive(Debug)]
pub(super) struct GpuTlbLatency {
    // `module` must be dropped before `context`. Rust specifies the drop order as the field order
    // in the struct. See RFC 1857: https://github.com/rust-lang/rfcs/pull/1857
    module: Module,
    context: Context,
    device: Device,
    device_id: u32,
    template: DataPoint,

    #[cfg(not(target_arch = "aarch64"))]
    nvml: nvml_wrapper::NVML,
}

impl GpuTlbLatency {
    #[cfg(not(target_arch = "aarch64"))]
    pub(super) fn new(device_id: u32, template: DataPoint) -> Result<Self> {
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id)?;
        let context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let module = Self::load_module()?;
        let nvml = NVML::init()?;

        let device_codename = device.name().map_err(|_| "Couldn't get device codename")?;
        let gpu_template = DataPoint {
            device_codename: Some(device_codename),
            ..template
        };

        Ok(Self {
            context,
            device,
            device_id,
            module,
            template: gpu_template,
            nvml,
        })
    }

    #[cfg(target_arch = "aarch64")]
    pub(super) fn new(device_id: u32, template: DataPoint) -> Result<Self> {
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id)?;
        let context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let module = Self::load_module()?;

        let device_codename = device.name().map_err(|_| "Couldn't get device codename")?;
        let gpu_template = DataPoint {
            device_codename: Some(device_codename),
            ..template
        };

        Ok(Self {
            context,
            device,
            device_id,
            module,
            template: gpu_template,
        })
    }

    fn load_module() -> Result<Module> {
        let module_path = CString::new(env!("CUDAUTILS_PATH")).map_err(|_| {
            ErrorKind::NulCharError(
                "Failed to load CUDA module, check your CUDAUTILS_PATH".to_string(),
            )
        })?;
        let module = Module::load_from_file(&module_path)?;

        Ok(module)
    }

    pub(super) fn measure(
        &self,
        mem_type: MemType,
        ranges: RangeInclusive<usize>,
        strides: &[usize],
        do_iotlb_flush: bool,
    ) -> Result<Vec<DataPoint>> {
        let device = CurrentContext::get_device()?;
        let grid_size =
            GridSize::from(device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32);
        let block_size =
            BlockSize::from(device.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32);
        let max_shared_mem_bytes =
            device.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlockOptin)? as u32;

        let mut tlb_data_points: usize =
            max_shared_mem_bytes as usize / (mem::size_of::<u32>() + mem::size_of::<u64>());
        if !tlb_data_points.is_power_of_two() {
            // Set number of points to next-lower power of two
            tlb_data_points = tlb_data_points.checked_next_power_of_two().ok_or_else(|| {
                ErrorKind::IntegerOverflow("Overflow when computing shared memory size".to_string())
            })? >> 1;
        }

        let position_bytes = mem::size_of::<Position>();
        let data_bytes = *ranges.end();
        let data_len = data_bytes / position_bytes;
        let mut data: Mem<Position> = Allocator::alloc_mem(mem_type, data_len);
        data.mlock()?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // Touch the memory from the GPU to stabalize the measurements
        unsafe {
            let module = &self.module;
            launch!(module.initialize_strides<<<grid_size.clone(), block_size.clone(), 0, stream>>>(
            data.as_launchable_ptr(),
            data.len(),
            0,
            1
            ))?;
        }
        stream.synchronize()?;

        let mut cycles: DerefMem<u32> =
            Allocator::alloc_deref_mem(DerefMemType::CudaPinnedMem, tlb_data_points);
        let mut indices: DerefMem<Position> =
            Allocator::alloc_deref_mem(DerefMemType::CudaPinnedMem, tlb_data_points);

        // Set a stable GPU clock rate to make the measurements more accurate
        #[cfg(not(target_arch = "aarch64"))]
        if let Err(e) = self
            .nvml
            .device_by_index(self.device_id as u32)?
            .set_max_gpu_clocks()
        {
            eprintln!("Warning: Failed to set the maximum GPU clockrate [{}]", e);
        }

        let cycle_counter_overhead = self.cycle_counter_overhead()?;

        // FIXME: factor into a closure, so that we can measure different GPU kernels

        let generate_range_iter = |stride| {
            let current_range = ranges
                .clone()
                .skip_while(move |r| r % stride != 0)
                .step_by(stride);
            let old_range = iter::once(0).chain(current_range.clone());
            current_range.zip(old_range).zip(iter::repeat(stride))
        };

        let data_points: Vec<DataPoint> = strides
            .iter()
            .cloned()
            .flat_map(generate_range_iter).map(|((range, old_range), stride)| {
            let module = &self.module;
            let size: usize = range / position_bytes;
            let iterations: u32 = cmp::min(
                tlb_data_points as u32,
                2 * range
                    .checked_div(stride)
                    .ok_or_else(|| ErrorKind::InvalidArgument("Stride is zero".to_string()))?
                    as u32,
            );

            match (&mut data).try_into() {
                Ok(slice) => {
                    let _: &mut [Position] = slice;
                    Self::write_strides(&mut slice[0..size], stride, Some(old_range / position_bytes));
                }
                Err((_, dev_slice)) => {
                    let start_offset: usize = 0;
                    let p_stride: usize = stride / position_bytes;
                    unsafe {
                        launch!(module.initialize_strides<<<grid_size.clone(), block_size.clone(), 0, stream>>>(
                                dev_slice.as_device_ptr(),
                                size,
                                start_offset,
                                p_stride
                                ))?;
                    }
                    stream.synchronize()?;
                }
            }

            // Flush the TLB for the data range so that it's cold when starting
            // the measurement.
            //
            // Note that we can only control the CPU's TLB, the GPU TLB is not
            // flushed.
            if do_iotlb_flush {
                match (&data).try_into() {
                    Ok(slice) => Self::flush_cpu_tlb(slice)?,
                    Err(_) => {}
                }
            }

            // Get GPU clock rate that applications run at
            #[cfg(not(target_arch = "aarch64"))]
            let clock_rate_mhz = self
                .nvml
                .device_by_index(self.device_id as u32)?
                .clock_info(Clock::Graphics)?;

            #[cfg(target_arch = "aarch64")]
            let clock_rate_mhz = CurrentContext::get_device()?.clock_rate()?;

            let kernel_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"tlb_stride_single_thread\0") };
            let mut kernel = module.get_function(kernel_name)?;
            kernel.set_max_dynamic_shared_size_bytes(max_shared_mem_bytes)?;

            unsafe {
                launch!(kernel<<<1, 1, max_shared_mem_bytes, stream>>>(
                data.as_launchable_ptr(),
                size,
                iterations,
                tlb_data_points,
                cycles.as_launchable_mut_ptr(),
                indices.as_launchable_mut_ptr()
                ))?;
            }
            stream.synchronize()?;

            // Check if GPU is running in a throttled state
            #[cfg(not(target_arch = "aarch64"))]
            let throttle_reasons: Option<ThrottleReasons> = Some(
                self.nvml
                    .device_by_index(self.device_id as u32)?
                    .current_throttle_reasons()?
                    .into(),
            );

            #[cfg(target_arch = "aarch64")]
            let throttle_reasons: Option<&str> = None;

            // Indexes are written as the "next" index. To get the actual index,
            // the stride must be subtracted.
            //
            // Index == -1 indicates an uninitialized measurement value, that
            // should be filtered out.
            let fixup_index = |&index| {
                if index as i64 == -1 {
                    -1
                }
                else {
                    ((index * position_bytes as u64 - stride as u64) % range as u64) as i64
                }
            };

            let warm_cold = |stride_id| {
                if iterations <= tlb_data_points as u32 {
                if stride_id < range / stride {
                    Some("cold".to_string())
                } else {
                    Some("warm".to_string())
                }
                } else {
                    None
                }
            };

            let dp: Vec<DataPoint> = cycles
                .iter()
                .take(iterations as usize)
                .zip(indices.iter().map(fixup_index))
                .enumerate()
                .map(|(stride_id, (&cycles, index))| DataPoint {
                    grid_size: Some(1),
                    block_size: Some(1),
                    iotlb_flush: Some(do_iotlb_flush),
                    range_bytes: Some(range),
                    stride_bytes: Some(stride),
                    throttle_reasons: throttle_reasons.as_ref().map(|r| r.to_string()),
                    clock_rate_mhz: Some(clock_rate_mhz),
                    cycle_counter_overhead_cycles: Some(cycle_counter_overhead),
                    stride_id: Some(stride_id),
                    index_bytes: Some(index),
                    tlb_status: warm_cold(stride_id),
                    cycles: Some(cycles),
                    ns: Some((1000 * cycles) / clock_rate_mhz),
                    ..self.template.clone()
                })
                .collect();

            Ok(dp)
        }).collect::<Result<Vec<Vec<DataPoint>>>>()?.concat();

        Ok(data_points)
    }

    /// Measures the overhead of `get_clock()` in cycles
    ///
    /// `get_clock()` is implemented as a native special register on each
    /// architecture. See the documentation in the CUDA module for details.
    fn cycle_counter_overhead(&self) -> Result<u32> {
        let mut overhead = DeviceBox::new(&0_u32)?;
        let mut fake_dependency = DeviceBox::new(&0_u32)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let module = &self.module;

        unsafe {
            launch!(module.cycle_counter_overhead<<<1, 1, 0, stream>>>(
            overhead.as_device_ptr(),
            fake_dependency.as_device_ptr()
            ))?;
        }
        stream.synchronize()?;

        let mut overhead_host = 0_u32;
        overhead.copy_to(&mut overhead_host)?;

        Ok(overhead_host)
    }

    /// Initializes a slice with strides
    ///
    /// After initialization, each element of the slice contains the position of
    /// its successor. For example, element `4` points to element `8` if the
    /// stride is `4` (as measured in `size_of::<Position>()`).
    ///
    /// If the slice is being resized without changing the stride, then
    /// `old_len` may be set to specify the previous slice length to reduce
    /// initialization time.
    fn write_strides(data: &mut [Position], stride_bytes: usize, old_len: Option<usize>) -> usize {
        let position_bytes = mem::size_of::<Position>();
        let len = data.len() as Position;

        let start_offset = match old_len {
            None | Some(0) => 0,
            Some(ol) => ol - stride_bytes / mem::size_of::<Position>(),
        };

        let number_written = data
            .iter_mut()
            .zip((stride_bytes / position_bytes) as Position..)
            .skip(start_offset)
            .map(|(it, next)| *it = next % len)
            .count();

        number_written
    }

    /// Flushes the CPU's TLB
    ///
    /// The flush is atomic and system-wide. The flush also includes the IOTLB.
    ///
    /// For a high-level description, refer to the book Gorman "Understanding
    /// the Linux Virtual Memory Manager" p. 44, Table 3.2"
    ///
    /// For the code, see the Linux kernel:
    /// https://code.woboq.org/linux/linux/mm/hugetlb.c.html#hugetlb_change_protection
    fn flush_cpu_tlb(data: &[Position]) -> Result<()> {
        data.mprotect(MemProtectFlags::NONE)?;
        data.mprotect(MemProtectFlags::READ | MemProtectFlags::WRITE)?;

        Ok(())
    }
}

impl Drop for GpuTlbLatency {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "aarch64"))]
        self.nvml
            .device_by_index(self.device_id as u32)
            .unwrap()
            .set_default_gpu_clocks()
            .expect("Failed to reset default GPU clock rates");
    }
}
