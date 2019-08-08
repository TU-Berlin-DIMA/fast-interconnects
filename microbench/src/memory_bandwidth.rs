use numa_gpu::runtime::allocator::{Allocator, MemType};
use numa_gpu::runtime::hw_info;
use numa_gpu::runtime::memory::{DerefMem, Mem};
use numa_gpu::runtime::numa;
use numa_gpu::runtime::utils::EnsurePhysicallyBacked;

#[cfg(feature = "nvml")]
use nvml_wrapper::{enum_wrappers::device::Clock, NVML};

use rustacuda::context::CurrentContext;
use rustacuda::device::DeviceAttribute;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;

use serde_derive::Serialize;

use std::convert::{TryFrom, TryInto};
use std::iter;
use std::mem::size_of;
use std::ops::RangeInclusive;
use std::rc::Rc;
use std::time::Instant;

use crate::types::*;

extern "C" {
    fn cpu_bandwidth_seq(
        op: MemoryOperation,
        data: *mut u32,
        size: usize,
        tid: usize,
        num_threads: usize,
    );
    fn gpu_bandwidth_seq(
        op: MemoryOperation,
        data: *mut u32,
        size: u32,
        cycles: *mut u64,
        grid: u32,
        block: u32,
    );
    fn cpu_bandwidth_lcg(
        op: MemoryOperation,
        data: *mut u32,
        size: usize,
        tid: usize,
        num_threads: usize,
    );
    fn gpu_bandwidth_lcg(
        op: MemoryOperation,
        data: *mut u32,
        size: u32,
        cycles: *mut u64,
        grid: u32,
        block: u32,
    );
}

type GpuBandwidthFn = unsafe extern "C" fn(
    op: MemoryOperation,
    data: *mut u32,
    size: u32,
    cycles: *mut u64,
    grid: u32,
    block: u32,
);
type CpuBandwidthFn = unsafe extern "C" fn(
    op: MemoryOperation,
    data: *mut u32,
    size: usize,
    tid: usize,
    num_threads: usize,
);

#[derive(Clone, Debug)]
struct GpuNamedBandwidthFn<'n> {
    f: GpuBandwidthFn,
    name: &'n str,
}

#[derive(Clone, Debug)]
struct CpuNamedBandwidthFn<'n> {
    f: CpuBandwidthFn,
    name: &'n str,
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Serialize)]
enum MemoryOperation {
    Read,
    Write,
    CompareAndSwap,
}

pub struct MemoryBandwidth;

#[derive(Clone, Debug, Default, Serialize)]
struct DataPoint<'h, 'd, 'c, 'n> {
    pub hostname: &'h str,
    pub device_type: &'d str,
    pub device_codename: &'c str,
    pub function_name: &'n str,
    pub memory_operation: Option<MemoryOperation>,
    pub cpu_node: Option<u16>,
    pub memory_type: Option<BareMemType>,
    pub memory_node: Option<u16>,
    pub warm_up: bool,
    pub bytes: usize,
    pub threads: Option<ThreadCount>,
    pub grid_size: Option<Grid>,
    pub block_size: Option<Block>,
    pub ilp: Option<Ilp>,
    pub cycles: u64,
    pub ns: u64,
}

#[allow(dead_code)]
struct GpuMeasurement<'h, 'd, 'c, 'n> {
    oversub_ratio: RangeInclusive<OversubRatio>,
    warp_mul: RangeInclusive<WarpMul>,
    warp_size: Warp,
    sm_count: SM,
    ilp: RangeInclusive<Ilp>,
    template: DataPoint<'h, 'd, 'c, 'n>,
}

struct CpuMeasurement<'h, 'd, 'c, 'n> {
    threads: RangeInclusive<ThreadCount>,
    template: DataPoint<'h, 'd, 'c, 'n>,
}

#[allow(dead_code)]
struct GpuMeasurementParameters {
    grid_size: Grid,
    block_size: Block,
    ilp: Ilp,
}

#[allow(dead_code)]
#[derive(Debug)]
struct CpuMemoryBandwidth {
    cpu_node: u16,
}

#[derive(Debug)]
struct GpuMemoryBandwidth {
    device_id: u32,

    #[cfg(feature = "nvml")]
    nvml: nvml_wrapper::NVML,
}

impl MemoryBandwidth {
    pub fn measure<W>(
        device_id: DeviceId,
        mem_type: MemType,
        bytes: usize,
        threads: RangeInclusive<ThreadCount>,
        oversub_ratio: RangeInclusive<OversubRatio>,
        warp_mul: RangeInclusive<WarpMul>,
        ilp: RangeInclusive<Ilp>,
        repeat: u32,
        writer: Option<&mut W>,
    ) where
        W: std::io::Write,
    {
        let gpu_id = match device_id {
            DeviceId::Gpu(id) => id,
            _ => 0,
        };

        let device = Device::get_device(gpu_id).expect("Couldn't set CUDA device");
        let _context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .expect("Couldn't create CUDA context");

        numa::set_strict(true);

        let element_bytes = size_of::<u32>();
        let buffer_len = bytes / element_bytes;

        let hostname = hostname::get_hostname().expect("Couldn't get hostname");
        let (device_type, cpu_node) = match device_id {
            DeviceId::Cpu(id) => ("CPU", Some(id)),
            DeviceId::Gpu(_) => ("GPU", None),
        };
        let device_codename = match device_id {
            DeviceId::Cpu(_) => hw_info::cpu_codename(),
            DeviceId::Gpu(_) => device.name().expect("Couldn't get device code name"),
        };
        let mem_type_description: MemTypeDescription = (&mem_type).into();

        let template = DataPoint {
            hostname: hostname.as_str(),
            device_type,
            device_codename: device_codename.as_str(),
            cpu_node,
            memory_node: mem_type_description.location,
            memory_type: Some(mem_type_description.bare_mem_type),
            bytes,
            ..Default::default()
        };

        let mem = match DerefMem::<u32>::try_from(Allocator::alloc_mem(mem_type, buffer_len)) {
            Ok(mut demem) => {
                u32::ensure_physically_backed(demem.as_mut_slice());
                demem.into()
            }
            Err((_, mem)) => mem,
        };

        let latencies = match device_id {
            DeviceId::Cpu(cpu_node) => {
                let mnt = CpuMeasurement::new(threads, template);
                let demem: DerefMem<_> = mem.try_into().expect("Cannot run benchmark on CPU with the given type of memory. Did you specify GPU device memory?");
                mnt.measure(
                    &demem,
                    CpuMemoryBandwidth::new(cpu_node),
                    CpuMemoryBandwidth::run,
                    vec![
                        CpuNamedBandwidthFn {
                            f: cpu_bandwidth_seq,
                            name: "sequential",
                        },
                        CpuNamedBandwidthFn {
                            f: cpu_bandwidth_lcg,
                            name: "linear_congruential_generator",
                        },
                    ],
                    vec![
                        MemoryOperation::Read,
                        MemoryOperation::Write,
                        MemoryOperation::CompareAndSwap,
                    ],
                    repeat,
                )
            }
            DeviceId::Gpu(did) => {
                let warp_size = Warp(
                    device
                        .get_attribute(DeviceAttribute::WarpSize)
                        .expect("Couldn't get device warp size"),
                );
                let sm_count = SM(device
                    .get_attribute(DeviceAttribute::MultiprocessorCount)
                    .expect("Couldn't get device multiprocessor count"));
                let mnt = GpuMeasurement::new(
                    oversub_ratio,
                    warp_mul,
                    warp_size,
                    sm_count,
                    ilp,
                    template,
                );

                let ml = GpuMemoryBandwidth::new(did);
                let l = mnt.measure(
                    &mem,
                    ml,
                    GpuMemoryBandwidth::run,
                    vec![
                        GpuNamedBandwidthFn {
                            f: gpu_bandwidth_seq,
                            name: "sequential",
                        },
                        GpuNamedBandwidthFn {
                            f: gpu_bandwidth_lcg,
                            name: "linear_congruential_generator",
                        },
                    ],
                    vec![
                        MemoryOperation::Write,
                        MemoryOperation::Read,
                        MemoryOperation::CompareAndSwap,
                    ],
                    repeat,
                );
                l
            }
        };

        if let Some(w) = writer {
            let mut csv = csv::Writer::from_writer(w);
            latencies
                .iter()
                .try_for_each(|row| csv.serialize(row))
                .expect("Couldn't write serialized measurements")
        }
    }
}

impl<'h, 'd, 'c, 'n> GpuMeasurement<'h, 'd, 'c, 'n> {
    fn new(
        oversub_ratio: RangeInclusive<OversubRatio>,
        warp_mul: RangeInclusive<WarpMul>,
        warp_size: Warp,
        sm_count: SM,
        ilp: RangeInclusive<Ilp>,
        template: DataPoint<'h, 'd, 'c, 'n>,
    ) -> Self {
        Self {
            oversub_ratio,
            warp_mul,
            warp_size,
            sm_count,
            ilp,
            template,
        }
    }

    fn measure<R, S>(
        &self,
        mem: &Mem<u32>,
        mut state: S,
        run: R,
        futs: Vec<GpuNamedBandwidthFn<'n>>,
        ops: Vec<MemoryOperation>,
        repeat: u32,
    ) -> Vec<DataPoint<'h, 'd, 'c, 'n>>
    where
        R: Fn(
            GpuBandwidthFn,
            MemoryOperation,
            &mut S,
            &Mem<u32>,
            &GpuMeasurementParameters,
        ) -> (u64, u64),
    {
        // Convert newtypes to basic types while std::ops::Step is unstable
        // Step trait is required for std::ops::RangeInclusive Iterator trait
        let (OversubRatio(osr_l), OversubRatio(osr_u)) = self.oversub_ratio.clone().into_inner();
        let (WarpMul(wm_l), WarpMul(wm_u)) = self.warp_mul.clone().into_inner();
        let warp_mul_iter = wm_l..=wm_u;
        let warp_size = self.warp_size.clone();
        let sm_count = self.sm_count.clone();

        let data_points: Vec<_> = futs
            .iter()
            .flat_map(|fut| {
                iter::repeat(fut).zip(ops.iter().flat_map(|op| {
                    let oversub_ratio_iter = osr_l..=osr_u;
                    iter::repeat(op).zip(
                        oversub_ratio_iter
                            .filter(|osr| osr.is_power_of_two())
                            .map(|osr| OversubRatio(osr))
                            .flat_map(|osr| {
                                warp_mul_iter
                                    .clone()
                                    .filter(|wm| wm.is_power_of_two())
                                    .map(|wm| WarpMul(wm))
                                    .zip(std::iter::repeat(osr))
                                    .flat_map(|params| {
                                        iter::repeat(params)
                                            .zip(iter::once(true).chain(iter::repeat(false)))
                                            .zip(0..repeat)
                                    })
                            }),
                    )
                }))
            })
            .map(
                |(named_fut, (op, (((warp_mul, oversub_ratio), warm_up), _run)))| {
                    let block_size = warp_mul * warp_size;
                    let grid_size = oversub_ratio * sm_count;
                    let ilp = Ilp::default(); // FIXME: insert and use a real parameter
                    let mp = GpuMeasurementParameters {
                        grid_size,
                        block_size,
                        ilp,
                    };
                    let GpuNamedBandwidthFn { f: fut, name } = named_fut;

                    let (cycles, ns) = run(*fut, *op, &mut state, mem, &mp);

                    DataPoint {
                        function_name: name,
                        memory_operation: Some(*op),
                        warm_up,
                        grid_size: Some(grid_size),
                        block_size: Some(block_size),
                        ilp: None,
                        cycles,
                        ns,
                        ..self.template.clone()
                    }
                },
            )
            .collect();
        data_points
    }
}

impl<'h, 'd, 'c, 'n> CpuMeasurement<'h, 'd, 'c, 'n> {
    fn new(threads: RangeInclusive<ThreadCount>, template: DataPoint<'h, 'd, 'c, 'n>) -> Self {
        Self { threads, template }
    }

    fn measure<R, S>(
        &self,
        mem: &DerefMem<u32>,
        mut state: S,
        run: R,
        futs: Vec<CpuNamedBandwidthFn<'n>>,
        ops: Vec<MemoryOperation>,
        repeat: u32,
    ) -> Vec<DataPoint<'h, 'd, 'c, 'n>>
    where
        R: Fn(
            CpuBandwidthFn,
            MemoryOperation,
            &mut S,
            &DerefMem<u32>,
            Rc<rayon::ThreadPool>,
        ) -> (u64, u64),
    {
        let (ThreadCount(threads_l), ThreadCount(threads_u)) = self.threads.clone().into_inner();
        let cpu_node = 0;

        let data_points: Vec<_> = futs
            .iter()
            .flat_map(|fut| {
                iter::repeat(fut).zip(ops.iter().flat_map(|op| {
                    let threads_iter = threads_l..=threads_u;
                    iter::repeat(op).zip(threads_iter.flat_map(|t| {
                        let thread_pool = Rc::new(
                            rayon::ThreadPoolBuilder::new()
                                .num_threads(t)
                                .start_handler(move |_tid| {
                                    numa::run_on_node(cpu_node).expect("Couldn't set NUMA node")
                                })
                                .build()
                                .expect("Couldn't build Rayon thread pool"),
                        );

                        iter::repeat(thread_pool.clone())
                            .zip(iter::once(true).chain(iter::repeat(false)))
                            .zip(0..repeat)
                    }))
                }))
            })
            .map(|(named_fut, (op, ((thread_pool, warm_up), _run_number)))| {
                let threads = ThreadCount(thread_pool.current_num_threads());
                let CpuNamedBandwidthFn { f: fut, name } = named_fut;
                let (cycles, ns) = run(*fut, *op, &mut state, mem, thread_pool);

                DataPoint {
                    function_name: name,
                    memory_operation: Some(*op),
                    warm_up,
                    threads: Some(threads),
                    cycles,
                    ns,
                    ..self.template.clone()
                }
            })
            .collect();
        data_points
    }
}

impl GpuMemoryBandwidth {
    #[cfg(feature = "nvml")]
    fn new(device_id: u32) -> Self {
        let nvml = NVML::init().expect("Couldn't initialize NVML");

        Self { device_id, nvml }
    }

    #[cfg(not(feature = "nvml"))]
    fn new(device_id: u32) -> Self {
        Self { device_id }
    }

    fn run(
        f: GpuBandwidthFn,
        op: MemoryOperation,
        state: &mut Self,
        mem: &Mem<u32>,
        mp: &GpuMeasurementParameters,
    ) -> (u64, u64) {
        assert!(
            mem.len().is_power_of_two(),
            "Data size must be a power of two!"
        );

        let mut device_cycles = DeviceBox::new(&0_u64).expect("Couldn't allocate device memory");
        unsafe {
            f(
                op,
                mem.as_ptr() as *mut u32,
                mem.len() as u32,
                device_cycles.as_device_ptr().as_raw_mut(),
                mp.grid_size.0,
                mp.block_size.0,
            )
        };

        CurrentContext::synchronize().unwrap();

        // Get GPU clock rate that applications run at
        #[cfg(feature = "nvml")]
        let clock_rate_mhz = state
            .nvml
            .device_by_index(state.device_id as u32)
            .expect("Couldn't get NVML device")
            .clock_info(Clock::SM)
            .expect("Couldn't get clock rate with NVML");

        #[cfg(not(feature = "nvml"))]
        let clock_rate_mhz = CurrentContext::get_device()
            .expect("Couldn't get CUDA device")
            .clock_rate()
            .expect("Couldn't get clock rate");

        let mut cycles = 0;
        device_cycles
            .copy_to(&mut cycles)
            .expect("Couldn't transfer result from device");
        let ns: u64 = cycles * 1000 / (clock_rate_mhz as u64);

        (cycles, ns)
    }
}

impl CpuMemoryBandwidth {
    fn new(cpu_node: u16) -> Self {
        Self { cpu_node }
    }

    fn run(
        f: CpuBandwidthFn,
        op: MemoryOperation,
        _state: &mut Self,
        mem: &DerefMem<u32>,
        thread_pool: Rc<rayon::ThreadPool>,
    ) -> (u64, u64) {
        let threads = thread_pool.current_num_threads();
        let len = mem.len();

        let timer = Instant::now();

        thread_pool.scope(|s| {
            (0..threads)
                .zip(iter::repeat(mem))
                .for_each(|(tid, r_mem)| {
                    s.spawn(move |_| {
                        let ptr = r_mem.as_ptr() as *mut u32;

                        unsafe { f(op, ptr, len, tid, threads) };
                    });
                })
        });

        let duration = timer.elapsed();
        let ns: u64 = duration.as_secs() * 10_u64.pow(9) + duration.subsec_nanos() as u64;
        let cycles = 0;
        (cycles, ns)
    }
}