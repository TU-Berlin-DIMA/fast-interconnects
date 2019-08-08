use average::{concatenate, impl_from_iterator, Estimate, Max, Min, Quantile, Variance};

use numa_gpu::runtime::numa::{self, NumaMemory};

use serde_derive::Serialize;

use std::io;
use std::mem::size_of;
use std::time::{Duration, Instant};
use std::u8;

#[derive(Debug, Serialize)]
pub struct DataPoint<'h> {
    pub hostname: &'h String,
    pub warm_up: bool,
    pub bytes: usize,
    pub threads: usize,
    pub cpu_node: u16,
    pub src_node: u16,
    pub dst_node: u16,
    pub ns: u64,
}

pub struct NumaMemcopy {
    src: NumaMemory<u8>,
    dst: NumaMemory<u8>,
    cpu_node: u16,
    thread_pool: rayon::ThreadPool,
}

impl NumaMemcopy {
    pub fn new(
        size: usize,
        cpu_node: u16,
        src_node: u16,
        dst_node: u16,
        num_threads: usize,
    ) -> Self {
        // Force OS to allocate memory on the NUMA node that we say
        numa::set_strict(true);

        // Allocate NUMA memory
        let mut src = NumaMemory::new(size, src_node);
        let mut dst = NumaMemory::new(size, dst_node);

        // Ensure that arrays are physically backed by memory
        for (i, x) in src.as_mut_slice().iter_mut().by_ref().enumerate() {
            *x = (i % u8::MAX as usize) as u8;
        }
        for (i, x) in dst.as_mut_slice().iter_mut().by_ref().enumerate() {
            *x = ((i + 1) % u8::MAX as usize) as u8;
        }

        // Build thread pool
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .start_handler(move |_tid| numa::run_on_node(cpu_node).expect("Couldn't set NUMA node"))
            .build()
            .expect("Couldn't build Rayon thread pool");

        Self {
            src,
            dst,
            cpu_node,
            thread_pool,
        }
    }

    fn run_sequential(&mut self) -> Duration {
        numa::run_on_node(self.cpu_node).expect("Couldn't set NUMA node");

        let timer = Instant::now();
        unsafe {
            self.src
                .as_mut_slice()
                .as_mut_ptr()
                .copy_to_nonoverlapping(self.dst.as_mut_slice().as_mut_ptr(), self.src.len())
        };

        timer.elapsed()
    }

    fn run_rayon(&mut self) -> Duration {
        numa::run_on_node(self.cpu_node).expect("Couldn't set NUMA node");

        let threads = self.thread_pool.current_num_threads();
        let chunk_size = (self.src.len() + threads - 1) / threads;

        let src_partitions: Vec<&[u8]> = self.src.as_slice().chunks(chunk_size).collect();
        let dst_partitions: Vec<&mut [u8]> =
            self.dst.as_mut_slice().chunks_mut(chunk_size).collect();

        assert!(threads == src_partitions.len());
        assert!(threads == dst_partitions.len());

        let timer = Instant::now();

        self.thread_pool.scope(|s| {
            for ((_tid, local_src), local_dst) in
                (0..threads).zip(src_partitions).zip(dst_partitions)
            {
                s.spawn(move |_| {
                    unsafe {
                        local_src
                            .as_ptr()
                            .copy_to_nonoverlapping(local_dst.as_mut_ptr(), local_src.len())
                    };
                });
            }
        });

        timer.elapsed()
    }

    pub fn measure<W: io::Write>(&mut self, parallel: bool, writer: Option<&mut W>) {
        let repeat = 100;
        let hostname = hostname::get_hostname().expect("Couldn't get hostname");

        let mut measurements: Vec<DataPoint> = Vec::new();
        let mut warm_up = true;

        for _ in 0..repeat {
            let dur = if !parallel {
                self.run_sequential()
            } else {
                self.run_rayon()
            };

            let ns = dur.as_secs() * 10_u64.pow(9) + dur.subsec_nanos() as u64;

            measurements.push(DataPoint {
                hostname: &hostname,
                warm_up,
                bytes: self.src.len() * size_of::<u8>(),
                threads: self.thread_pool.current_num_threads(),
                cpu_node: self.cpu_node,
                src_node: self.src.node(),
                dst_node: self.dst.node(),
                ns,
            });

            warm_up = false;
        }

        if let Some(w) = writer {
            let mut csv = csv::Writer::from_writer(w);
            measurements
                .iter()
                .try_for_each(|row| csv.serialize(row))
                .expect("Couldn't write serialized measurements")
        }

        concatenate!(
            Estimator,
            [Variance, variance, mean, error],
            [Quantile, quantile, quantile],
            [Min, min, min],
            [Max, max, max]
        );

        let si_scale_factor = 10_f64.powf(9.0) / 2_f64.powf(30.0);
        let bw_scale_factor = 2.0;
        let stats: Estimator = measurements
            .iter()
            .map(|row| (row.bytes as f64, row.ns as f64))
            .map(|(bytes, ns)| bytes / ns)
            .collect();

        println!(
            r#"NUMA memcopy benchmark
Sample size: {}
               Throughput      Bandwidth
              GiB/s   GB/s   GiB/s   GB/s
Mean:        {:6.2} {:6.2}  {:6.2} {:6.2}
Stddev:      {:6.2} {:6.2}  {:6.2} {:6.2}
Median:      {:6.2} {:6.2}  {:6.2} {:6.2}
Min:         {:6.2} {:6.2}  {:6.2} {:6.2}
Max:         {:6.2} {:6.2}  {:6.2} {:6.2}"#,
            measurements.len(),
            stats.mean() * si_scale_factor,
            stats.mean(),
            stats.mean() * si_scale_factor * bw_scale_factor,
            stats.mean() * bw_scale_factor,
            stats.error() * si_scale_factor,
            stats.error(),
            stats.error() * si_scale_factor * bw_scale_factor,
            stats.error() * bw_scale_factor,
            stats.quantile() * si_scale_factor,
            stats.quantile(),
            stats.quantile() * si_scale_factor * bw_scale_factor,
            stats.quantile() * bw_scale_factor,
            stats.min() * si_scale_factor,
            stats.min(),
            stats.min() * si_scale_factor * bw_scale_factor,
            stats.min() * bw_scale_factor,
            stats.max() * si_scale_factor,
            stats.max(),
            stats.max() * si_scale_factor * bw_scale_factor,
            stats.max() * bw_scale_factor
        );
    }
}