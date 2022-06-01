#!/usr/bin/env python3
#
# Copyright 2021-2022 Clemens Lutz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Information
# ===========
#
# This script runs a GPU radix join, and the Triton join. The parameters are
# tuned for an IBM AC922 machine, which has two IBM POWER9 CPUs and two Nvidia
# V100 GPUs. The machine also has NVLink 2.0.
#
# The joins run on one GPU. The prefix sum runs on either a CPU or the GPU. The
# CPU is faster for the prefix sum, because NVLink is slower than main memory
# for unidirectional transfers.
#
# The Triton join currently requires the GPU memory to be visible as a NUMA
# node in Linux. This is the case with NVLink 2.0, but not with PCI-e GPUs.
# However, the normal GPU radix join is able to run with PCI-e.
#
# Setup notes
# ===========
#
# Before running this benchmark, allocate huge pages by running:
#
# sudo bash -c 'echo 1 > /proc/sys/vm/compact_memory'
# sudo bash -c 'echo 63000 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages'
# sudo bash -c 'echo 10000 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_overcommit_hugepages'
#
# In contrast to the no-partitioning join, the radix joins are not affected by
# page fragmentation.

import math
import subprocess
import socket
import itertools
import shlex
import tempfile
from os import path
import pandas

repeat = 10

tuples = [ 128 * x for x in range(1, 17) ]
tuple_bytes = [ 16 ]
radix_bits_snd = 9 # Tuning on Volta shows 9 bits to have best throughput
data_location = 0
execution_strategies = [ 'GpuRadixJoinTwoPass', 'GpuTritonJoinTwoPass' ]
hashing_schemes = [ 'Perfect', 'BucketChaining' ]
prefix_sum_algorithms = [ 'CpuChunkedSimd', 'GpuChunked' ]
partition_algorithms = [ 'GpuHSSWWCv4' ]
page_type = [ 'Huge2MB' ]
dmem_buffer_size = 8

# Estimated for Volta GPU with bucket chaining based on tuning
max_shared_memory_bytes = 79000

# Use 40 CPU threads as a divisor of 80 chunks/morsels (Volta has 80 SMs)
# 64 threads cause a load imbalance (64 threads = 80 GiB/s; 40 threads = 110 GiB/s)
# Compensate by balancing workload across L2+L3 cache sub-systems (shared by a core-pair)
cpu_mapping_1s_4way_smt = { 'sockets': 1, 'threads': 40, 'smt': 4, 'mapping': '0 8 16 24 32 40 48 56 4 12 20 28 36 44 52 60 1 9 17 25 33 41 49 57 5 13 21 29 37 45 53 61 2 10 18 26 34 42 50 58 6 14 22 30 38 46 54 62 3 11 19 27 35 43 51 59 7 15 23 31 39 47 55 63' }
cpu_mapping = cpu_mapping_1s_4way_smt

hostname = socket.gethostname()

def main():
    file_id = 0
    file_list = []

    out_dir = tempfile.mkdtemp()
    out_csv = path.join(out_dir, f'benchmark_gpu_radix_join_{hostname}.csv')

    cpu_affinity_file = path.join(out_dir, 'cpu_affinity.txt')
    with open(cpu_affinity_file, mode='w') as file:
        file.write(cpu_mapping['mapping'] + '\n')

    print(f"Writing CSV file to {out_csv}")

    for ts, es, hs, ha, pa, tb, pt in itertools.product(tuples, execution_strategies, hashing_schemes, prefix_sum_algorithms, partition_algorithms, tuple_bytes, page_type):

        radix_bits_fst = math.ceil(math.log2(ts * 10**6 * tb / max_shared_memory_bytes)) - radix_bits_snd
        rb = (radix_bits_fst, radix_bits_snd)

        print(f'Running { es } with { ha } and { pa } and { hs } with tuples: {ts!s}M radix bits: {rb[0] !s}:{rb[1] !s} tuple bytes: {tb !s} page type: { pt }')

        for count in range(0, repeat):
            print('.', end='', flush=True)

            tmp_csv = path.join(out_dir, f'tmp_{file_id !s}.csv')

            cmd = f'''
            cargo run                                          \
              --quiet                                          \
              --package radix-join                             \
              --release                                        \
              --                                               \
              --execution-strategy { es }                      \
              --hashing-scheme { hs }                          \
              --histogram-algorithm { ha }                     \
              --partition-algorithm { pa }                     \
              --partition-algorithm-2nd GpuSSWWCv2             \
              --radix-bits {rb[0] !s},{rb[1] !s}               \
              --page-type { pt }                               \
              --dmem-buffer-size {dmem_buffer_size !s}         \
              --threads { cpu_mapping['threads'] !s}           \
              --cpu-affinity {cpu_affinity_file}               \
              --rel-mem-type numa                              \
              --inner-rel-location {data_location !s}          \
              --outer-rel-location {data_location !s}          \
              --partitions-mem-type numa                       \
              --partitions-location {data_location !s}         \
              --data-set Custom                                \
              --inner-rel-tuples {ts * 10**6 !s}               \
              --outer-rel-tuples {ts * 10**6 !s}               \
              --tuple-bytes Bytes{tb !s}                       \
              --repeat 2                                       \
              --csv {tmp_csv}
            '''

            cmdfuture = subprocess.run(shlex.split(cmd), check = False)
            cmdfuture.check_returncode()
            # print(cmdfuture.stderr)

            file_list.append(tmp_csv)
            file_id += 1

        print('')

    csv_append(out_csv, file_list)

    print(f"Finished CSV file at {out_csv}")

def csv_append(accumulator_file, append_files):
    df_list = [pandas.read_csv(f) for f in append_files]
    df = pandas.concat(df_list)
    df.to_csv(accumulator_file, index = False)

if __name__ == "__main__":
    main()

