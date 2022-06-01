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
# This script runs multiple GPU radix partitioning algorithms. The parameters
# are tuned for an IBM AC922 machine, which has two IBM POWER9 CPUs and two
# Nvidia V100 GPUs. The machine also has NVLink 2.0.
#
# The radix partitioning algorithms are able to run on PCI-e and NVLink GPUs.
#
# Setup notes
# ===========
#
# Before running this benchmark, follow our guide to allocate huge pages on
# early boot to avoid page fragmentation:
#
# ./guides/huge_pages.md#reserving-2-mb-huge-pages-on-early-boot
#
# Performance typically begins to degrade for very large data sizes after
# several runs due to page fragmentation (except for HSSWWCv4, which is not
# affected). Reboot the machine to defrag.

import subprocess
import socket
import itertools
import shlex
import tempfile
from os import path
import pandas

repeat = 10

data_bytes = 15 * 2**30
tuple_bytes = [ 8, 16 ]
radix_bits = range(0, 12)
data_location = 0
prefix_sum_algorithms = [ 'GpuChunked', 'CpuChunkedSimd' ]
partition_algorithms = [ ('SSWWCv2', 255), ('HSSWWCv4', 0) ]
page_type = [ 'Huge2MB' ]
dmem_buffer_size = 8

# Balancing workload across L2+L3 cache sub-systems (shared by a core-pair)
cpu_mapping_1s_4way_smt = { 'sockets': 1, 'threads': 40, 'smt': 4, 'mapping': '0 8 16 24 32 40 48 56 4 12 20 28 36 44 52 60 1 9 17 25 33 41 49 57 5 13 21 29 37 45 53 61 2 10 18 26 34 42 50 58 6 14 22 30 38 46 54 62 3 11 19 27 35 43 51 59 7 15 23 31 39 47 55 63' }
cpu_mapping = cpu_mapping_1s_4way_smt

hostname = socket.gethostname()

def main():
    file_id = 0
    file_list = []

    out_dir = tempfile.mkdtemp()
    out_csv = path.join(out_dir, f'benchmark_gpu_radix_partitioning_algorithms_{hostname}.csv')

    cpu_affinity_file = path.join(out_dir, 'cpu_affinity.txt')
    with open(cpu_affinity_file, mode='w') as file:
        file.write(cpu_mapping['mapping'] + '\n')

    psum_algos = ','.join(prefix_sum_algorithms)

    print(f"Writing CSV file to {out_csv}")

    for rb, tb, pt, (pa, ol) in itertools.product(radix_bits, tuple_bytes, page_type, partition_algorithms):
        print(f'Running { psum_algos } and { pa } with radix bits: {rb !s} tuple bytes: {tb !s} page type: { pt } output location: {ol !s}')

        tuples = int(data_bytes / tb)

        for count in range(0, repeat):
            print('.', end='', flush=True)

            tmp_csv = path.join(out_dir, f'tmp_{file_id !s}.csv')

            cmd = f'''
            cargo bench                                      \
              --quiet                                        \
              --package sql-ops                              \
              --bench gpu_radix_partition_operator           \
              --                                             \
              --histogram-algorithms { psum_algos }          \
              --partition-algorithms { pa }                  \
              --input-mem-type numa                          \
              --output-mem-type numa                         \
              --input-location {data_location !s}            \
              --output-location {ol !s}                      \
              --tuples {tuples !s}                           \
              --tuple-bytes Bytes{tb !s}                     \
              --radix-bits {rb !s}                           \
              --page-type { pt }                             \
              --dmem-buffer-sizes {dmem_buffer_size !s}      \
              --threads { cpu_mapping['threads'] !s}         \
              --cpu-affinity {cpu_affinity_file}             \
              --repeat 2                                     \
              --csv {tmp_csv}
            '''

            cmdfuture = subprocess.run(shlex.split(cmd), check = False)
            cmdfuture.check_returncode()

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

