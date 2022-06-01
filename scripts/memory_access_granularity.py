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
# This script measures memory bandwidth. The parameters are tuned for an IBM
# AC922 machine, which has two IBM POWER9 CPUs and two Nvidia V100 GPUs.  The
# machine also has NVLink 2.0.
#
# Bandwidth is measured for multiple configurations:
#
#  - The GPU accessing CUDA device memory
#  - The GPU accessing CUDA device memory with a warp misalignment
#  - The GPU accessing Linux-allocated GPU memory
#  - The GPU accessing Linux-allocated GPU memory with a warp misalignment
#  - The GPU accessing main memory
#  - The GPU accessing main memory with a warp misalignment
#  - The CPU accessing main memory
#  - The CPU accessing main memory with a warp misalignment
#
# Each of these configurations measures sequential and random access patterns
# for a load, a store, and an atomic CAS.
#
# This script works on PCI-e and NVLink GPUs, as well as POWER9 and x86_64
# CPUs.
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
# Memory bandwidth is not affected by page fragmentation.

import subprocess
import socket
import itertools
import shlex
import tempfile
from os import path
import pandas

repeat = 10

devices = [
        { 'type': 'gpu', 'id': 0, 'mem_type': 'device', 'mem_location': 0, 'misalign_warp': False },
        { 'type': 'gpu', 'id': 0, 'mem_type': 'device', 'mem_location': 0, 'misalign_warp': True },
        { 'type': 'gpu', 'id': 0, 'mem_type': 'numa', 'mem_location': 255, 'misalign_warp': False },
        { 'type': 'gpu', 'id': 0, 'mem_type': 'numa', 'mem_location': 255, 'misalign_warp': True },
        { 'type': 'gpu', 'id': 0, 'mem_type': 'numa', 'mem_location': 0, 'misalign_warp': False },
        { 'type': 'gpu', 'id': 0, 'mem_type': 'numa', 'mem_location': 0, 'misalign_warp': True },
        { 'type': 'cpu', 'id': 0, 'mem_type': 'numa', 'mem_location': 0, 'misalign_warp': False }
        ]
grid_size = [ 80, 160 ]
block_size = [ 128, 256, 512, 1024 ]
cpu_threads = [ 16, 32, 64 ]
data_size = [ 1024 ] # MB
page_type = [ 'Huge2MB' ]

# Balancing workload across L2+L3 cache sub-systems (shared by a core-pair)
cpu_mapping_1s_4way_smt = { 'sockets': 1, 'threads': 64, 'smt': 4, 'mapping': '0 8 16 24 32 40 48 56 4 12 20 28 36 44 52 60 1 9 17 25 33 41 49 57 5 13 21 29 37 45 53 61 2 10 18 26 34 42 50 58 6 14 22 30 38 46 54 62 3 11 19 27 35 43 51 59 7 15 23 31 39 47 55 63' }
cpu_mapping = cpu_mapping_1s_4way_smt

hostname = socket.gethostname()

def main():
    file_id = 0
    file_list = []

    out_dir = tempfile.mkdtemp()
    out_csv = path.join(out_dir, f'memory_access_granularity_{hostname}.csv')

    cpu_affinity_file = path.join(out_dir, 'cpu_affinity.txt')
    with open(cpu_affinity_file, mode='w') as file:
        file.write(cpu_mapping['mapping'] + '\n')

    print(f"Writing CSV file to {out_csv}")

    for dev, ds, pt in itertools.product(devices, data_size, page_type):

        dt = dev['type']
        di = dev['id']
        mt = dev['mem_type']
        ml = dev['mem_location']
        mw = dev['misalign_warp']
        threads = ','.join(map(lambda x: str(x), cpu_threads))
        grids = ','.join(map(lambda x: str(x), grid_size))
        blocks = ','.join(map(lambda x: str(x), block_size))

        print(f'Running { dt } with mem type: { mt } location: {ml !s} and data size: {ds !s} MB page type: { pt }')

        tmp_csv = path.join(out_dir, f'tmp_{file_id !s}.csv')

        cmd = f'''
        cargo run                                          \
          --quiet                                          \
          --package microbench                             \
          --release                                        \
          --                                               \
          --csv {tmp_csv}                                  \
          bandwidth                                        \
          --device-type {dt}                               \
          --device-id {di !s}                              \
          --threads {threads}                              \
          --grid-sizes {grids}                             \
          --block-sizes {blocks}                           \
          --size {ds !s}                                   \
          --mem-location {ml !s}                           \
          --mem-type {mt}                                  \
          --page-type {pt}                                 \
          --misalign-warp { str(mw).lower() }              \
          --cpu-affinity {cpu_affinity_file}               \
          --repeat {repeat !s}
        '''

        cmdfuture = subprocess.run(shlex.split(cmd), check = False)
        cmdfuture.check_returncode()

        file_list.append(tmp_csv)
        file_id += 1

    csv_append(out_csv, file_list)

    print(f"Finished CSV file at {out_csv}")

def csv_append(accumulator_file, append_files):
    df_list = [pandas.read_csv(f) for f in append_files]
    df = pandas.concat(df_list)
    df.to_csv(accumulator_file, index = False)

if __name__ == "__main__":
    main()

