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
# This script measures the TLB latency. The parameters are tuned for an IBM
# AC922 machine, which has two IBM POWER9 CPUs and two Nvidia V100 GPUs.  The
# machine also has NVLink 2.0.
#
# This script is only tested on a GPU with NVLink 2.0; it might produce
# incorrect results on a PCI-e GPU!
#
# Setup notes
# ===========
#
# Before running this benchmark, follow our guide to allocate huge pages on
# early boot to avoid page fragmentation:
#
# ./guides/huge_pages.md#reserving-2-mb-huge-pages-on-early-boot
#
# This measurement is _very_ sensitive to page fragmentation! Reboot the
# machine to defrag.

import subprocess
import socket
import itertools
import shlex
import tempfile
from os import path
import pandas

hostname = socket.gethostname()

device_id = 0

parameters = [
        {
            "description": "GPU TLB L1 | GPU mem",
            "mem_type": "device",
            "mem_location": None,
            "range": (4, 80),
            "strides": [ 1024, 2048, 4096 ],
            "page_type": [ "Huge2MB" ]
        },
        {
            "description": "GPU TLB L2 | GPU mem",
            "mem_type": "device",
            "mem_location": None,
            "range": (1024, 10944),
            "strides": [ 16384, 32768, 65536 ],
            "page_type": [ "Huge2MB" ]
        },
        {
            "description": "GPU TLB L1 | CPU mem",
            "mem_type": "numa",
            "mem_location": 0,
            "range": (4, 80),
            "strides": [ 1024, 2048, 4096 ],
            "page_type": [ "Huge2MB" ]
        },
        {
            "description": "GPU TLB L2 | CPU mem",
            "mem_type": "numa",
            "mem_location": 0,
            "range": (6144, 12160),
            "strides": [ 16384, 32768, 65536 ],
            "page_type": [ "Huge2MB" ]
        },
        {
            "description": "I/O radix MMU TLB | CPU mem",
            "mem_type": "numa",
            "mem_location": 0,
            "range": (1024, 89600),
            "strides": [ 16384, 32768, 65536 ],
            "page_type": [ "Huge2MB" ]
        }
    ]

def main():
    file_id = 0
    file_list = []

    out_dir = tempfile.mkdtemp()
    out_csv = path.join(out_dir, f'tlb_latency_{hostname}_gpu.csv.xz')

    print(f"Writing CSV file to {out_csv}")

    for params in parameters:
        print(params["description"])

        mem_location = params["mem_location"]
        if mem_location is None:
            mem_location = 0

        strides = ','.join([ str(s) for s in params["strides"] ])

        for page_type in params["page_type"]:
            print(f'  Running page_type: {page_type}', flush=True)

            tmp_csv = path.join(out_dir, f'tmp_{file_id !s}.csv')

            cmd = f'''                                       \
            cargo run                                        \
              --quiet                                        \
              --release                                      \
              --package microbench                           \
              --                                             \
              --csv {tmp_csv}                                \
              tlb-latency                                    \
              --device-id {device_id !s}                     \
              --mem-type {params["mem_type"]}                \
              --mem-location {mem_location !s}               \
              --range-lower {params["range"][0] !s}          \
              --range-upper {params["range"][1] !s}          \
              --strides {strides}                            \
              --page-type {page_type}
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
    df.to_csv(accumulator_file, index = False, compression = 'xz')

if __name__ == "__main__":
    main()
