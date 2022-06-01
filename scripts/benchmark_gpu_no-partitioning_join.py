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
# This script runs a GPU no-partitioning join. The parameters are tuned for an
# IBM AC922 machine, which has two IBM POWER9 CPUs and two Nvidia V100 GPUs.
# The machine also has NVLink 2.0.
#
# The no-partitioning join is able to run on PCI-e and NVLink GPUs.
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
# several runs due to page fragmentation. Reboot the machine to defrag.

import subprocess
import socket
import itertools
import shlex
import tempfile
from os import path
import math
import pandas

repeat = 10

tuples = [ 128 * x for x in range(1, 17) ]
tuple_bytes = [ 16 ]
data_location = 0
device_mem_bytes = 16 * 2**30
ht_location_gpu = 255
ht_location_cpu = 0
hybrid_hash_table = [ False ]
exec_method = [ 'Gpu' ]
hashing_scheme = [ ('Perfect', 1), ('LinearProbing', 2) ]
page_type = [ 'Huge2MB' ]

hostname = socket.gethostname()

def main():
    file_id = 0
    file_list = []

    out_dir = tempfile.mkdtemp()
    out_csv = path.join(out_dir, f'benchmark_gpu_no-partitioning_join_{hostname}.csv')

    print(f"Writing CSV file to {out_csv}")

    for em, ts, (hs, lf), hht, tb, pt in itertools.product(exec_method, tuples, hashing_scheme, hybrid_hash_table, tuple_bytes, page_type):
        print(f"Running {em} with {hs} and hybrid HT: {hht !s} with {ts!s}Mtuples tuple bytes: {tb !s} page type: {pt}")

        ht_loc = ht_location_gpu
        if not hht:
            if hs == 'Perfect' and ts * 10**6 * tb >= device_mem_bytes:
                ht_loc = ht_location_cpu
            elif hs == 'LinearProbing' and 2**math.ceil(math.log2(lf * ts * 10**6 * tb)) >= device_mem_bytes:
                ht_loc = ht_location_cpu

        for count in range(0, repeat):
            print('.', end='', flush=True)

            tmp_csv = path.join(out_dir, f'tmp_{file_id !s}.csv')

            cmd = f'''
            cargo run                                      \
              --quiet                                      \
              --package hashjoin                           \
              --release                                    \
              --                                           \
              --data-set custom                            \
              --execution-method {em}                      \
              --hash-table-mem-type Numa                   \
              --hash-table-location {ht_loc !s}            \
              --spill-hash-table { str(hht).lower() }      \
              --rel-mem-type Numa                          \
              --page-type {pt}                             \
              --inner-rel-location {data_location !s}      \
              --outer-rel-location {data_location !s}      \
              --inner-rel-tuples {ts * 10**6 !s}           \
              --outer-rel-tuples {ts * 10**6 !s}           \
              --hashing-scheme {hs}                        \
              --tuple-bytes Bytes{tb !s}                   \
              --repeat 2                                   \
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

