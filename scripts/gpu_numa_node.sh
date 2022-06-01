#!/usr/bin/env bash
#
# Copyright 2022 Clemens Lutz
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

for gpu in $(ls --directory -1 /proc/driver/nvidia/gpus/*)
do
    gpu_id=$(awk -e '/Device Minor/ { print $3 }' $gpu/information)
    numa_node=$(awk -e '/Node/ { print $2 }' $gpu/numa_status)

    echo "GPU memory of GPU $gpu_id is NUMA node $numa_node"
done
