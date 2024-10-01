# GPUtil - GPU utilization
#
# A Python module for programmically getting the GPU utilization from NVIDA GPUs using nvidia-smi
#
# Author: Anders Krogh Mortensen (anderskm)
# Date:   16 January 2017
# Web:    https://github.com/anderskm/gputil
#
# LICENSE
#
# MIT License
#
# Copyright (c) 2017 anderskm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# With slightly modified.

import subprocess
import os
import math
import random
import time
import sys
import platform
from distutils.spawn import find_executable


__version__ = '1.4.0'


class GPU:
    def __init__(
        self, index, uuid, load, memory_total, memory_used, memory_free, driver, 
        name, serial, display_mode, display_active, temperature,
    ):
        self.index = index
        self.uuid = uuid
        self.load = load
        self.memory_util = float(memory_used) / float(memory_total)
        self.memory_total = memory_total
        self.memory_used = memory_used
        self.memory_free = memory_free
        self.driver = driver
        self.name = name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active
        self.temperature = temperature

    def __str__(self):
        return f"Index: {self.index}, Load: {self.load}, Util: {self.memory_util}"


def safe_float_cast(str_number):
    try:
        return float(str_number)
    except ValueError:
        return float('nan')


def get_gpus():
    if platform.system() == "Windows":
        nvidia_smi = find_executable('nvidia-smi')
        if nvidia_smi is None:
            nvidia_smi = os.path.join(
                os.environ['systemdrive'], "Program Files", 
                "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"
            )
    else:
        nvidia_smi = "nvidia-smi"

    try:
        process = subprocess.Popen(
            [
                nvidia_smi, "--query-gpu=index,uuid,utilization.gpu,memory.total,"
                "memory.used,memory.free,driver_version,name,gpu_serial,display_active,"
                "display_mode,temperature.gpu", 
                "--format=csv,noheader,nounits"
            ], stdout=subprocess.PIPE,
        )
        stdout, _ = process.communicate()
    except Exception:
        return []

    output = stdout.decode('UTF-8')
    lines = output.split(os.linesep)
    num_devices = len(lines) - 1
    gpus = []

    for index in range(num_devices):
        line = lines[index]
        values = line.split(', ')
        gpu = GPU(
            index=int(values[0]),
            uuid=values[1],
            load=safe_float_cast(values[2]),
            memory_total=safe_float_cast(values[3]),
            memory_used=safe_float_cast(values[4]),
            memory_free=safe_float_cast(values[5]),
            driver=values[6],
            name=values[7],
            serial=values[8],
            display_mode=values[9],
            display_active=values[10],
            temperature=safe_float_cast(values[11])
        )
        gpus.append(gpu)

    return gpus


def get_available(
    order='first', limit=1, max_load=0.5, max_memory=0.5, memory_free=0, 
    include_nan=False, exclude_id=[], exclude_uuid=[],
):
    gpus = get_gpus()
    available_gpu_indices = [
        index for index, gpu in enumerate(gpus) if gpu.memory_free >= memory_free 
        and gpu.load < max_load and gpu.memory_util < max_memory 
        and gpu.index not in exclude_id and gpu.uuid not in exclude_uuid
    ]
    available_gpus = [gpus[index] for index in available_gpu_indices]

    # Sort available GPUs according to the order argument
    if order == 'first':
        available_gpus.sort(key=lambda x: x.index)
    elif order == 'last':
        available_gpus.sort(key=lambda x: x.index, reverse=True)
    elif order == 'random':
        random.shuffle(available_gpus)
    elif order == 'load':
        available_gpus.sort(key=lambda x: x.load)
    elif order == 'memory':
        available_gpus.sort(key=lambda x: x.memory_util)

    # Limit the number of GPUs
    available_gpus = available_gpus[:limit]
    return [gpu.index for gpu in available_gpus]


def get_device_info(device_ids=None):
    device_info = {}
    devices = get_gpus()
    devices = sorted(devices, key=lambda x: x.index)

    for gpu in devices:
        if device_ids is not None and gpu.index not in device_ids:
            continue

        device_info[gpu.index] = {
            "gpu_info": {
                "gpu_name": gpu.name,
                "memory_usage": {
                    "total": gpu.memory_total,
                    "used": gpu.memory_used,
                    "free": gpu.memory_free,
                },
                "gpu_utilization": gpu.load,
                "process": {}
            }
        }

    return device_info


def get_avg_device_utilization():
    """Get averaged GPU-Util and memory usage among all devices"""

    def avg(arr, round_digits=2):
        if not arr:
            return -1
        return round(sum(arr) / len(arr), round_digits)

    utilization_info = {}

    device_info = get_device_info()

    mem_occupy_rates = []
    device_utilizations = []

    for device_idx, device_info in device_info.items():
        curr_mem_total = device_info["gpu_info"]["memory_usage"]["total"]
        curr_mem_used = device_info["gpu_info"]["memory_usage"]["used"]
        mem_occupy_rate = curr_mem_used / curr_mem_total * 100
        mem_occupy_rates.append(mem_occupy_rate)

        curr_utilization = device_info["gpu_info"]["gpu_utilization"]
        device_utilizations.append(curr_utilization)

    utilization_info[f"Avg Memory-Usage"] = avg(mem_occupy_rates)
    utilization_info[f"Avg GPU-Util"] = avg(device_utilizations)

    return utilization_info


if __name__ == "__main__":
    start_time = time.time()
    print(get_avg_device_utilization())
    print(time.time() - start_time)
