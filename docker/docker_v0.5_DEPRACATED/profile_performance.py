#!/usr/bin/env python3
"""
Performance profiling script for OCR training.
Monitors GPU utilization, memory usage, and training throughput.
"""

import argparse
import time
import warnings
from contextlib import contextmanager

# Suppress the pynvml deprecation warning before importing torch
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*", category=FutureWarning)

import psutil
import torch


@contextmanager
def gpu_profiler(device=0, log_interval=10):
    """Context manager for GPU profiling using PyTorch's native CUDA functions."""
    start_time = time.time()

    print("🚀 Starting GPU profiling...")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print("-" * 60)

    try:
        yield
    finally:
        end_time = time.time()
        total_time = end_time - start_time

        # Use PyTorch's native GPU memory monitoring
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        used_mem = total_mem - free_mem

        print("-" * 60)
        print(f"Total training time: {total_time:.2f}s")
        print(f"Final GPU Memory: {used_mem / 1024**2:.0f}MB / {total_mem / 1024**2:.0f}MB ({used_mem / total_mem * 100:.1f}%)")


def profile_system():
    """Profile system resources."""
    print("📊 System Profile:")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"Total RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f}GB")

    if torch.cuda.is_available():
        print("CUDA Available: Yes")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
    else:
        print("CUDA Available: No")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Profile OCR training performance")
    parser.add_argument("--profile-system", action="store_true", help="Profile system resources")
    parser.add_argument("--gpu-device", type=int, default=0, help="GPU device to profile")

    args = parser.parse_args()

    if args.profile_system:
        profile_system()
    else:
        print("Use --profile-system to see system profile")
        print("Use this script as a context manager in your training code:")
        print()
        print("from profile_performance import gpu_profiler")
        print()
        print("with gpu_profiler():")
        print("    # Your training code here")
        print("    trainer.fit(model, datamodule)")


if __name__ == "__main__":
    main()
