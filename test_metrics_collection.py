#!/usr/bin/env python3
"""
Standalone script to test metric collection for resource monitoring
"""

import time

import GPUtil
import psutil


def get_gpu_metrics():
    """
    Collect GPU metrics for all available GPUs.
    """
    gpus = GPUtil.getGPUs()
    metrics = {}
    for i, gpu in enumerate(gpus):
        metrics[f"gpu_{i}_util"] = gpu.load * 100
        metrics[f"gpu_{i}_memory_used_mb"] = gpu.memoryUsed
        metrics[f"gpu_{i}_memory_total_mb"] = gpu.memoryTotal
        metrics[f"gpu_{i}_memory_pct"] = gpu.memoryUtil * 100
        metrics[f"gpu_{i}_temp_c"] = gpu.temperature
    return metrics


def get_cpu_memory_metrics():
    """
    Collect CPU and memory metrics.
    """
    process = psutil.Process()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "cpu_process_percent": process.cpu_percent(),
        "memory_system_mb": psutil.virtual_memory().total / 1024 / 1024,
        "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
        "memory_percent": psutil.virtual_memory().percent,
        "memory_process_mb": process.memory_info().rss / 1024 / 1024,
    }


def get_io_metrics():
    """
    Collect disk I/O metrics.
    """
    # Get initial I/O stats
    initial_io = psutil.disk_io_counters()
    initial_time = time.time()
    initial_process_io = psutil.Process().io_counters()

    # Wait a short time to calculate rates
    time.sleep(0.5)

    # Get final I/O stats
    final_io = psutil.disk_io_counters()
    final_time = time.time()
    final_process_io = psutil.Process().io_counters()

    # Calculate time difference
    time_diff = final_time - initial_time

    # Calculate system-wide rates
    read_rate = (final_io.read_bytes - initial_io.read_bytes) / 1024 / 1024 / time_diff if time_diff > 0 else 0
    write_rate = (final_io.write_bytes - initial_io.write_bytes) / 1024 / 1024 / time_diff if time_diff > 0 else 0

    # Calculate process-specific rates
    process_read_rate = (final_process_io.read_bytes - initial_process_io.read_bytes) / 1024 / 1024 / time_diff if time_diff > 0 else 0
    process_write_rate = (final_process_io.write_bytes - initial_process_io.write_bytes) / 1024 / 1024 / time_diff if time_diff > 0 else 0

    return {
        "io_read_rate_mbps": round(read_rate, 2),
        "io_write_rate_mbps": round(write_rate, 2),
        "io_process_read_rate_mbps": round(process_read_rate, 2),
        "io_process_write_rate_mbps": round(process_write_rate, 2),
        "io_wait_time": psutil.disk_io_counters().busy_time,  # Not percentage, need to calculate differently
    }


if __name__ == "__main__":
    print("Testing resource metrics collection...")
    print("=" * 50)

    print("\nGPU Metrics:")
    try:
        gpu_metrics = get_gpu_metrics()
        for key, value in gpu_metrics.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"  GPU metrics collection failed: {str(e)}")

    print("\nCPU/Memory Metrics:")
    cpu_mem_metrics = get_cpu_memory_metrics()
    for key, value in cpu_mem_metrics.items():
        print(f"  {key}: {value}")

    print("\nI/O Metrics:")
    io_metrics = get_io_metrics()
    for key, value in io_metrics.items():
        print(f"  {key}: {value}")

    print("\nTest completed successfully!")
