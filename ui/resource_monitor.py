"""
Resource Monitor UI for OCR Training

A comprehensive monitoring interface for system resources, training processes,
and GPU utilization during OCR model training.
"""

import warnings

# Suppress known wandb Pydantic compatibility warnings
# This is a known issue where wandb uses incorrect Field() syntax in Annotated types
# The warnings come from Pydantic when processing wandb's type annotations
warnings.filterwarnings("ignore", message=r"The '(repr|frozen)' attribute.*Field.*function.*no effect", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*(repr|frozen).*Field.*function.*no effect", category=UserWarning)

# Also suppress by category for more reliable filtering
try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except ImportError:
    pass  # In case the warning class is not available in future pydantic versions

import contextlib
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import psutil
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.monitoring.process_monitor import get_training_processes, get_worker_processes, terminate_processes


def get_system_resources() -> dict[str, float]:
    """Get current system resource usage."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)

        # GPU information (if available)
        gpu_info = get_gpu_info()

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_used_gb": memory_used_gb,
            "memory_total_gb": memory_total_gb,
            **gpu_info,
        }
    except Exception as e:
        st.error(f"Error getting system resources: {e}")
        return {}


def get_gpu_info() -> dict[str, float]:
    """Get GPU information using nvidia-smi."""
    with contextlib.suppress(subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                # Take first GPU if multiple
                parts = [x.strip() for x in lines[0].split(",")]
                if len(parts) >= 4:
                    memory_used = float(parts[0])
                    memory_total = float(parts[1])
                    gpu_util = float(parts[3])

                    memory_free = float(parts[2])
                    return {
                        "gpu_memory_used_mb": memory_used,
                        "gpu_memory_total_mb": memory_total,
                        "gpu_memory_free_mb": memory_free,
                        "gpu_utilization_percent": gpu_util,
                        "gpu_memory_percent": ((memory_used / memory_total) * 100 if memory_total > 0 else 0),
                    }

    # Return empty dict if GPU info not available
    return {
        "gpu_memory_used_mb": 0,
        "gpu_memory_total_mb": 0,
        "gpu_memory_free_mb": 0,
        "gpu_utilization_percent": 0,
        "gpu_memory_percent": 0,
    }


def create_resource_charts(resources: dict[str, float]):
    """Create resource usage charts."""
    col1, col2 = st.columns(2)

    with col1:
        # CPU and Memory Usage
        st.subheader("🖥️ System Resources")

        # CPU Usage
        cpu_percent = resources.get("cpu_percent", 0)
        st.metric("CPU Usage", f"{cpu_percent:.1f}%")
        st.progress(min(cpu_percent / 100, 1.0), text=f"CPU: {cpu_percent:.1f}%")

        # Memory Usage
        memory_percent = resources.get("memory_percent", 0)
        memory_used = resources.get("memory_used_gb", 0)
        memory_total = resources.get("memory_total_gb", 0)
        st.metric("Memory Usage", f"{memory_percent:.1f}%")
        st.progress(
            min(memory_percent / 100, 1.0),
            text=f"RAM: {memory_used:.1f}/{memory_total:.1f} GB",
        )

    with col2:
        # GPU Usage (if available)
        st.subheader("🎮 GPU Resources")

        gpu_memory_percent = resources.get("gpu_memory_percent", 0)
        gpu_util = resources.get("gpu_utilization_percent", 0)

        if gpu_memory_percent > 0 or gpu_util > 0:
            # GPU Memory
            gpu_mem_used = resources.get("gpu_memory_used_mb", 0)
            gpu_mem_total = resources.get("gpu_memory_total_mb", 0)
            st.metric("GPU Memory", f"{gpu_memory_percent:.1f}%")
            st.progress(
                min(gpu_memory_percent / 100, 1.0),
                text=f"VRAM: {gpu_mem_used:.0f}/{gpu_mem_total:.0f} MB",
            )

            # GPU Utilization
            st.metric("GPU Utilization", f"{gpu_util:.1f}%")
            st.progress(min(gpu_util / 100, 1.0), text=f"GPU: {gpu_util:.1f}%")
        else:
            st.info("No GPU detected or nvidia-smi not available")


def display_process_table(processes: list[tuple[int, str, str]], title: str):
    """Display a table of processes with management options."""
    if not processes:
        st.info(f"No {title.lower()} found.")
        return

    st.subheader(f"📋 {title} ({len(processes)})")

    # Convert to DataFrame for better display
    df_data = []
    for pid, cmd, user in processes:
        df_data.append(
            {
                "PID": pid,
                "Command": cmd[:50] + "..." if len(cmd) > 50 else cmd,
                "User": user,
            }
        )

    df = pd.DataFrame(df_data)

    # Display table
    st.dataframe(df, width="stretch")

    # Process management
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(f"🔄 Refresh {title}", key=f"refresh_{title.lower().replace(' ', '_')}"):
            st.rerun()

    with col2:
        if st.button(
            f"⚠️ Terminate All {title}",
            key=f"terminate_{title.lower().replace(' ', '_')}",
            type="secondary",
        ):
            if st.session_state.get(f"confirm_terminate_{title.lower().replace(' ', '_')}", False):
                terminated = terminate_processes(processes, force=False)
                if terminated > 0:
                    st.success(f"Terminated {terminated} {title.lower()}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to terminate processes")
                st.session_state[f"confirm_terminate_{title.lower().replace(' ', '_')}"] = False
            else:
                st.session_state[f"confirm_terminate_{title.lower().replace(' ', '_')}"] = True
                st.warning(f"Click again to confirm termination of {len(processes)} {title.lower()}")

    with col3:
        if st.button(
            f"💀 Force Kill All {title}",
            key=f"force_kill_{title.lower().replace(' ', '_')}",
            type="secondary",
        ):
            if st.session_state.get(f"confirm_force_kill_{title.lower().replace(' ', '_')}", False):
                terminated = terminate_processes(processes, force=True)
                if terminated > 0:
                    st.success(f"Force killed {terminated} {title.lower()}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to force kill processes")
                st.session_state[f"confirm_force_kill_{title.lower().replace(' ', '_')}"] = False
            else:
                st.session_state[f"confirm_force_kill_{title.lower().replace(' ', '_')}"] = True
                st.error(f"⚠️ Click again to FORCE KILL {len(processes)} {title.lower()}")


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="OCR Resource Monitor", page_icon="📊", layout="wide")

    st.title("📊 OCR Training Resource Monitor")
    st.markdown("Monitor system resources, training processes, and GPU utilization in real-time")

    # Auto-refresh toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### System Overview")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)

    # Get current resources
    resources = get_system_resources()

    if resources:
        create_resource_charts(resources)
    else:
        st.error("Unable to retrieve system resource information")

    st.markdown("---")

    # Process monitoring
    st.markdown("### Process Monitoring")

    # Get training processes
    training_procs = get_training_processes()
    worker_procs = get_worker_processes([pid for pid, _, _ in training_procs]) if training_procs else []

    # Display process tables
    col1, col2 = st.columns(2)

    with col1:
        display_process_table(training_procs, "Training Processes")

    with col2:
        display_process_table(worker_procs, "Worker Processes")

    # Quick actions
    st.markdown("---")
    st.markdown("### Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔍 Scan for Processes", width="stretch"):
            st.rerun()

    with col2:
        if st.button("🧹 Clean All Processes", width="stretch", type="secondary"):
            all_procs = training_procs + worker_procs
            if all_procs:
                terminated = terminate_processes(all_procs, force=False)
                st.success(f"Cleaned up {terminated} processes")
                time.sleep(1)
                st.rerun()
            else:
                st.info("No processes to clean")

    with col3:
        if st.button("🚨 Emergency Stop", width="stretch", type="secondary"):
            all_procs = training_procs + worker_procs
            if all_procs:
                terminated = terminate_processes(all_procs, force=True)
                st.error(f"Emergency stop: terminated {terminated} processes")
                time.sleep(1)
                st.rerun()
            else:
                st.info("No processes running")

    with col4:
        if st.button("📋 View Logs", width="stretch"):
            st.info("Log viewer coming soon...")

    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This monitor provides real-time system resource information. "
        "Use the process management features carefully to avoid interrupting important training sessions."
    )

    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
