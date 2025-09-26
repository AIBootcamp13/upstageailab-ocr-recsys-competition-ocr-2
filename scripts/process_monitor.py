#!/usr/bin/env python3
"""
Process Monitor and Cleanup Utility for OCR Training

This utility helps monitor and clean up orphaned training processes
and their worker processes that may be left behind from crashed or
interrupted training runs.
"""

import argparse
import os
import signal
import subprocess
import sys


def get_training_processes() -> list[tuple[int, str, str]]:
    """Get all processes related to OCR training.

    Returns:
        List of tuples (pid, command, user)
    """
    try:
        # Get all processes with training-related commands
        result = subprocess.run(
            ["pgrep", "-f", "runners/train.py"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode not in (0, 1):  # 1 means no matches found
            return []

        pids = result.stdout.strip().split("\n") if result.stdout.strip() else []

        processes = []
        for pid in pids:
            if not pid.strip():
                continue

            try:
                # Get process details
                ps_result = subprocess.run(
                    ["ps", "-p", pid.strip(), "-o", "pid,ppid,cmd,user"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if ps_result.returncode == 0:
                    lines = ps_result.stdout.strip().split("\n")
                    if len(lines) >= 2:  # Header + data
                        parts = lines[1].split()
                        if len(parts) >= 4:
                            pid_num = int(parts[0])
                            parts[1]
                            user = parts[-1]
                            cmd = " ".join(parts[2:-1])
                            processes.append((pid_num, cmd, user))

            except (ValueError, subprocess.TimeoutExpired):
                continue

        return processes

    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Fallback method using ps
        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=10)

            processes = []
            for line in result.stdout.split("\n"):
                if "runners/train.py" in line:
                    parts = line.split()
                    if len(parts) >= 11:  # Standard ps aux format
                        user = parts[0]
                        pid = int(parts[1])
                        cmd = " ".join(parts[10:])
                        processes.append((pid, cmd, user))

            return processes

        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("Error: Unable to run process monitoring commands (pgrep/ps not found)")
            return []


def get_worker_processes(parent_pids: list[int]) -> list[tuple[int, str, str]]:
    """Get worker processes spawned by training processes.

    Args:
        parent_pids: List of parent process IDs

    Returns:
        List of worker process tuples (pid, command, user)
    """
    if not parent_pids:
        return []

    workers = []
    try:
        # Find processes with PPID in parent_pids
        for parent_pid in parent_pids:
            result = subprocess.run(
                ["pgrep", "-P", str(parent_pid)],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode in (0, 1):
                child_pids = result.stdout.strip().split("\n") if result.stdout.strip() else []

                for pid in child_pids:
                    if not pid.strip():
                        continue

                    try:
                        ps_result = subprocess.run(
                            ["ps", "-p", pid.strip(), "-o", "pid,cmd,user"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )

                        if ps_result.returncode == 0:
                            lines = ps_result.stdout.strip().split("\n")
                            if len(lines) >= 2:
                                parts = lines[1].split()
                                if len(parts) >= 3:
                                    pid_num = int(parts[0])
                                    user = parts[-1]
                                    cmd = " ".join(parts[1:-1])
                                    workers.append((pid_num, cmd, user))

                    except (ValueError, subprocess.TimeoutExpired):
                        continue

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return workers


def terminate_processes(processes: list[tuple[int, str, str]], force: bool = False) -> int:
    """Terminate the given processes.

    Args:
        processes: List of process tuples (pid, command, user)
        force: Whether to use SIGKILL instead of SIGTERM

    Returns:
        Number of processes successfully terminated
    """
    if not processes:
        return 0

    signal_type = signal.SIGKILL if force else signal.SIGTERM
    signal_name = "SIGKILL" if force else "SIGTERM"

    terminated = 0
    for pid, cmd, user in processes:
        try:
            print(f"Terminating process {pid} ({cmd[:50]}...) with {signal_name}")
            os.kill(pid, signal_type)
            terminated += 1
        except (ProcessLookupError, PermissionError) as e:
            print(f"Failed to terminate process {pid}: {e}")
        except Exception as e:
            print(f"Unexpected error terminating process {pid}: {e}")

    return terminated


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Monitor and cleanup orphaned OCR training processes")
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all training processes and their workers",
    )
    parser.add_argument(
        "--cleanup",
        "-c",
        action="store_true",
        help="Terminate all orphaned training processes and workers",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Use SIGKILL instead of SIGTERM for forceful termination",
    )
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Show what would be done without actually doing it",
    )

    args = parser.parse_args()

    if not args.list and not args.cleanup:
        parser.print_help()
        return 1

    # Get training processes
    training_procs = get_training_processes()
    if training_procs:
        worker_procs = get_worker_processes([pid for pid, _, _ in training_procs])
    else:
        worker_procs = []

    all_procs = training_procs + worker_procs

    if args.list:
        if not all_procs:
            print("No training processes found.")
            return 0

        print(f"Found {len(training_procs)} training process(es) and {len(worker_procs)} worker process(es):")
        print("-" * 80)

        for pid, cmd, user in all_procs:
            proc_type = "TRAIN" if (pid, cmd, user) in training_procs else "WORKER"
            print("2d")
        print("-" * 80)

    if args.cleanup:
        if not all_procs:
            print("No processes to clean up.")
            return 0

        if args.dry_run:
            print(f"DRY RUN: Would terminate {len(all_procs)} process(es)")
            for pid, cmd, user in all_procs:
                proc_type = "TRAIN" if (pid, cmd, user) in training_procs else "WORKER"
                print(f"  Would terminate {proc_type} process {pid}: {cmd[:50]}...")
        else:
            print(f"Terminating {len(all_procs)} process(es)...")
            terminated = terminate_processes(all_procs, force=args.force)

            if terminated > 0:
                print(f"Successfully terminated {terminated} process(es)")

                # Wait a moment and check if any are still running
                import time

                time.sleep(1)

                remaining = get_training_processes()
                if remaining:
                    remaining_workers = get_worker_processes([pid for pid, _, _ in remaining])
                    total_remaining = len(remaining) + len(remaining_workers)
                    print(f"Warning: {total_remaining} process(es) may still be running")
                else:
                    print("All processes terminated successfully")
            else:
                print("No processes were terminated")

    return 0


if __name__ == "__main__":
    sys.exit(main())
