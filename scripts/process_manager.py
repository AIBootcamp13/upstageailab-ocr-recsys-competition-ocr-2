#!/usr/bin/env python3
"""
Streamlit Process Manager

This script provides proper process management for Streamlit UI applications
to prevent zombie processes and ensure clean startup/shutdown.
"""

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

from ocr.utils.path_utils import get_path_resolver, setup_project_paths

setup_project_paths()


class StreamlitProcessManager:
    """Manages Streamlit processes with proper lifecycle handling."""

    def __init__(self):
        self.project_root = get_path_resolver().config.project_root
        self.processes = {}

    def _get_ui_path(self, ui_name: str) -> Path:
        """Get the path to a UI script."""
        ui_paths = {
            "command_builder": "ui/command_builder.py",
            "evaluation_viewer": "ui/evaluation_viewer.py",
            "inference": "ui/inference_ui.py",
            "preprocessing_viewer": "ui/preprocessing_viewer_app.py",
            "resource_monitor": "ui/resource_monitor.py",
        }

        if ui_name not in ui_paths:
            raise ValueError(f"Unknown UI: {ui_name}. Available: {list(ui_paths.keys())}")

        return self.project_root / ui_paths[ui_name]

    def _get_pid_file(self, ui_name: str, port: int) -> Path:
        """Get the PID file path for a UI process."""
        return self.project_root / f".{ui_name}_{port}.pid"

    def _write_pid_file(self, ui_name: str, port: int, pid: int):
        """Write the PID file."""
        pid_file = self._get_pid_file(ui_name, port)
        with open(pid_file, "w") as f:
            f.write(str(pid))

    def _read_pid_file(self, ui_name: str, port: int) -> int | None:
        """Read the PID file."""
        pid_file = self._get_pid_file(ui_name, port)
        if pid_file.exists():
            try:
                with open(pid_file) as f:
                    return int(f.read().strip())
            except (ValueError, OSError):
                return None
        return None

    def _get_log_files(self, ui_name: str, port: int) -> tuple[Path, Path]:
        """Get the log file paths for a UI process."""
        log_dir = self.project_root / "logs" / "ui"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = log_dir / f"{ui_name}_{port}.out"
        stderr_log = log_dir / f"{ui_name}_{port}.err"
        return stdout_log, stderr_log

    def _remove_pid_file(self, ui_name: str, port: int):
        """Remove the PID file."""
        pid_file = self._get_pid_file(ui_name, port)
        if pid_file.exists():
            pid_file.unlink()

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is running."""
        try:
            os.kill(pid, 0)  # Signal 0 doesn't kill, just checks if process exists
            return True
        except OSError:
            return False

    def start(self, ui_name: str, port: int = 8501, background: bool = True, enable_logging: bool = True):
        """Start a Streamlit UI process."""
        ui_path = self._get_ui_path(ui_name)

        # Check if already running
        existing_pid = self._read_pid_file(ui_name, port)
        if existing_pid and self._is_process_running(existing_pid):
            print(f"UI {ui_name} is already running on port {port} (PID: {existing_pid})")
            return existing_pid

        # Build command
        cmd = [
            "uv",
            "run",
            "streamlit",
            "run",
            str(ui_path),
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--server.runOnSave",
            "false",
        ]

        print(f"Starting {ui_name} on port {port}...")

        if background:
            # Start in background with proper process group management
            if enable_logging:
                stdout_log, stderr_log = self._get_log_files(ui_name, port)
                with open(stdout_log, "w") as stdout_handle, open(stderr_log, "w") as stderr_handle:
                    print(f"Logs will be written to: {stdout_log} and {stderr_log}")
                    process = subprocess.Popen(
                        cmd,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        preexec_fn=os.setsid,  # Create new process group
                        cwd=self.project_root,
                    )
            else:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid,  # Create new process group
                    cwd=self.project_root,
                )

            # Wait a moment for process to start
            time.sleep(2)

            if self._is_process_running(process.pid):
                self._write_pid_file(ui_name, port, process.pid)
                print(f"Started {ui_name} (PID: {process.pid}) on port {port}")
                return process.pid
            else:
                print(f"Failed to start {ui_name}")
                return None
        else:
            # Run in foreground
            try:
                subprocess.run(cmd, cwd=self.project_root)
            except KeyboardInterrupt:
                print(f"\nStopped {ui_name}")
            return None

    def stop(self, ui_name: str, port: int = 8501):
        """Stop a Streamlit UI process."""
        pid = self._read_pid_file(ui_name, port)

        if not pid:
            print(f"No PID file found for {ui_name} on port {port}")
            return False

        if not self._is_process_running(pid):
            print(f"Process {pid} is not running")
            self._remove_pid_file(ui_name, port)
            return True

        print(f"Stopping {ui_name} (PID: {pid})...")

        try:
            # Try graceful shutdown first
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)

            if self._is_process_running(pid):
                # Force kill if still running
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                time.sleep(1)

            if not self._is_process_running(pid):
                print(f"Stopped {ui_name}")
                self._remove_pid_file(ui_name, port)
                return True
            else:
                print(f"Failed to stop {ui_name}")
                return False

        except OSError as e:
            print(f"Error stopping process: {e}")
            return False

    def status(self, ui_name: str, port: int = 8501):
        """Check the status of a UI process."""
        pid = self._read_pid_file(ui_name, port)

        if not pid:
            print(f"{ui_name}: Not managed (no PID file)")
            return False

        if self._is_process_running(pid):
            print(f"{ui_name}: Running (PID: {pid}, Port: {port})")
            return True
        else:
            print(f"{ui_name}: Stopped (stale PID file: {pid})")
            self._remove_pid_file(ui_name, port)
            return False

    def list_running(self):
        """List all managed UI processes."""
        ui_names = ["command_builder", "evaluation_viewer", "inference", "preprocessing_viewer", "resource_monitor"]

        running = []
        for ui_name in ui_names:
            # Check common ports
            for port in [8501, 8502, 8503, 8504, 8505]:
                if self.status(ui_name, port):
                    running.append((ui_name, port))

        if running:
            print("Running UI processes:")
            for ui_name, port in running:
                print(f"  - {ui_name}: port {port}")
        else:
            print("No managed UI processes running")

    def view_logs(self, ui_name: str, port: int = 8501, lines: int = 50, follow: bool = False):
        """View logs for a UI process."""
        stdout_log, stderr_log = self._get_log_files(ui_name, port)

        if not stdout_log.exists() and not stderr_log.exists():
            print(f"No log files found for {ui_name} on port {port}")
            print(f"Expected locations: {stdout_log} and {stderr_log}")
            return

        print(f"=== Logs for {ui_name} on port {port} ===")

        # Show stdout log
        if stdout_log.exists():
            print(f"\n--- STDOUT ({stdout_log}) ---")
            try:
                if follow:
                    # For follow mode, show last N lines and keep watching
                    with open(stdout_log) as f:
                        lines_content = f.readlines()
                        for line in lines_content[-lines:]:
                            print(line.rstrip())
                        print("\n--- Following stdout log (Ctrl+C to stop) ---")
                        f.seek(0, 2)  # Seek to end
                        while True:
                            line = f.readline()
                            if line:
                                print(line.rstrip())
                            time.sleep(0.1)
                else:
                    # Show last N lines
                    result = subprocess.run(["tail", f"-{lines}", str(stdout_log)], capture_output=True, text=True)
                    if result.returncode == 0:
                        print(result.stdout.rstrip())
                    else:
                        print(f"Error reading stdout log: {result.stderr}")
            except KeyboardInterrupt:
                print("\nStopped following logs")
            except Exception as e:
                print(f"Error reading stdout log: {e}")

        # Show stderr log
        if stderr_log.exists():
            print(f"\n--- STDERR ({stderr_log}) ---")
            try:
                if follow and not stdout_log.exists():
                    # Only follow stderr if stdout doesn't exist
                    with open(stderr_log) as f:
                        lines_content = f.readlines()
                        for line in lines_content[-lines:]:
                            print(line.rstrip())
                        print("\n--- Following stderr log (Ctrl+C to stop) ---")
                        f.seek(0, 2)  # Seek to end
                        while True:
                            line = f.readline()
                            if line:
                                print(line.rstrip())
                            time.sleep(0.1)
                else:
                    # Show last N lines
                    result = subprocess.run(["tail", f"-{lines}", str(stderr_log)], capture_output=True, text=True)
                    if result.returncode == 0:
                        print(result.stderr.rstrip())
                    else:
                        print(f"Error reading stderr log: {result.stderr}")
            except KeyboardInterrupt:
                print("\nStopped following logs")
            except Exception as e:
                print(f"Error reading stderr log: {e}")

    def clear_logs(self, ui_name: str, port: int = 8501):
        """Clear log files for a UI process."""
        stdout_log, stderr_log = self._get_log_files(ui_name, port)

        cleared = []
        if stdout_log.exists():
            stdout_log.unlink()
            cleared.append(str(stdout_log))

        if stderr_log.exists():
            stderr_log.unlink()
            cleared.append(str(stderr_log))

        if cleared:
            print(f"Cleared log files: {', '.join(cleared)}")
        else:
            print(f"No log files found for {ui_name} on port {port}")

    def stop_all(self):
        """Stop all managed UI processes."""
        ui_names = ["command_builder", "evaluation_viewer", "inference", "preprocessing_viewer", "resource_monitor"]

        stopped = []
        for ui_name in ui_names:
            for port in [8501, 8502, 8503, 8504, 8505]:
                if self.stop(ui_name, port):
                    stopped.append((ui_name, port))

        if stopped:
            print("Stopped processes:")
            for ui_name, port in stopped:
                print(f"  - {ui_name}: port {port}")
        else:
            print("No processes to stop")


def main():
    parser = argparse.ArgumentParser(description="Streamlit Process Manager")
    parser.add_argument("action", choices=["start", "stop", "status", "list", "stop-all", "logs", "clear-logs"], help="Action to perform")
    parser.add_argument("ui_name", nargs="?", help="UI name (for start/stop/status/logs/clear-logs)")
    parser.add_argument("--port", type=int, default=8501, help="Port number (default: 8501)")
    parser.add_argument("--foreground", action="store_true", help="Run in foreground (only for start action)")
    parser.add_argument("--no-logging", action="store_true", help="Disable logging to files (only for start action)")
    parser.add_argument("--lines", type=int, default=50, help="Number of log lines to show (default: 50)")
    parser.add_argument("--follow", action="store_true", help="Follow log files in real-time (only for logs action)")

    args = parser.parse_args()

    manager = StreamlitProcessManager()

    if args.action == "start":
        if not args.ui_name:
            parser.error("UI name required for start action")
        background = not args.foreground
        enable_logging = not args.no_logging
        manager.start(args.ui_name, args.port, background, enable_logging)

    elif args.action == "stop":
        if not args.ui_name:
            parser.error("UI name required for stop action")
        manager.stop(args.ui_name, args.port)

    elif args.action == "status":
        if not args.ui_name:
            parser.error("UI name required for status action")
        manager.status(args.ui_name, args.port)

    elif args.action == "logs":
        if not args.ui_name:
            parser.error("UI name required for logs action")
        manager.view_logs(args.ui_name, args.port, args.lines, args.follow)

    elif args.action == "clear-logs":
        if not args.ui_name:
            parser.error("UI name required for clear-logs action")
        manager.clear_logs(args.ui_name, args.port)

    elif args.action == "list":
        manager.list_running()

    elif args.action == "stop-all":
        manager.stop_all()


if __name__ == "__main__":
    main()
