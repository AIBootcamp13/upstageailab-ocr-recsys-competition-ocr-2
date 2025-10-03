"""
Command Executor

Handles the execution of CLI commands with process management and streaming output.
"""

import os
import shlex
import signal
import subprocess
import time
from collections.abc import Callable


class CommandExecutor:
    """Execute CLI commands with process management."""

    def execute_command_streaming(
        self,
        command: str,
        cwd: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[int, str, str]:
        """Execute a command with streaming output and process group management.

        Args:
            command: Command string to execute.
            cwd: Working directory for execution.
            progress_callback: Optional callback function to handle output lines in real-time.

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        if cwd is None:
            from pathlib import Path

            cwd = str(Path(__file__).resolve().parent.parent.parent.parent)  # Project root

        try:
            # Use Popen with process group for better cleanup control
            # Use shell-aware splitting to preserve quoted arguments
            process = subprocess.Popen(
                shlex.split(command),
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                preexec_fn=os.setsid,  # Create new process group
            )

            stdout_lines: list[str] = []
            stderr_lines: list[str] = []

            while process.poll() is None:
                if process.stdout and (line := process.stdout.readline()):
                    stripped = line.rstrip()
                    stdout_lines.append(stripped)
                    if progress_callback:
                        progress_callback(f"OUT: {stripped}")

                if process.stderr and (line := process.stderr.readline()):
                    stripped = line.rstrip()
                    stderr_lines.append(stripped)
                    if progress_callback:
                        progress_callback(f"ERR: {line}")

                time.sleep(0.1)

            remaining_stdout, remaining_stderr = process.communicate()
            if remaining_stdout:
                for line in remaining_stdout.splitlines():
                    stdout_lines.append(line)
                    if progress_callback:
                        progress_callback(f"OUT: {line}")
            if remaining_stderr:
                for line in remaining_stderr.splitlines():
                    stderr_lines.append(line)
                    if progress_callback:
                        progress_callback(f"ERR: {line}")

            return process.returncode, "\n".join(stdout_lines), "\n".join(stderr_lines)

        except FileNotFoundError:
            return (
                -1,
                "",
                "Execution error: 'uv' command not found. Is it installed and in your PATH?",
            )
        except Exception as e:
            return -1, "", f"An unexpected execution error occurred: {e}"

    def terminate_process_group(self, process: subprocess.Popen) -> bool:
        """Terminate a process group to ensure all child processes are killed.

        Args:
            process: The Popen process object.

        Returns:
            True if termination was successful, False otherwise.
        """
        try:
            if process.poll() is None:  # Process is still running
                # Send SIGTERM to the entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)

                # Wait a bit for graceful shutdown
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    # If still running, force kill
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
            return True
        except OSError:
            # Process might already be dead
            return False
