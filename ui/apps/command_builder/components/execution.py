from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import streamlit as st

from ui.utils.command import CommandBuilder, CommandExecutor, CommandValidator

from ..models.command import CommandPageData
from ..services.formatting import format_command_output
from ..state import CommandType


def find_latest_submission_json(exp_name: str) -> Path | None:
    """Find the most recent submission JSON file for a given experiment."""
    outputs_dir = Path("outputs") / exp_name / "submissions"
    if not outputs_dir.exists():
        return None
    json_files = sorted(outputs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files[0] if json_files else None


def render_execution_panel(
    *,
    command_builder: CommandBuilder,
    command: str,
    page: CommandPageData,
    command_type: CommandType,
    json_file: Path | None = None,  # New param for auto-selection
    exp_name: str | None = None,  # New param for exp_name
) -> None:
    st.markdown("### ⚙️ Execute command")

    if not command.strip():
        st.info("Generate a command above to enable execution.")
        return

    validator = CommandValidator()
    is_valid, validation_error = validator.validate_command(command)
    if not is_valid:
        st.error(f"Command validation failed: {validation_error}")
        return

    status_placeholder = st.empty()
    live_output_placeholder = st.empty()
    error_placeholder = st.empty()

    if st.button("Run command", type="primary", key=f"command_builder_execute_{command_type.value}"):
        start_time = time.time()
        output_lines: deque[str] = deque(maxlen=100)
        page.execution.mark_running()
        status_placeholder.info("🚀 Starting command execution...")

        def progress_callback(line: str) -> None:
            output_lines.append(line)
            recent_output = "\n".join(list(output_lines)[-25:])
            formatted_output = format_command_output(recent_output)
            live_output_placeholder.code(f"🖥️ Live Output (Last 25 lines):\n{formatted_output}", language="text")
            if any(keyword in line.lower() for keyword in ["error", "exception", "failed", "traceback"]):
                error_placeholder.warning(f"⚠️ Potential issue detected: {line[:120]}...")

        try:
            executor = CommandExecutor()
            return_code, stdout, stderr = executor.execute_command_streaming(command, progress_callback=progress_callback)
            duration = time.time() - start_time
            page.execution.mark_finished(return_code, duration, stdout, stderr)
            status_placeholder.empty()
            if return_code == 0:
                status_placeholder.success(f"✅ Command completed successfully in {duration:.1f}s")
                # Auto-set JSON path for predict commands
                if command_type == CommandType.PREDICT and exp_name:
                    latest_json = find_latest_submission_json(exp_name)
                    if latest_json:
                        st.session_state["predict_auto_selected_path"] = str(latest_json)
                        st.rerun()  # Refresh to show auto-selected file
            else:
                status_placeholder.error(f"❌ Command failed with return code {return_code} after {duration:.1f}s")
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - start_time
            page.execution.status = "error"
            page.execution.duration = duration
            page.execution.stdout = ""
            page.execution.stderr = str(exc)
            status_placeholder.error(f"💥 Execution error after {duration:.1f}s: {exc}")
        finally:
            live_output_placeholder.empty()

    if page.execution.status in {"success", "error"}:
        if page.execution.stdout.strip():
            with st.expander("📄 Complete Standard Output", expanded=page.execution.status == "error"):
                st.code(format_command_output(page.execution.stdout), language="text")
        if page.execution.stderr.strip():
            with st.expander("⚠️ Complete Standard Error", expanded=True):
                st.code(format_command_output(page.execution.stderr), language="text")
