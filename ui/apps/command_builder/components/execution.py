from __future__ import annotations

import time
from collections import deque

import streamlit as st

from ui.utils.command_builder import CommandBuilder

from ..models.command import CommandPageData
from ..services.formatting import format_command_output
from ..state import CommandType


def render_execution_panel(
    *,
    command_builder: CommandBuilder,
    command: str,
    page: CommandPageData,
    command_type: CommandType,
) -> None:
    st.markdown("### ⚙️ Execute command")

    if not command.strip():
        st.info("Generate a command above to enable execution.")
        return

    is_valid, validation_error = command_builder.validate_command(command)
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
            return_code, stdout, stderr = command_builder.execute_command_streaming(command, progress_callback=progress_callback)
            duration = time.time() - start_time
            page.execution.mark_finished(return_code, duration, stdout, stderr)
            status_placeholder.empty()
            if return_code == 0:
                status_placeholder.success(f"✅ Command completed successfully in {duration:.1f}s")
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
