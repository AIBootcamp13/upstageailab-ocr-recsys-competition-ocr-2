"""
OCR Training Command Builder - Streamlit UI

A Streamlit application for building and executing OCR training, testing,
and prediction commands through an intuitive UI.
"""

import logging
import shlex
import sys
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st

from ui.utils.command_builder import CommandBuilder
from ui.utils.config_parser import ConfigParser
from ui.utils.ui_generator import generate_ui_from_schema
from ui.utils.ui_validator import validate_inputs


def main():
    """Main Streamlit application."""
    # Basic logging for UI-side diagnostics
    logging.basicConfig(level=logging.ERROR)
    st.set_page_config(page_title="OCR Command Builder", page_icon="🔍", layout="wide")

    st.title("🔍 OCR Training Command Builder")
    st.markdown("Build and execute training, testing, and prediction commands with ease")

    # Initialize utilities
    config_parser = ConfigParser()
    command_builder = CommandBuilder()

    # Sidebar for command type selection
    st.sidebar.header("Command Type")
    command_type = st.sidebar.selectbox(
        "Select Command",
        ["train", "test", "predict"],
        help="Choose the type of command to build",
    )

    # Main content area
    if command_type == "train":
        render_train_interface(config_parser, command_builder)
    elif command_type == "test":
        render_test_interface(config_parser, command_builder)
    elif command_type == "predict":
        render_predict_interface(config_parser, command_builder)


def render_train_interface(config_parser: ConfigParser, command_builder: CommandBuilder):
    """Render the training command interface using declarative schema."""
    st.header("🚀 Training Configuration")

    schema_path = Path(__file__).parent / "schemas" / "command_builder_train.yaml"

    # 1) Generate UI from schema
    gen = generate_ui_from_schema(str(schema_path))  # 2) Validate inputs
    errors = validate_inputs(gen.values, str(schema_path))

    # Option to avoid name collisions across models
    append_model_to_name = st.checkbox(
        "Append encoder to experiment name",
        value=True,
        help="Helps segregate outputs per model (e.g., ocr_training-resnet18)",
        key="train_append_model_suffix",
    )
    if gen.values.get("resume_training") and append_model_to_name:
        st.warning(
            "Resuming training disables experiment name suffixing to keep directories "
            "consistent with the original run. If multiple models share the same exp_name, "
            "outputs may collide. Consider adjusting exp_name manually."
        )

    if "train_generated_cmd" not in st.session_state:
        st.session_state["train_generated_cmd"] = ""

    if "train_command_input" not in st.session_state:
        st.session_state["train_command_input"] = ""

    col_g1, col_g2 = st.columns([1, 3])
    with col_g1:
        if st.button("Generate", help="Generate a command from current settings"):
            overrides = maybe_suffix_exp_name(list(gen.overrides), gen.values, append_model_to_name)
            st.session_state["train_generated_cmd"] = command_builder.build_command_from_overrides(
                script="train.py",
                overrides=overrides,
                constant_overrides=gen.constant_overrides,
            )
            st.session_state["train_overrides"] = overrides
            st.session_state["train_constant_overrides"] = list(gen.constant_overrides)
            # Update the editable text area state so the UI reflects immediately
            st.session_state["train_command_input"] = st.session_state["train_generated_cmd"]

    st.subheader("Generated Command")
    edited_command = st.text_area(
        "Command (editable)",
        height=150,
        help="Click Generate to update based on settings. You can still edit it before executing.",
        key="train_command_input",
    )

    with st.expander("🔧 Overrides Preview"):
        const_ov = st.session_state.get("train_constant_overrides", gen.constant_overrides)
        ovr = st.session_state.get("train_overrides", gen.overrides)
        st.text("Constant overrides:\n" + "\n".join(const_ov or []))
        st.text("\nOverrides:\n" + "\n".join(ovr or []))
        if st.session_state.get("train_generated_cmd"):
            st.code(
                pretty_format_command(st.session_state["train_generated_cmd"]),
                language="bash",
            )

    if errors:
        for err in errors:
            st.error(err)

    # Guard against None from text_area (shouldn't happen, but be safe)
    edited_command = edited_command or st.session_state.get("train_generated_cmd", "")
    is_valid_cmd, error_msg = command_builder.validate_command(edited_command)
    if not is_valid_cmd:
        st.error(f"Command validation failed: {error_msg}")
    elif not errors:
        st.success("Command is valid")
        if st.button("🚀 Execute Training", type="primary"):
            execute_command(command_builder, edited_command)


def maybe_suffix_exp_name(overrides: list[str], values: dict[str, Any], append_suffix: bool) -> list[str]:
    """Optionally append encoder name to exp_name in overrides to avoid collisions.

    Only applies when append_suffix is True, encoder is set, and not resuming.
    """
    if not append_suffix or not values.get("encoder") or values.get("resume_training"):
        return overrides
    encoder = values["encoder"]
    for i, ov in enumerate(overrides):
        if ov.startswith("exp_name="):
            base_name = ov.split("=", 1)[1]
            overrides[i] = f"exp_name={base_name}-{encoder}"
            break
    return overrides


def render_test_interface(config_parser: ConfigParser, command_builder: CommandBuilder):
    """Render the testing command interface (schema-driven)."""
    st.header("🧪 Testing Configuration")

    schema_path = Path(__file__).parent / "schemas" / "command_builder_test.yaml"
    gen = generate_ui_from_schema(str(schema_path))
    errors = validate_inputs(gen.values, str(schema_path))

    if "test_generated_cmd" not in st.session_state:
        st.session_state["test_generated_cmd"] = ""

    if "test_command" not in st.session_state:
        st.session_state["test_command"] = ""

    col_g1, col_g2 = st.columns([1, 3])
    with col_g1:
        if st.button("Generate", key="test_generate"):
            st.session_state["test_generated_cmd"] = command_builder.build_command_from_overrides(
                script="test.py",
                overrides=gen.overrides,
                constant_overrides=gen.constant_overrides,
            )
            st.session_state["test_command"] = st.session_state["test_generated_cmd"]
            st.session_state["test_overrides"] = list(gen.overrides)
            st.session_state["test_constant_overrides"] = list(gen.constant_overrides)

    st.subheader("Generated Command")
    edited_command = st.text_area(
        "Command (editable)",
        height=150,
        help="Click Generate to update based on settings.",
        key="test_command",
    )

    with st.expander("🔧 Overrides Preview"):
        const_ov = st.session_state.get("test_constant_overrides", gen.constant_overrides)
        ovr = st.session_state.get("test_overrides", gen.overrides)
        st.text("Constant overrides:\n" + "\n".join(const_ov or []))
        st.text("\nOverrides:\n" + "\n".join(ovr or []))
        if st.session_state.get("test_generated_cmd"):
            st.code(
                pretty_format_command(st.session_state["test_generated_cmd"]),
                language="bash",
            )

    edited_command = edited_command or st.session_state.get("test_generated_cmd", "")
    if errors:
        for err in errors:
            st.error(err)

    is_valid, error_msg = command_builder.validate_command(edited_command)
    if not is_valid:
        st.error(f"Command validation failed: {error_msg}")
    elif not errors:
        st.success("Command is valid")
        if st.button("🧪 Execute Testing", type="primary"):
            execute_command(command_builder, edited_command)


def render_predict_interface(config_parser: ConfigParser, command_builder: CommandBuilder):
    """Render the prediction command interface (schema-driven)."""
    st.header("🔮 Prediction Configuration")

    schema_path = Path(__file__).parent / "schemas" / "command_builder_predict.yaml"
    gen = generate_ui_from_schema(str(schema_path))
    errors = validate_inputs(gen.values, str(schema_path))

    if "predict_generated_cmd" not in st.session_state:
        st.session_state["predict_generated_cmd"] = ""

    if "predict_command" not in st.session_state:
        st.session_state["predict_command"] = ""

    col_g1, col_g2 = st.columns([1, 3])
    with col_g1:
        if st.button("Generate", key="predict_generate"):
            st.session_state["predict_generated_cmd"] = command_builder.build_command_from_overrides(
                script="predict.py",
                overrides=gen.overrides,
                constant_overrides=gen.constant_overrides,
            )
            st.session_state["predict_command"] = st.session_state["predict_generated_cmd"]
            st.session_state["predict_overrides"] = list(gen.overrides)
            st.session_state["predict_constant_overrides"] = list(gen.constant_overrides)

    st.subheader("Generated Command")
    edited_command = st.text_area(
        "Command (editable)",
        height=150,
        help="Click Generate to update based on settings.",
        key="predict_command",
    )

    with st.expander("🔧 Overrides Preview"):
        const_ov = st.session_state.get("predict_constant_overrides", gen.constant_overrides)
        ovr = st.session_state.get("predict_overrides", gen.overrides)
        st.text("Constant overrides:\n" + "\n".join(const_ov or []))
        st.text("\nOverrides:\n" + "\n".join(ovr or []))
        if st.session_state.get("predict_generated_cmd"):
            st.code(
                pretty_format_command(st.session_state["predict_generated_cmd"]),
                language="bash",
            )

    edited_command = edited_command or st.session_state.get("predict_generated_cmd", "")
    if errors:
        for err in errors:
            st.error(err)

    is_valid, error_msg = command_builder.validate_command(edited_command)
    if not is_valid:
        st.error(f"Command validation failed: {error_msg}")
    elif not errors:
        st.success("Command is valid")
        if st.button("🔮 Execute Prediction", type="primary"):
            execute_command(command_builder, edited_command)


def execute_command(command_builder: CommandBuilder, command: str):
    """Execute a command and display results with real-time progress and safety mechanisms."""
    import time
    from collections import deque

    # Create placeholders for dynamic updates
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    output_container = st.container()
    error_container = st.container()

    status_placeholder.info("🚀 Starting command execution...")

    # Create a deque to store recent output lines for display
    output_lines = deque(maxlen=100)  # Keep last 100 lines for safety
    output_display = output_container.empty()
    error_display = error_container.empty()

    start_time = time.time()
    execution_failed = False

    def progress_callback(line: str):
        """Callback to handle real-time output with error handling."""
        try:
            output_lines.append(line)
            # Update display with recent output
            recent_output = "\n".join(list(output_lines)[-25:])  # Show last 25 lines
            formatted_output = format_command_output(recent_output)
            output_display.code(f"🖥️ Live Output (Last 25 lines):\n{formatted_output}", language="text")

            # Check for common error patterns and warn user
            if any(error_word in line.lower() for error_word in ["error", "exception", "failed", "traceback"]):
                error_display.warning(f"⚠️ Potential issue detected: {line[:100]}...")

        except (ValueError, TypeError) as e:
            logging.error(f"Output display error: {e}\n{traceback.format_exc()}")
            error_display.error(f"⚠️ Output display error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error in progress_callback: {e}\n{traceback.format_exc()}")
            error_display.error(f"⚠️ Unexpected error in output display: {e}")

    try:
        # Use streaming execution with timeout safety
        return_code, stdout, stderr = command_builder.execute_command_streaming(command, progress_callback=progress_callback)

        execution_time = time.time() - start_time

        # Clear progress indicators
        status_placeholder.empty()
        progress_placeholder.empty()

        if return_code == 0:
            status_placeholder.success(f"✅ Command completed successfully in {execution_time:.1f}s!")
        else:
            status_placeholder.error(f"❌ Command failed with return code {return_code} after {execution_time:.1f}s")
            execution_failed = True

        # Always display complete outputs in organized sections
        with output_container:
            if stdout.strip():
                with st.expander("📄 Complete Standard Output", expanded=execution_failed):
                    formatted_output = format_command_output(stdout)
                    st.code(formatted_output, language="text")

            if stderr.strip():
                with st.expander("⚠️ Complete Standard Error", expanded=True):
                    formatted_error = format_command_output(stderr)
                    st.code(formatted_error, language="text")

    except Exception as e:
        execution_time = time.time() - start_time
        status_placeholder.error(f"💥 Execution error after {execution_time:.1f}s: {e}")
        logging.error(f"Execution error: {e}\n{traceback.format_exc()}")
        error_display.error(f"Full error: {str(e)}")
        execution_failed = True

    finally:
        # Clean up displays but keep final output visible
        try:
            if not execution_failed:
                output_display.empty()
                error_display.empty()
        except Exception as cleanup_exception:
            st.error(f"⚠️ Cleanup error: {cleanup_exception}")

        # Ensure we always show some indication of completion
        if execution_failed:
            st.error("❌ Command execution encountered issues. Check the output above for details.")
        else:
            st.success("✅ Command execution completed. Check outputs above for results.")


def format_command_output(output: str) -> str:
    """Format command output for better readability."""
    if not output.strip():
        return output

    lines = output.split("\n")
    formatted_lines = []

    for line in lines:
        # Add visual indicators for common patterns
        if "epoch" in line.lower() and "loss" in line.lower():
            formatted_lines.append(f"📊 {line}")
        elif "error" in line.lower() or "exception" in line.lower():
            formatted_lines.append(f"❌ {line}")
        elif "warning" in line.lower():
            formatted_lines.append(f"⚠️ {line}")
        elif "success" in line.lower() or "completed" in line.lower():
            formatted_lines.append(f"✅ {line}")
        elif line.strip().startswith("[") and "]" in line:
            # Progress bars or bracketed output
            formatted_lines.append(f"🔄 {line}")
        else:
            formatted_lines.append(line)

    return "\n".join(formatted_lines)


def pretty_format_command(cmd: str) -> str:
    """Pretty-format a long uv/python command into multiple lines with trailing backslashes.

    This is for display only; copy-paste back to a shell should still work.
    """
    if not cmd:
        return cmd
    fallback_note = None
    try:
        parts = shlex.split(cmd)
    except Exception as e:
        # Fallback: naive split
        parts = cmd.split()
        fallback_note = f"# Note: best-effort formatting due to quoting error: {e}"

    # Identify main script boundary and config path
    first_break_idx = min(4, len(parts))
    with suppress(ValueError):
        cfg_idx = parts.index("--config-path")
        # include the path value as part of the first group
        first_break_idx = max(first_break_idx, cfg_idx + 2)

    head = parts[:first_break_idx]
    tail = parts[first_break_idx:]

    lines = []
    if head:
        head_line = " ".join(head)
        lines.append(head_line + (" \\" if tail else ""))
    if tail:
        # Put each remaining token on its own line for readability
        for i, tok in enumerate(tail):
            is_last = i == len(tail) - 1
            cont = "" if is_last else " \\"
            lines.append(f"  {tok}{cont}")
    if fallback_note:
        lines.insert(0, fallback_note)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
