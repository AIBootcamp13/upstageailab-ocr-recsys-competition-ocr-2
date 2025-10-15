from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

from ui.utils.command import CommandBuilder, CommandValidator
from ui.utils.ui_generator import generate_ui_from_schema
from ui.utils.ui_validator import validate_inputs

from ..models.command import CommandPageData
from ..state import CommandBuilderState, CommandType
from .command_preview import render_command_preview
from .execution import render_execution_panel

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "command_builder_predict.yaml"


def render_predict_page(state: CommandBuilderState, command_builder: CommandBuilder) -> None:
    page: CommandPageData = state.get_page(CommandType.PREDICT)

    # Add sidebar for JSON file selection
    with st.sidebar:
        st.markdown("### 📁 Submission File Selector")
        # Use separate session state variable for auto-selected path
        auto_selected_path = st.session_state.get("predict_auto_selected_path", "")
        selected_json_path = st.text_input(
            "JSON File Path",
            value=auto_selected_path,  # Use auto-selected path as default
            help="Path to the prediction JSON file (auto-filled after predict run)",
            key="predict_json_path",
        )
        uploaded_json = st.file_uploader(
            "Or upload JSON file", type=["json"], help="Upload a prediction JSON file directly", key="predict_json_upload"
        )
        # Handle uploaded file
        json_file: Path | None = None
        if uploaded_json is not None:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                data = json.load(uploaded_json)
                json.dump(data, f)
                json_file = Path(f.name)
            st.info(f"📁 Using uploaded file: `{uploaded_json.name}`")
        else:
            json_file = Path(selected_json_path) if selected_json_path else None

    st.markdown("### 🔮 Prediction configuration")

    schema_result = generate_ui_from_schema(str(SCHEMA_PATH))
    values = dict(schema_result.values)
    errors = validate_inputs(values, str(SCHEMA_PATH))

    page.generated.values = values
    page.generated.overrides = schema_result.overrides
    page.generated.constant_overrides = list(schema_result.constant_overrides)
    page.generated.errors = errors

    if st.button("Generate command", type="primary", key="command_builder_predict_generate"):
        if errors:
            st.error("Resolve validation errors before generating the command.")
        else:
            command_text = command_builder.build_command_from_overrides(
                script="predict.py",
                overrides=schema_result.overrides,
                constant_overrides=schema_result.constant_overrides,
            )
            page.generated.generated = command_text
            page.generated.edited = command_text
            validator = CommandValidator()
            validation_ok, validation_error = validator.validate_command(command_text)
            page.generated.validation_error = None if validation_ok else validation_error
            st.session_state[f"command_builder_{CommandType.PREDICT.value}_command_text"] = command_text
            if validation_ok:
                st.success("Command is valid and ready to run.")
            else:
                st.warning(f"Command generated but validation raised an issue: {validation_error}")

    if errors:
        for message in errors:
            st.error(message)
    elif page.generated.validation_error:
        st.error(page.generated.validation_error)
    elif page.generated.generated:
        st.success("Command is valid")

    edited_command = render_command_preview(CommandType.PREDICT, page)
    render_execution_panel(
        command_builder=command_builder,
        command=edited_command,
        page=page,
        command_type=CommandType.PREDICT,
        exp_name=values.get("exp_name", ""),
    )

    # Submission Export Section - Always available
    render_submission_export_panel(page, values, json_file)

    state.persist()


def find_latest_submission_json(exp_name: str) -> Path | None:
    """Find the most recent submission JSON file for a given experiment."""
    outputs_dir = Path("outputs") / exp_name / "submissions"
    if not outputs_dir.exists():
        return None
    json_files = sorted(outputs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files[0] if json_files else None


def render_submission_export_panel(page: CommandPageData, values: dict, json_file: Path | None = None) -> None:
    """Render the submission CSV export panel after successful prediction."""
    st.markdown("---")
    st.markdown("### 📤 Export Submission (CSV)")

    # Add experiment selector for outputs directory
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        exp_dirs = [d.name for d in outputs_dir.iterdir() if d.is_dir() and (d / "submissions").exists()]
        exp_dirs.sort(reverse=True)  # Most recent first
    else:
        exp_dirs = []

    if exp_dirs:
        selected_exp = st.selectbox(
            "Select Experiment",
            options=[""] + exp_dirs,
            format_func=lambda x: "Choose experiment..." if x == "" else x,
            help="Select an experiment to browse its submission files",
            key="export_experiment_selector",
        )

        if selected_exp:
            submissions_dir = outputs_dir / selected_exp / "submissions"
            json_files = list(submissions_dir.glob("*.json"))
            json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            if json_files:
                json_options = [""] + [str(f) for f in json_files]
                selected_json_option = st.selectbox(
                    "Select JSON File",
                    options=json_options,
                    format_func=lambda x: "Choose file..." if x == "" else Path(x).name,
                    help="Select a specific JSON file to convert",
                    key="export_json_selector",
                )

                if selected_json_option:
                    json_file = Path(selected_json_option)

    # Use provided json_file, or fall back to finding latest
    selected_json: Path | None = None
    if json_file and json_file.exists():
        selected_json = json_file
        st.success(f"✅ Using selected file: `{selected_json}`")
    else:
        exp_name = values.get("exp_name", "")
        selected_json = find_latest_submission_json(exp_name)
        if selected_json:
            st.info(f"ℹ️ Using latest file: `{selected_json}` (no file selected)")
        else:
            st.warning("⚠️ No JSON file available. Select an experiment above or upload a file in the sidebar.")
            return

    assert selected_json is not None  # At this point selected_json is guaranteed to be not None

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        output_csv_path = st.text_input(
            "Output CSV filename",
            value="submission.csv",
            help="Name for the generated CSV submission file",
            key="submission_csv_output_path",
        )
    with col2:
        include_confidence = st.checkbox(
            "Include confidence",
            value=False,
            help="Include confidence scores in CSV for analysis (not for competition submission)",
            key="include_confidence_csv",
        )
    with col3:
        convert_clicked = st.button("🔄 Convert to CSV", type="secondary", key="convert_submission_csv")

    if convert_clicked:
        if not output_csv_path.strip():
            st.error("❌ Please provide a valid output filename")
            return

        with st.spinner("Converting JSON to CSV..."):
            try:
                cmd = [
                    "uv",
                    "run",
                    "python",
                    "ocr/utils/convert_submission.py",
                    "--json_path",
                    str(selected_json),
                    "--output_path",
                    output_csv_path,
                ]
                if include_confidence:
                    cmd.append("--include_confidence")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=60,
                )
                st.success(f"✅ Successfully converted to `{output_csv_path}`")
                if result.stdout:
                    with st.expander("📄 Conversion output"):
                        st.code(result.stdout, language="text")

                # Show file info
                csv_path = Path(output_csv_path)
                if csv_path.exists():
                    file_size = csv_path.stat().st_size
                    st.info(f"📊 File created: {csv_path.absolute()} ({file_size:,} bytes)")

            except subprocess.TimeoutExpired:
                st.error("❌ Conversion timed out after 60 seconds")
            except subprocess.CalledProcessError as exc:
                st.error(f"❌ Conversion failed with return code {exc.returncode}")
                if exc.stderr:
                    with st.expander("⚠️ Error details"):
                        st.code(exc.stderr, language="text")
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Unexpected error: {exc}")

    # Show usage instructions
    with st.expander("ℹ️ How to use submission files"):
        json_path_display = selected_json or "outputs/{exp_name}/submissions/{timestamp}.json"
        st.markdown(
            f"""
            **Submission Workflow:**

            1. **Run Prediction**: Click "Run command" above to generate predictions
            2. **Convert to CSV**: Click "Convert to CSV" to create the submission file
            3. **Submit**: Upload the `submission.csv` file to the competition platform

            **Manual Conversion:**
            ```bash
            uv run python ocr/utils/convert_submission.py \\
              --json_path {json_path_display} \\
              --output_path submission.csv
            ```

            **CSV Format:**
            ```
            filename,polygons
            image001.jpg,123 456 789 012|234 567 890 123
            image002.jpg,111 222 333 444|555 666 777 888
            ```

            Each row contains:
            - `filename`: Image filename
            - `polygons`: Pipe-separated (`|`) polygons with space-separated coordinates

            📚 See: `docs/generating-submissions.md` for complete documentation
            """
        )
