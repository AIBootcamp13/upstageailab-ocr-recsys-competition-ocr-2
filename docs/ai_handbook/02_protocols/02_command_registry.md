# 02_command_registry.md
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=automation,commands -->

# **Command Registry for AI Agents**

This document lists approved, safe-to-run scripts for autonomous execution. All commands use uv run to ensure the correct project environment is used.

## **1. Validation & Smoke Tests**

### **Validate a Hydra Configuration**

* **Purpose:** Checks a configuration file for syntax errors and ensures it can be resolved by Hydra without launching a full run.
* **Command:** `uv run python scripts/agent_tools/validate_config.py --config-name <name>`
* **Example:** `uv run python scripts/agent_tools/validate_config.py --config-name train`
* **Expected Output:** A success message or a detailed error trace.
* **Resources:** Low (CPU, < 5s)

### **Run a Model Smoke Test**

* **Purpose:** Runs a single training and validation step to quickly verify that the model, data, and pipeline are wired correctly.
* **Command:** `uv run python runners/train.py --config-name train trainer.fast_dev_run=true`
* **Expected Output:** PyTorch Lightning summary of a successful single-batch run.
* **Resources:** Medium (Requires GPU, < 60s)

## **2. Data & Preprocessing**

### **Generate Offline Preprocessing Samples**

* **Purpose:** Creates visual examples of the Microsoft Lens-style preprocessing pipeline on a few sample images.
* **Command:** `uv run python scripts/agent_tools/generate_samples.py --num-samples 5`
* **Expected Output:** A new directory outputs/samples with original, processed, and comparison images.
* **Resources:** Low (CPU, < 15s)

## **3. Querying Information**

### **List Available Checkpoints**

* **Purpose:** Lists all trained model checkpoints available in the `outputs/` directory.
* **Command:** `uv run python scripts/agent_tools/list_checkpoints.py`
* **Expected Output:** A formatted list of checkpoint paths and their creation dates.
* **Resources:** Low (CPU, < 2s)

## **4. Data Diagnostics**

### **EXIF Orientation & Polygon Audit**

* **Purpose:** Counts EXIF orientations across the dataset and reports polygon retention after transforms via selectable modes.
* **Command:** `uv run python tests/debug/data_analyzer.py --mode orientation|polygons|both [--limit N]`
* **Recommended Usage:**
	* `--mode orientation` for a quick EXIF histogram sanity check.
	* `--mode polygons --limit 50` to sample a few batches while iterating quickly.
	* `--mode both` before major data refactors to capture a complete baseline.
* **Expected Output:** Orientation histogram, polygon audit summary (including drop counts and example file ids).
* **Resources:** Low (CPU, < 15s)

## **5. Streamlit UI Launchers**

All Streamlit entrypoints are consolidated behind `run_ui.py`. Pass one of the commands below to launch the corresponding app without relying on deprecated monolithic scripts.

### **Evaluation Viewer**

* **Purpose:** Launch the modular evaluation results dashboard.
* **Command:** `uv run python run_ui.py evaluation_viewer`
* **Expected Output:** Streamlit UI served on the configured port (default 8501) showing comparison dashboards.
* **Resources:** Medium (CPU/GPU as needed, keep < 1m startup).

### **Inference Sandbox**

* **Purpose:** Run the interactive inference UI with modular engine back-end.
* **Command:** `uv run python run_ui.py inference`
* **Expected Output:** Streamlit UI for uploading images, running inference, and visualizing predictions.
* **Resources:** Medium (GPU optional; CPU inference available with lighter checkpoints).

### **Command Builder**

* **Purpose:** Generate hydra command strings using the modular workflow forms.
* **Command:** `uv run python run_ui.py command_builder`
* **Expected Output:** Streamlit UI with command presets and copy-to-clipboard helpers.
* **Resources:** Low (CPU, < 10s startup).

### **Resource Monitor**

* **Purpose:** Monitor training/inference processes in real time.
* **Command:** `uv run python run_ui.py resource_monitor`
* **Expected Output:** Streamlit UI with system metrics, experiment status, and queue overview.
* **Resources:** Low (CPU, < 10s startup).

> **Note:** The legacy `ui/test_viewer.py` entrypoint has been removed. Use the commands above for all future UI access.

## **6. Documentation & Context Utilities**

### **Doc Context Bundle Loader**

* **Purpose:** Prints the recommended documentation bundle for a given task (as defined in `docs/ai_handbook/index.json`).
* **Command:** `uv run python scripts/agent_tools/get_context.py --bundle <bundle-id>`
* **Example:** `uv run python scripts/agent_tools/get_context.py --bundle streamlit-maintenance`
* **Expected Output:** Titles, paths, priorities, and tag summaries for each doc in the bundle.
* **Resources:** Negligible.

### **Start Context Log**

* **Purpose:** Creates a timestamped JSONL log under `logs/agent_runs/` and prints its path for the current session.
* **Command:** `uv run python scripts/agent_tools/context_log.py start --label <short-label>`
* **Make Shortcut:** `make context-log-start LABEL="streamlit-maintenance"`
* **Expected Output:** Absolute path to the log file that subsequent logging calls should append to.
* **Resources:** Negligible.

### **Summarize Context Log**

* **Purpose:** Generates a Markdown summary from a structured context log using the LLM helper.
* **Command:** `uv run python scripts/agent_tools/context_log.py summarize --log-file <path>`
* **Make Shortcut:** `make context-log-summarize LOG=logs/agent_runs/<file>.jsonl`
* **Expected Output:** Markdown file stored in `docs/ai_handbook/04_experiments/` and a success message containing the path.
* **Resources:** Negligible beyond LLM call latency.

### **Validate Handbook Manifest**

* **Purpose:** Ensures the handbook manifest stays consistent (unique IDs, valid paths, and bundle/command integrity).
* **Command:** `uv run python scripts/agent_tools/validate_manifest.py`
* **Options:** `--allow-unbundled` downgrades warnings when intentionally leaving entries outside bundles.
* **Expected Output:** `Manifest check: PASS` with optional warnings, or a detailed list of errors before exiting with status 1.
* **Resources:** Negligible.

### **Strip AI Documentation Markers**

* **Purpose:** Removes `AI_DOCS` annotations from source files before sharing publicly (with an option to restore from snapshot).
* **Command:**
	* Dry run: `uv run python scripts/agent_tools/strip_doc_markers.py --dry-run`
	* Apply: `uv run python scripts/agent_tools/strip_doc_markers.py --apply`
	* Restore: `uv run python scripts/agent_tools/strip_doc_markers.py --restore`
* **Expected Output:** Lists files containing markers, and when `--apply` is used, stores a snapshot to `tmp/ai_docs_markers.json` for later restoration.
* **Resources:** Negligible.
