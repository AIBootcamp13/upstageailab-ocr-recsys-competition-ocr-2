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
