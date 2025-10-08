# **filename: docs/ai_handbook/03_references/03_utility_functions.md**

# **Reference: Utility Functions**

This document serves as a quick reference for key reusable utility functions and scripts within the project. The goal is to centralize common operations to avoid code duplication.

## **1. Path Management (ocr/utils/path_utils.py)**

For consistent and reliable file path resolution across all scripts, use the centralized path utilities. **Never hardcode paths.**

> **⚠️ DEPRECATION NOTICE**: The convenience functions (`get_project_root()`, `get_data_path()`, etc.) and `PathUtils` class are deprecated. Use the `OCRPathResolver` pattern below for new development.

### **Recommended Modern Approach**

* `get_path_resolver()` -> OCRPathResolver: Returns the global path resolver instance with all project paths.
* Access paths via: `resolver.config.project_root`, `resolver.config.data_dir`, `resolver.config.config_dir`, etc.

### **Legacy Functions (Deprecated)**

* get_project_root() -> Path: Returns the absolute path to the project's root directory. (DEPRECATED)
* get_data_path() -> Path: Returns the path to the main data/ directory. (DEPRECATED)
* get_outputs_path() -> Path: Returns the path to the outputs/ directory where experiment results are saved. (DEPRECATED)
* get_logs_path() -> Path: Returns the path for log files. (DEPRECATED)
* get_checkpoints_path() -> Path: Returns the path for model checkpoints. (DEPRECATED)

### **Usage Examples**

**Modern approach:**
```python
from ocr.utils.path_utils import get_path_resolver

# Get standardized paths through the resolver
resolver = get_path_resolver()
project_root = resolver.config.project_root
config_dir = resolver.config.config_dir
data_dir = resolver.config.data_dir
outputs_dir = resolver.config.output_dir
annotations_path = resolver.config.annotations_dir / "train.json"

if not annotations_path.exists():
    logger.error(f"Annotations not found at: {annotations_path}")
```

**Legacy approach (shows deprecation warnings):**
```python
from ocr.utils.path_utils import get_data_path, get_project_root

# Get standardized paths (deprecated - shows warnings)
project_root = get_project_root()
annotations_path = get_data_path() / "jsons" / "train.json"

if not annotations_path.exists():
    logger.error(f"Annotations not found at: {annotations_path}")
```

## **2. W&B Integration (ocr/utils/wandb_utils.py)**

Use these helper functions to ensure consistent experiment tracking with Weights & Biases.

* generate_run_name(config: DictConfig) -> str: Generates a descriptive and consistent run name based on the experiment's Hydra configuration.
* finalize_run(final_metric: float): Updates a finished W&B run, typically to append the final F1-score to its name for easy identification.
* log_validation_images(...): Logs a batch of validation images with their ground truth and predicted bounding boxes overlaid for visual debugging.

## **3. Visualization (ocr/utils/ocr_utils.py)**

This module contains helpers for visualizing data and model predictions.

* draw_boxes(image, det_polys, gt_polys) -> np.ndarray: Takes an image and draws the predicted polygons (in green) and ground truth polygons (in red) on it. Returns the annotated image.

## **4. Prediction Visualization (ui/visualize_predictions.py)**

A standalone script for visualizing model predictions overlaid on images. Essential for debugging OCR model performance and understanding detection failures.

### **Key Features**

* Loads trained DBNet/CRAFT models from checkpoints
* Runs inference on arbitrary image directories
* Overlays predicted bounding boxes with confidence scores
* Supports both interactive display and file output
* Handles batch processing for efficient visualization

### **Usage Examples**

```bash
# Visualize predictions on problematic images (save to file)
python ui/visualize_predictions.py \
  --image_dir LOW_PERFORMANCE_IMGS \
  --checkpoint outputs/checkpoints/model.ckpt \
  --max_images 5 \
  --save_dir outputs/debug_visualization \
  --score_threshold 0.5

# Interactive visualization (shows plot window)
python ui/visualize_predictions.py \
  --image_dir data/datasets/images/val \
  --checkpoint outputs/checkpoints/last.ckpt \
  --max_images 3
```

### **Parameters**

* `--image_dir`: Directory containing images to analyze (supports jpg, jpeg, png)
* `--checkpoint`: Path to PyTorch Lightning checkpoint file
* `--max_images`: Maximum number of images to process (default: 5)
* `--save_dir`: Directory to save visualization PNG (optional, shows plot if not provided)
* `--score_threshold`: Minimum confidence score for displaying predictions (default: 0.5)

### **Use Cases**

* **Debug Performance Issues**: Visualize what the model detects on low-performing validation batches
* **Quality Assurance**: Verify model predictions on new datasets
* **Error Analysis**: Compare predictions vs ground truth to identify systematic failures
* **Model Comparison**: Generate visualizations for different model checkpoints

## **6. System Resource Monitoring (scripts/monitoring/monitor.sh)**

A convenient script that provides AI-powered system monitoring through Qwen agents.

### **Key Features**

* **Comprehensive Monitoring**: CPU, memory, disk usage, and process analysis
* **Orphaned Process Detection**: Identifies processes that have lost their parent
* **Zombie Process Detection**: Finds defunct processes consuming resources
* **Process Management**: Safe process termination capabilities
* **AI-Powered Analysis**: Natural language queries for system diagnostics

### **Usage Examples**

```bash
# Quick system health check
./scripts/monitoring/monitor.sh "Show system health status"

# Check for problematic processes
./scripts/monitoring/monitor.sh "Monitor system resources and check for orphaned processes"

# Process investigation
./scripts/monitoring/monitor.sh "List top 10 processes by CPU usage"

# Resource analysis
./scripts/monitoring/monitor.sh "Check memory usage and identify high consumers"
```

### **Available Tools**

* **monitor_system_resources**: Comprehensive system diagnostics
* **list_processes**: Process listing with sorting options
* **kill_process**: Safe process termination (with confirmation)

### **Parameters**

* **detailed** (boolean): Include advanced system diagnostics
* **check_orphans** (boolean): Detect orphaned processes
* **sort_by**: Sort processes by "cpu", "memory", "pid", or "time"
* **limit**: Maximum number of processes to display (1-100)

### **Use Cases**

* **Performance Monitoring**: Track system resource usage over time
* **Process Cleanup**: Identify and terminate orphaned or runaway processes
* **System Diagnostics**: Comprehensive health checks and troubleshooting
* **Resource Planning**: Monitor memory and CPU usage patterns

### **Safety Features**

* **Process Validation**: Verifies process existence before termination
* **Graceful Shutdown**: Uses SIGTERM before SIGKILL
* **Permission Checks**: Only terminates processes you have access to
* **Confirmation**: Reports success/failure of operations

### **Integration**

The monitoring tools are available through Qwen's MCP (Model Context Protocol) system. The script automatically enables the system-monitor MCP server for seamless AI-powered monitoring.
