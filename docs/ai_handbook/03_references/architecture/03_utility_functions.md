# **filename: docs/ai_handbook/03_references/architecture/03_utility_functions.md**
<!-- ai_cue:priority=medium -->
<!-- ai_cue:use_when=utilities,helpers,tools,functions -->

# **Reference: Utility Functions**

This reference document provides comprehensive information about key reusable utility functions and scripts within the OCR project for quick lookup and detailed understanding.

## **Overview**

The OCR project provides a comprehensive set of utility functions and scripts for common operations including path management, W&B integration, visualization, prediction analysis, and system monitoring. These utilities are designed to avoid code duplication and ensure consistent operations across the project.

## **Key Concepts**

### **Path Management**
Centralized path resolution utilities for consistent and reliable file path handling across all scripts. Never hardcode paths - use the standardized path resolver pattern.

### **W&B Integration**
Helper functions for consistent experiment tracking with Weights & Biases, including run naming, finalization, and validation image logging.

### **Visualization Tools**
Utilities for visualizing data and model predictions, including bounding box overlay and prediction visualization scripts.

### **System Monitoring**
AI-powered system monitoring through Qwen agents for comprehensive resource tracking and process management.

## **Detailed Information**

### **Path Management (ocr/utils/path_utils.py)**
For consistent and reliable file path resolution across all scripts, use the centralized path utilities.

**Modern Approach (Recommended):**
- `get_path_resolver()` → OCRPathResolver: Returns the global path resolver instance with all project paths
- Access paths via: `resolver.config.project_root`, `resolver.config.data_dir`, etc.

**Legacy Functions (Deprecated):**
- `get_project_root()`, `get_data_path()`, `get_outputs_path()`, `get_logs_path()`, `get_checkpoints_path()` - All deprecated in favor of OCRPathResolver

### **W&B Integration (ocr/utils/wandb_utils.py)**
Helper functions for consistent experiment tracking:
- `generate_run_name(config: DictConfig) → str`: Generates descriptive run names from Hydra config
- `finalize_run(final_metric: float)`: Updates finished runs with final metrics
- `log_validation_images(...)`: Logs validation images with predictions overlaid

### **Visualization (ocr/utils/ocr_utils.py)**
Visualization helpers:
- `draw_boxes(image, det_polys, gt_polys) → np.ndarray`: Draws predicted (green) and ground truth (red) polygons on images

## **Examples**

### **Basic Usage**
```python
# Modern path management
from ocr.utils.path_utils import get_path_resolver

resolver = get_path_resolver()
project_root = resolver.config.project_root
data_dir = resolver.config.data_dir
annotations_path = resolver.config.annotations_dir / "train.json"
```

### **Advanced Usage**
```bash
# Prediction visualization
python ui/visualize_predictions.py \
  --image_dir LOW_PERFORMANCE_IMGS \
  --checkpoint outputs/checkpoints/model.ckpt \
  --max_images 5 \
  --save_dir outputs/debug_visualization \
  --score_threshold 0.5

# System monitoring
./scripts/monitoring/monitor.sh "Show system health status"
```

## **Configuration Options**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| image_dir | str | - | Directory containing images to analyze |
| checkpoint | str | - | Path to PyTorch Lightning checkpoint |
| max_images | int | 5 | Maximum number of images to process |
| save_dir | str | - | Directory to save visualizations |
| score_threshold | float | 0.5 | Minimum confidence for predictions |
| detailed | bool | false | Include advanced diagnostics |
| check_orphans | bool | false | Detect orphaned processes |
| sort_by | str | cpu | Sort processes by cpu/memory/pid/time |
| limit | int | 10 | Maximum processes to display |

## **Best Practices**

- **Use Modern Path Resolver**: Always use `get_path_resolver()` instead of deprecated convenience functions
- **Consistent W&B Naming**: Use `generate_run_name()` for all experiment runs
- **Visualization for Debugging**: Regularly visualize predictions on validation sets
- **System Monitoring**: Use monitoring scripts for performance troubleshooting
- **Process Safety**: Always verify process ownership before termination

## **Troubleshooting**

### **Common Issues**
- **Path Resolution Errors**: Ensure OCRPathResolver is properly initialized
- **W&B Connection Issues**: Check API keys and network connectivity
- **Visualization Failures**: Verify checkpoint file exists and is compatible
- **Process Termination**: Only terminate processes you own

### **Debug Information**
- Enable debug logging: `export LOG_LEVEL=DEBUG`
- Check path resolver: `python -c "from ocr.utils.path_utils import get_path_resolver; print(get_path_resolver().config)"`
- Validate checkpoint: `python -c "import torch; torch.load('path/to/checkpoint.ckpt')"`

## **Related References**

- `docs/ai_handbook/03_references/architecture/01_architecture.md` - System architecture overview
- `docs/ai_handbook/03_references/architecture/06_wandb_integration.md` - W&B integration details
- `docs/ai_handbook/03_references/guides/performance_monitoring_callbacks_usage.md` - Performance monitoring

---

*This document follows the references template. Last updated: 2025-01-15*

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
