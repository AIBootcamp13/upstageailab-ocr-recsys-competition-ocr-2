# **filename: docs/ai_handbook/03_references/03_utility_functions.md**

# **Reference: Utility Functions**

This document serves as a quick reference for key reusable utility functions and scripts within the project. The goal is to centralize common operations to avoid code duplication.

## **1. Path Management (ocr/utils/path_utils.py)**

For consistent and reliable file path resolution across all scripts, use the centralized path utilities. **Never hardcode paths.**

### **Key Functions**

* get_project_root() -> Path: Returns the absolute path to the project's root directory.
* get_data_path() -> Path: Returns the path to the main data/ directory.
* get_outputs_path() -> Path: Returns the path to the outputs/ directory where experiment results are saved.
* get_logs_path() -> Path: Returns the path for log files.
* get_checkpoints_path() -> Path: Returns the path for model checkpoints.

### **Usage Example**

```python
from ocr.utils.path_utils import get_data_path, get_project_root

# Get standardized paths
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

## **4. General Best Practices**

* **Import Convention**: Always import utility functions from their canonical modules.
* **Reusability**: If you write a helper function that could be useful elsewhere, add it to the appropriate utility module and document it here.
* **Configuration over Code**: Prefer writing functions that are configured via Hydra DictConfig objects rather than taking many individual arguments.
