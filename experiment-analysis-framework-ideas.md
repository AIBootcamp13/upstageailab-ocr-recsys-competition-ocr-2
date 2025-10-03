
# Experiment Analysis Framework: DBNet++ vs CRAFT Performance Investigation

## Improved Prompt Structure

### 1. **Context & Objective** (Clear Problem Statement)
```markdown
**Experiment Goal**: Compare DBNet++ vs CRAFT architectures for curved text detection
**Primary Issue**: Sudden validation performance drop at step 2049
**Key Metrics**: Recall (0.89→0.74), Precision (minor drop), H-mean (significant drop)
**Hypothesis**: Data quality issues in receipt images dataset (~4000 images)
```

### 2. **Specific Technical Details**
```markdown
**Run Configuration**:
- Model: DBNet++ with ResNet50 encoder
- Training Command: [your command]
- Wandb Run: `wchoi189_dbnetpp-resnet18-unet-db-head-db-loss-bs8-lr1e-3_hmean0.898`
- Dataset: Receipt images (4000+ samples)
- Suspected Issues: Rotated/corrupted images (~200 samples)
```

### 3. **Structured Questions with Priority**
```markdown
**Priority 1 - Root Cause Analysis**:
1. How to trace performance drop to specific validation batches?
2. Methods to identify corrupted/rotated images in dataset?

**Priority 2 - Analysis Framework**:
3. What metrics should be logged for AI analysis?
4. How to automate useful metric collection?
```

---


## Analysis Strategy

### A. Immediate Investigation Steps

#### 1. **Batch-Level Performance Tracking**
```python
# Add to your training loop
def log_batch_metrics(batch_idx, step, metrics, images_paths=None):
    """Log per-batch validation metrics to identify problematic batches"""
    wandb.log({
        f"batch_{batch_idx}/recall": metrics['recall'],
        f"batch_{batch_idx}/precision": metrics['precision'],
        f"batch_{batch_idx}/hmean": metrics['hmean'],
        "global_step": step
    })

    # Log image paths for problematic batches
    if metrics['recall'] < 0.8:  # Threshold for investigation
        wandb.log({f"problematic_batch_{batch_idx}": images_paths})
```

#### 2. **Data Quality Audit Script**
```python
import cv2
import numpy as np
from pathlib import Path

def audit_image_quality(image_dir):
    """Identify potentially problematic images"""
    issues = []

    for img_path in Path(image_dir).glob("*.jpg"):
        img = cv2.imread(str(img_path))

        # Check for rotation issues
        if is_severely_rotated(img):
            issues.append({"path": img_path, "issue": "rotation"})

        # Check for corruption
        if is_corrupted(img):
            issues.append({"path": img_path, "issue": "corruption"})

    return issues

def is_severely_rotated(img):
    # Implement rotation detection logic
    # E.g., text orientation analysis
    pass

def is_corrupted(img):
    # Check for unusual aspect ratios, very dark/bright images
    return img is None or img.shape[0] < 10 or img.shape[1] < 10
```

### B. Automated Metrics Collection

#### 1. **Essential Metrics for AI Analysis**
```python
experiment_summary = {
    "model_config": {
        "architecture": "dbnetpp",
        "encoder": "resnet50",
        "batch_size": 8,
        "learning_rate": 1e-3
    },
    "performance_metrics": {
        "best_hmean": 0.898,
        "final_recall": 0.74,
        "final_precision": 0.85,
        "performance_drop_step": 2049
    },
    "data_insights": {
        "total_images": 4000,
        "suspected_bad_images": 200,
        "drop_frequency": "2 times per 3000 steps"
    },
    "anomalies": [
        {
            "step": 2049,
            "metric": "recall",
            "drop_magnitude": 0.15,
            "suspected_cause": "batch with rotated images"
        }
    ]
}
```

#### 2. **Automated Analysis Script**
```python
def generate_experiment_report(wandb_run_name, csv_files):
    """Generate concise experiment analysis"""

    # Load metrics
    metrics_data = load_csv_metrics(csv_files)

    # Detect anomalies
    anomalies = detect_performance_drops(metrics_data)

    # Generate summary
    report = {
        "experiment_id": wandb_run_name,
        "performance_summary": calculate_summary_stats(metrics_data),
        "anomalies_detected": anomalies,
        "recommendations": generate_recommendations(anomalies)
    }

    return report
```

---
