# **filename: docs/ai_handbook/02_protocols/configuration/21_experiment_analysis_framework_handbook.md**
<!-- ai_cue:priority=medium -->
<!-- ai_cue:use_when=experiment_analysis,mlops_monitoring,automated_reporting -->

# **Protocol: Experiment Analysis Framework**

## **Overview**
This protocol provides a comprehensive framework for analyzing, debugging, and automating the evaluation of machine learning experiments. It establishes systematic approaches to identify performance issues, understand root causes, and implement robust monitoring and data quality pipelines for reproducible ML experimentation.

## **Prerequisites**
- Access to WandB for experiment tracking and metrics logging
- Python environment with required packages: wandb, opencv-python, pandas, scipy, ruptures
- Training scripts with validation loops that log batch-level metrics
- Dataset access for quality audits and profiling
- Understanding of ML experiment lifecycle and performance metrics

## **Procedure**

### **Step 1: Set Up Basic Experiment Monitoring**
**Action:** Implement batch-level metric logging and basic anomaly detection in training scripts:
```python
# Add to training validation loop
def log_batch_metrics(batch_idx, step, metrics, images_paths=None):
    """Log per-batch validation metrics to identify problematic batches"""
    wandb.log({
        f"batch_{batch_idx}/recall": metrics['recall'],
        f"batch_{batch_idx}/precision": metrics['precision'],
        f"batch_{batch_idx}/hmean": metrics['hmean'],
        "global_step": step
    })

    # Log image paths for problematic batches based on threshold
    if metrics['recall'] < 0.8:
        wandb.log({f"problematic_batch_{batch_idx}": images_paths})
```

**Expected Outcome:** Real-time visibility into batch-level performance variations.

### **Step 2: Implement Automated Analysis Scripts**
**Action:** Deploy the experiment analysis automation script for post-training evaluation:
```bash
# Create analyze_experiment.py with comprehensive analysis capabilities
python analyze_experiment.py --run-name "your_wandb_run_name" --data-dir "/path/to/dataset" --output-dir "./experiment_reports"
```

**Expected Outcome:** Automated generation of experiment reports with anomaly detection and root cause analysis.

### **Step 3: Configure Real-time Monitoring System**
**Action:** Integrate TrainingMonitor class into training pipeline for proactive anomaly detection:
```python
from collections import defaultdict

class TrainingMonitor:
    def __init__(self, alert_thresholds):
        self.alert_thresholds = alert_thresholds
        self.metric_history = defaultdict(list)

    def log_step(self, step, metrics, batch_info=None):
        """Log metrics and check for anomalies in real-time"""
        for metric, value in metrics.items():
            self.metric_history[metric].append(value)

        anomalies = self.detect_real_time_anomalies(metrics)
        if anomalies:
            self.handle_anomalies(step, anomalies, batch_info)
        return anomalies
```

**Expected Outcome:** Immediate detection and alerting of performance anomalies during training.

### **Step 4: Establish Data Quality Pipeline**
**Action:** Implement automated data quality filtering and preprocessing pipeline:
```python
class ImageQualityFilter:
    def __init__(self, min_resolution=(100, 100), max_rotation=15, blur_threshold=100):
        self.min_resolution = min_resolution
        self.max_rotation = max_rotation
        self.blur_threshold = blur_threshold

    def filter_batch(self, images):
        """Filter out low-quality images from training batches"""
        filtered_images = [img for img in images if self.assess_image_quality(img)["is_acceptable"]]
        return filtered_images
```

**Expected Outcome:** Proactive prevention of data quality issues affecting model performance.

## **Configuration Structure**
```
# Experiment Analysis Framework Configuration
experiment_monitoring:
  enable_real_time_analysis: true
  anomaly_detection:
    performance_drop_threshold: 0.05
    statistical_anomaly_zscore: 3.0
  auto_recovery:
    enable: true
    strategies:
      - rollback_on_severe_drop
      - adjust_lr_on_plateau
      - trigger_early_stopping
  reporting:
    auto_generate_reports: true
    report_formats: [markdown, json, html]
  alerts:
    slack_webhook: "https://hooks.slack.com/..."
    email_recipients: ["team@company.com"]

# Training Script Integration
training_config:
  validation_interval: 100
  batch_logging: true
  checkpoint_dir: "./checkpoints"
  experiment_name: "ocr_model_v1"

# Data Quality Pipeline
data_quality:
  min_resolution: [100, 100]
  max_rotation_degrees: 15
  blur_threshold: 100
  enable_filtering: true
```

## **Validation**
Run these validation checks after implementing the framework:

```python
# Test 1: Verify WandB integration
import wandb
run = wandb.init(project="test_project", name="validation_test")
wandb.log({"test_metric": 0.95})
run.finish()
print("✅ WandB logging functional")

# Test 2: Validate analysis script
import subprocess
result = subprocess.run([
    "python", "analyze_experiment.py",
    "--run-name", "test_run",
    "--output-dir", "./test_reports"
], capture_output=True, text=True)
assert "Analysis complete" in result.stdout, "Analysis script failed"
print("✅ Analysis script functional")

# Test 3: Test monitoring integration
from experiment_analyzer import TrainingMonitor
monitor = TrainingMonitor({'val/recall': 0.75})
anomalies = monitor.log_step(100, {'val/recall': 0.70})
assert len(anomalies) > 0, "Anomaly detection failed"
print("✅ Real-time monitoring functional")

# Test 4: Validate data quality pipeline
from data_quality import ImageQualityFilter
filter = ImageQualityFilter()
# Test with sample image
quality_result = filter.assess_image_quality(sample_image)
assert 'is_acceptable' in quality_result, "Quality assessment failed"
print("✅ Data quality pipeline functional")

print("🎉 All validation checks passed - Experiment Analysis Framework ready!")
```

## **Troubleshooting**

### **Issue: WandB metrics not logging properly**
**Solution:** Verify WandB initialization and API key:
```python
import wandb
wandb.login()  # Ensure API key is set
run = wandb.init(project="your_project", name="test_run")
print(f"WandB run URL: {run.url}")
```

### **Issue: Analysis script fails to connect to WandB**
**Solution:** Check project and run name configuration:
```bash
# Verify run exists
wandb runs list --project your_project

# Test API connection
python -c "import wandb; api = wandb.Api(); print('API connected')"
```

### **Issue: Real-time monitoring not detecting anomalies**
**Solution:** Adjust anomaly detection thresholds:
```python
# Lower thresholds for more sensitive detection
alert_thresholds = {
    'val/recall': 0.85,  # Increased sensitivity
    'val/precision': 0.80,
    'train/loss': 2.0
}
```

### **Issue: Data quality filter too aggressive**
**Solution:** Tune quality thresholds based on dataset characteristics:
```python
# Adjust based on your dataset
quality_filter = ImageQualityFilter(
    min_resolution=(50, 50),  # Less restrictive
    max_rotation=30,          # More tolerant
    blur_threshold=50         # Less strict
)
```

### **Issue: Automated reports not generating**
**Solution:** Check output directory permissions and dependencies:
```bash
# Ensure output directory exists and is writable
mkdir -p ./experiment_reports
python -c "import pandas, scipy; print('Dependencies OK')"

# Test report generation manually
python analyze_experiment.py --run-name test_run --output-dir ./experiment_reports
```

## **Related Documents**
- `02_protocols/configuration/20_command_builder_testing_guide.md` - Testing strategies for experiment configurations
- `02_protocols/configuration/20_hydra_config_resolution_troubleshooting.md` - Troubleshooting configuration issues
- `02_protocols/configuration/23_hydra_configuration_testing_implementation_plan.md` - Comprehensive testing for ML configurations
- `02_protocols/development/06_context_logging.md` - Context logging for experiment tracking
- `03_references/architecture/06_wandb_integration.md` - WandB integration reference
- `03_references/guides/performance_monitoring_callbacks_usage.md` - Performance monitoring implementation

---

*This document follows the configuration protocol template. Last updated: October 13, 2025*
