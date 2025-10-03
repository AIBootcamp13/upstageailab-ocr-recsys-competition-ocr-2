# **Experiment Analysis Framework Handbook**

This document provides a comprehensive guide to a framework for analyzing, debugging, and automating the evaluation of machine learning experiments. It is designed to help teams identify performance issues, understand their root causes, and implement robust monitoring and data quality pipelines.

## **Table of Contents**
1. [**Introduction**](#1-introduction)
    * [1.1. Purpose](#11-purpose)
    * [1.2. When to Use This Framework](#12-when-to-use-this-framework)
2. [**Quick Start & Usage**](#2-quick-start--usage)
    * [2.1. Basic Analysis Commands](#21-basic-analysis-commands)
    * [2.2. Team Workflow Integration](#22-team-workflow-integration)
3. [**Phase 1: Problem Definition & Initial Investigation**](#3-phase-1-problem-definition--initial-investigation)
    * [3.1. Structuring an Analysis Request](#31-structuring-an-analysis-request)
    * [3.2. Immediate Investigation Steps](#32-immediate-investigation-steps)
      * [3.2.1. Batch-Level Performance Tracking](#321-batch-level-performance-tracking)
      * [3.2.2. Data Quality Audit Script](#322-data-quality-audit-script)
4. [**Phase 2: Automated Analysis & Reporting**](#4-phase-2-automated-analysis--reporting)
    * [4.1. The Complete Analysis Script](#41-the-complete-analysis-script)
    * [4.2. Essential Metrics for AI Analysis](#42-essential-metrics-for-ai-analysis)
    * [4.3. Auto-Generated Experiment Reports](#43-auto-generated-experiment-reports)
5. [**Phase 3: Building a Robust MLOps Pipeline**](#5-phase-3-building-a-robust-mlops-pipeline)
    * [5.1. Data Quality Pipeline](#51-data-quality-pipeline)
    * [5.2. Real-time Monitoring System](#52-real-time-monitoring-system)
    * [5.3. Integration with the Training Pipeline](#53-integration-with-the-training-pipeline)
6. [**Advanced Analysis Techniques**](#6-advanced-analysis-techniques)
    * [6.1. Statistical Anomaly Detection](#61-statistical-anomaly-detection)
    * [6.2. Batch-Level Data Profiling](#62-batch-level-data-profiling)
    * [6.3. Automated Experiment Comparison](#63-automated-experiment-comparison)
    * [6.4. Automated Hyperparameter Impact Analysis](#64-automated-hyperparameter-impact-analysis)
7. [**Advanced Automation Strategies**](#7-advanced-automation-strategies)
    * [7.1. Automated Recovery Strategies](#71-automated-recovery-strategies)
    * [7.2. Full MLOps Integration Example](#72-full-mlops-integration-example)
8. [**Implementation Guide & Best Practices**](#8-implementation-guide--best-practices)
    * [8.1. Implementation Checklist](#81-implementation-checklist)
    * [8.2. Actionable Recommendations Summary](#82-actionable-recommendations-summary)
    * [8.3. Success Metrics](#83-success-metrics)
9. [**Appendix: Full Code**](#9-appendix-full-code)
    * [9.1. Final Automation Script (analyze_experiment.py)](#91-final-automation-script-analyze_experimentpy)
    * [9.2. Complete Training Script Example](#92-complete-training-script-example)

## **1. Introduction**

### **1.1. Purpose**

The Experiment Analysis Framework provides a systematic approach to diagnosing performance drops, understanding data quality issues, and automating the reporting process for machine learning experiments. By standardizing analysis, it aims to reduce debugging time, improve reproducibility, and enable data-driven decision-making.

### **1.2. When to Use This Framework**

* ✅ After any training run with unexpected results
* ✅ When comparing multiple model architectures or hyperparameters
* ✅ When investigating data quality as a potential root cause of failure
* ✅ For creating standardized documentation and ensuring reproducibility

## **2. Quick Start & Usage**

### **2.1. Basic Analysis Commands**

```bash
# Basic analysis of a finished experiment
python analyze_experiment.py --run-name "your_wandb_run_name"

# Run analysis including a data quality audit
python analyze_experiment.py
    --run-name "your_wandb_run_name"
    --data-dir "/path/to/your/dataset"
    --output-dir "./experiment_reports"

# Compare multiple runs
python compare_experiments.py
    --runs "run_name_1,run_name_2,run_name_3"
    --output-dir "./comparison_reports"
```

### **2.2. Team Workflow Integration**

1. **Before Training**: Set up the monitoring configuration (config.yaml).
2. **During Training**: The automated TrainingMonitor runs in the background, providing real-time alerts.
3. **After Training**: Run the analyze_experiment.py script for a comprehensive report.
4. **Decision Making**: Use the generated insights and comparisons to plan the next steps.
5. **Documentation**: Archive the auto-generated reports for experiment tracking.

## **3. Phase 1: Problem Definition & Initial Investigation**

A successful analysis begins with a clear problem statement and immediate, targeted investigation.

### **3.1. Structuring an Analysis Request**

Use this template to clearly define the problem, provide technical context, and ask specific, prioritized questions.

```markdown
# Experiment Analysis Request Template

## Context
**Objective**: [e.g., "Compare DBNet++ vs CRAFT for curved text detection"]
**Primary Issue**: [e.g., "Sudden validation recall drop from 0.89 to 0.74 at step 2049"]
**Hypothesis**: [e.g., "Data quality issues in the validation set, such as rotated images"]

## Technical Details
**Model Configuration**:
- Architecture: [e.g., DBNet++]
- Key parameters: [e.g., batch_size=8, learning_rate=1e-3]
- Training command: `python train.py --config configs/dbnetpp.yaml`

**Data Information**:
- Dataset size: [e.g., ~4000 images]
- Data type: [e.g., receipt images]
- Suspected issues: [e.g., ~200 rotated or corrupted images]

## Specific Questions (Prioritized)
1.  **Root Cause**: How can we trace the performance drop to specific validation batches?
2.  **Prevention**: How can we automatically detect and filter problematic images in the future?
```

### **3.2. Immediate Investigation Steps**

#### **3.2.1. Batch-Level Performance Tracking**

Modify your validation loop to log metrics for each batch. This helps pinpoint exactly which data causes a performance drop.

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

    # Log image paths for problematic batches based on a threshold
    if metrics['recall'] < 0.8:
        wandb.log({f"problematic_batch_{batch_idx}": images_paths})
```

#### **3.2.2. Data Quality Audit Script**

Run this standalone script on your dataset to identify potentially corrupted, rotated, or low-quality images before training.

```python
import cv2
import numpy as np
from pathlib import Path

def audit_image_quality(image_dir):
    """Identify potentially problematic images"""
    issues = []
    for img_path in Path(image_dir).glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if is_corrupted(img):
            issues.append({"path": img_path, "issue": "corruption"})
        elif is_severely_rotated(img):
            issues.append({"path": img_path, "issue": "rotation"})
    return issues

def is_corrupted(img):
    # Check for load errors, unusual aspect ratios, or very small images
    return img is None or img.shape[0] < 10 or img.shape[1] < 10

def is_severely_rotated(img):
    # Placeholder for rotation detection logic (e.g., text orientation analysis)
    pass
```

## **4. Phase 2: Automated Analysis & Reporting**

The core of the framework is a script that automates the entire analysis process, from data extraction to report generation.

### **4.1. The Complete Analysis Script**

The analyze_experiment.py script serves as the main entry point for post-experiment analysis. It connects to wandb, extracts metrics, detects anomalies, performs root cause analysis, and generates reports. See the [Appendix](https://www.google.com/search?q=%23appendix-full-code) for the full script.

### **4.2. Essential Metrics for AI Analysis**

For the automated analysis to be effective, the script needs a structured summary of the experiment. This includes model configuration, key performance metrics, data insights, and identified anomalies.

```python
experiment_summary = {
    "model_config": {
        "architecture": "dbnetpp", "encoder": "resnet50", "batch_size": 8
    },
    "performance_metrics": {
        "best_hmean": 0.898, "final_recall": 0.74, "performance_drop_step": 2049
    },
    "data_insights": {
        "total_images": 4000, "suspected_bad_images": 200
    },
    "anomalies": [{
        "step": 2049, "metric": "recall", "drop_magnitude": 0.15,
        "suspected_cause": "batch with rotated images"
    }]
}
```

### **4.3. Auto-Generated Experiment Reports**

The script can generate comprehensive markdown reports automatically, ensuring consistent and thorough documentation for every experiment.

```python
def generate_comprehensive_report(data):
    """Generate a comprehensive experiment analysis report in markdown"""
    report = f"""
# Experiment Analysis: {data['run_name']}

## Executive Summary
- **Key Finding**: {data['key_finding']}
- **Recommendation**: {data['recommendation']}

## Performance Analysis
| Metric    | Best  | Final | Drop  |
|-----------|-------|-------|-------|
| Recall    | {data['best_recall']:.3f} | {data['final_recall']:.3f} | {data['recall_drop']:.3f} |
| H-mean    | {data['best_hmean']:.3f} | {data['final_hmean']:.3f} | {data['hmean_drop']:.3f} |

## Root Cause Analysis
- **Primary Cause**: {data['root_cause']}
- **Evidence**: {data['evidence']}
"""
    return report
```

## **5. Phase 3: Building a Robust MLOps Pipeline**

Integrate the framework's components directly into your training pipeline for real-time monitoring and proactive quality control.

### **5.1. Data Quality Pipeline**

Create a preprocessing pipeline that automatically filters out low-quality images before they enter the training loop.

```python
class ImageQualityFilter:
    def __init__(self, min_resolution=(100, 100), max_rotation=15, blur_threshold=100):
        self.min_resolution = min_resolution
        self.max_rotation = max_rotation
        self.blur_threshold = blur_threshold

    def assess_image_quality(self, image):
        """Comprehensive image quality assessment"""
        # ... implementation for checking resolution, rotation, blur, etc. ...
        quality_metrics = self.calculate_metrics(image)
        is_acceptable = self.check_thresholds(quality_metrics)
        return {"is_acceptable": is_acceptable, "metrics": quality_metrics}

    def filter_batch(self, images):
        """Filter out low-quality images from a batch"""
        filtered_images = [img for img in images if self.assess_image_quality(img)["is_acceptable"]]
        return filtered_images
```

### **5.2. Real-time Monitoring System**

The TrainingMonitor class hooks into your training loop to detect anomalies as they happen, enabling early stopping or immediate alerts.

```python
from collections import defaultdict

class TrainingMonitor:
    def __init__(self, alert_thresholds):
        self.alert_thresholds = alert_thresholds
        self.metric_history = defaultdict(list)

    def log_step(self, step, metrics, batch_info=None):
        """Log metrics for a step and check for anomalies in real-time"""
        for metric, value in metrics.items():
            self.metric_history[metric].append(value)

        anomalies = self.detect_real_time_anomalies(metrics)
        if anomalies:
            self.handle_anomalies(step, anomalies, batch_info)
        return anomalies

    def detect_real_time_anomalies(self, current_metrics):
        # ... implementation for statistical and threshold-based anomaly detection ...
        pass

    def handle_anomalies(self, step, anomalies, batch_info):
        # ... implementation for logging, alerting, or saving debug checkpoints ...
        print(f"Anomaly detected at step {step}: {anomalies}")
```

### **5.3. Integration with the Training Pipeline**

Here is how you would integrate the TrainingMonitor into a standard training script.

```python
# In your main training script
monitor = TrainingMonitor(alert_thresholds={'val/recall': 0.75})

for epoch in range(max_epochs):
    for batch_idx, batch in enumerate(train_loader):
        # ... training logic ...
        global_step += 1

        if global_step % validation_interval == 0:
            val_metrics = validate_model(model, val_loader)
            monitor.log_step(step=global_step, metrics=val_metrics)
```

## **6. Advanced Analysis Techniques**

### **6.1. Statistical Anomaly Detection**

Go beyond simple thresholds by using statistical methods like Z-scores or change-point detection to identify significant deviations in performance metrics.

```python
import numpy as np
from scipy import stats
import ruptures as rpt

def detect_statistical_anomalies(metrics_timeseries):
    """Detect anomalies using Z-score and change-point detection"""
    # Z-score based detection
    z_scores = np.abs(stats.zscore(metrics_timeseries))
    z_anomalies = np.where(z_scores > 3)[0]

    # Change point detection
    algo = rpt.Pelt(model="rbf").fit(np.array(metrics_timeseries))
    change_points = algo.predict(pen=10)

    return {"z_score_anomalies": z_anomalies, "change_points": change_points}
```

### **6.2. Batch-Level Data Profiling**

For deep-dive analysis, profile every validation batch across multiple dimensions to correlate performance with data characteristics.

```python
def profile_validation_batches(val_loader, model):
    """Profile each validation batch to identify problematic ones"""
    batch_profiles = []
    for images, targets in val_loader:
        profile = {
            "image_stats": analyze_image_batch_quality(images),
            "target_complexity": analyze_target_complexity(targets),
            "prediction_confidence": get_prediction_confidence(model, images),
            "performance": evaluate_batch(model, images, targets)
        }
        batch_profiles.append(profile)
    return batch_profiles
```

### **6.3. Automated Experiment Comparison**

Automate the comparison of multiple runs to identify key differences in configuration, performance, and stability.

```python
class ExperimentComparator:
    def __init__(self, wandb_project):
        self.project = wandb_project
        self.api = wandb.Api()

    def compare_runs(self, run_names):
        """Compare multiple experiment runs and summarize findings"""
        run_data = {name: self.extract_run_data(name) for name in run_names}

        comparison = {
            "best_performer": self.identify_best_performer(run_data),
            "key_differences": self.identify_key_differences(run_data),
            "recommendations": self.generate_recommendations(run_data)
        }
        return comparison
```

### **6.4. Automated Hyperparameter Impact Analysis**

Analyze a collection of experiments to determine which hyperparameters have the most significant impact on performance and are most correlated with anomalies.

```python
import pandas as pd

class HyperparameterAnalyzer:
    def __init__(self, project_name):
        self.project = project_name
        self.api = wandb.Api()

    def analyze_hyperparameter_impact(self):
        """Analyze which hyperparameters most impact performance drops"""
        runs = self.api.runs(self.project)
        # ... extract config and summary from all runs ...
        df = pd.DataFrame(analysis_data)

        # Calculate correlation between numeric params and performance drop
        correlations = df.corr()['performance_drop'].sort_values()

        # Calculate anomaly rates by categorical params
        anomaly_rates = df.groupby('architecture')['had_anomaly'].mean()

        return {"correlations": correlations, "anomaly_rates": anomaly_rates}
```

## **7. Advanced Automation Strategies**

### **7.1. Automated Recovery Strategies**

Build a system that not only detects anomalies but also attempts to recover from them automatically during training.

```python
class AutoRecoverySystem:
    def implement_recovery_strategy(self, anomaly, model_state):
        """Automatically implement recovery actions based on anomaly type"""
        if anomaly['severity'] == 'high' and anomaly['type'] == 'sudden_drop':
            print("High severity drop detected. Rolling back to best checkpoint.")
            return self.rollback_to_checkpoint()

        elif 'loss_plateau' in anomaly['type']:
            print("Loss plateau detected. Adjusting learning rate.")
            return self.adjust_learning_rate(factor=0.5)

        else:
            print("Triggering early stopping due to unrecoverable state.")
            return self.trigger_early_stopping()
```

### **7.2. Full MLOps Integration Example**

Define the entire monitoring, recovery, and reporting pipeline in a central configuration file.

```yaml
# config/experiment_config.yaml
experiment_monitoring:
  enable_real_time_analysis: true
  anomaly_detection:
    performance_drop_threshold: 0.05
  auto_recovery:
    enable: true
    strategies:
      - rollback_on_severe_drop
      - adjust_lr_on_plateau
  reporting:
    auto_generate_reports: true
  alerts:
    slack_webhook: "[https://hooks.slack.com/](https://hooks.slack.com/)..."
```

## **8. Implementation Guide & Best Practices**

### **8.1. Implementation Checklist**

#### **Phase 1: Immediate Setup (Day 1)**

* [ ] Install dependencies: wandb, opencv-python, pandas, scipy, ruptures.
* [ ] Run the standalone data quality audit script on your current dataset.
* [ ] Implement basic batch-level metric logging in your validation loop.

#### **Phase 2: Enhanced Monitoring (Week 1)**

* [ ] Integrate the TrainingMonitor class into your main training script.
* [ ] Set up the analyze_experiment.py script to run automatically post-training.
* [ ] Configure alerts for Slack or email.

#### **Phase 3: Advanced Analytics (Week 2-3)**

* [ ] Implement the HyperparameterAnalyzer for deeper insights.
* [ ] Deploy the AutoRecoverySystem for critical training jobs.
* [ ] Fully integrate the framework using a central config.yaml.

### **8.2. Actionable Recommendations Summary**

#### **Immediate Actions (Next 24 hours)**

1. Run the data quality audit script on the primary dataset.
2. Identify and manually inspect images from any problematic batches.
3. Implement batch-level logging for all future experiments.

#### **Short-term Improvements (Next week)**

1. Set up automated monitoring using the TrainingMonitor class.
2. Create a data preprocessing pipeline with quality filters.
3. Implement early stopping based on severe anomaly detection.

#### **Long-term Framework (Next month)**

1. Build a comprehensive experiment comparison dashboard.
2. Automate report generation for all training runs.
3. Establish team-wide best practices for using the framework.

### **8.3. Success Metrics**

* [ ] Reduce time-to-detect anomalies from hours to minutes.
* [ ] Achieve >90% accuracy in automated root cause identification.
* [ ] Decrease manual experiment analysis time by >80%.
* [ ] Prevent >50% of training failures through early detection and recovery.

## **9. Appendix: Full Code**

### **9.1. Final Automation Script (analyze_experiment.py)**

```python
#!/usr/bin/env python3
"""
Complete experiment analysis automation script.
Connects to WandB, extracts metrics, detects anomalies, analyzes data quality,
and generates a comprehensive markdown report.

Usage: python analyze_experiment.py --run-name <wandb_run_name> --data-dir /path/to/data
"""
import argparse
import wandb
import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class ExperimentAnalyzer:
    def __init__(self, run_name):
        self.run_name = run_name
        self.api = wandb.Api()
        # NOTE: Update "your-entity/your-project" to your wandb path
        self.run = self.api.run(f"your-entity/your-project/{run_name}")

    def extract_metrics(self):
        """Extracts key metrics from the wandb run history."""
        metrics_data = defaultdict(list)
        history = self.run.scan_history()
        for row in history:
            if 'val/recall' in row:
                metrics_data['recall'].append(row['val/recall'])
                metrics_data['precision'].append(row.get('val/precision'))
                metrics_data['hmean'].append(row.get('val/hmean'))
                metrics_data['steps'].append(row.get('_step'))
        return metrics_data

    def detect_anomalies(self, metrics_data):
        """Detects anomalies like sudden performance drops in metrics data."""
        anomalies = []
        recall = metrics_data['recall']
        for i in range(1, len(recall)):
            drop = recall[i-1] - recall[i]
            if drop > 0.1:  # Significant drop threshold
                anomalies.append({
                    'type': 'sudden_drop', 'metric': 'recall',
                    'step': metrics_data['steps'][i], 'drop_magnitude': drop
                })
        return anomalies

    # ... Other methods from the prompt would go here ...

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--run-name', required=True, help='Wandb run name')
    parser.add_argument('--output-dir', default='./analysis_reports', help='Output directory')
    parser.add_argument('--data-dir', help='Path to training data for quality analysis')
    args = parser.parse_args()

    analyzer = ExperimentAnalyzer(args.run_name)

    print("1. Extracting metrics...")
    metrics_data = analyzer.extract_metrics()

    print("2. Detecting anomalies...")
    anomalies = analyzer.detect_anomalies(metrics_data)

    # ... The rest of the main execution logic from the prompt ...

    print(f"nAnalysis complete! Reports saved to: {args.output-dir}")

if __name__ == "__main__":
    main()
```

### **9.2. Complete Training Script Example**

```python
# complete_training_script.py
import yaml
import wandb
from experiment_analyzer import TrainingMonitor
from auto_recovery import AutoRecoverySystem

def main():
    with open('config/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    wandb.init(project=config['project_name'], name=config['experiment_name'], config=config)

    monitor = TrainingMonitor(config['experiment_monitoring']['anomaly_detection'])
    recovery_system = AutoRecoverySystem(
        config['checkpoint_dir'],
        config['experiment_monitoring']['auto_recovery']['strategies']
    )

    model, train_loader, val_loader, optimizer = setup_training(config)
    global_step = 0

    for epoch in range(config['trainer']['max_epochs']):
        for batch in train_loader:
            train_step(model, batch, optimizer)

            if global_step % config['validation_interval'] == 0:
                val_metrics = validate_model(model, val_loader)
                anomalies = monitor.log_step(global_step, val_metrics)

                if anomalies and config['experiment_monitoring']['auto_recovery']['enable']:
                    recovery_system.implement_recovery_strategy(anomalies[0], model.state_dict())

            global_step += 1

    finalize_experiment(model)

if __name__ == "__main__":
    main()
```
