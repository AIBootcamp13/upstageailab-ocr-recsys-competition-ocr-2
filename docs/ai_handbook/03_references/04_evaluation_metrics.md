# **filename: docs/ai_handbook/03_references/04_evaluation_metrics.md**

# **Reference: Evaluation Metrics**

This document details the evaluation metrics used to measure model performance for this OCR project.

## **1. Primary Metric: CLEval**

The primary and official metric for this project is **CLEval (Character-Level Evaluation)**. It is the industry-standard benchmark for OCR tasks because it provides a more nuanced and accurate measure of performance than simple object detection metrics like IoU.

### **1.1. How CLEval Works**

Instead of just measuring the overlap of bounding boxes, CLEval assesses performance at the character level. The process is as follows:

1. **Polygon Matching:** Predicted text polygons are matched with ground truth polygons.
2. **Character Estimation:** The number of characters within each ground truth polygon is estimated.
3. **Character-Level Scoring:** For each correctly matched polygon pair, character-level precision and recall are calculated based on the estimated character counts. This penalizes predictions that are significantly larger or smaller than the ground truth, even if the box overlap (IoU) is high.

### **1.2. Key Metrics Reported**

The CLEvalMetric class in our project reports the following:

* **Precision:** Of all the characters the model predicted, what fraction were correct?
* **Recall:** Of all the ground truth characters in the dataset, what fraction did the model correctly identify?
* **F1-Score:** The harmonic mean of Precision and Recall, providing a single, balanced measure of a model's performance. This is the main metric to watch.

### **1.3. Implementation Details**

* **Class:** ocr.metrics.cleval_metric.CLEvalMetric
* **Configuration:** The metric is instantiated via Hydra. Key parameters like case_sensitive and penalty weights can be adjusted in the configuration files.

## **2. Secondary / Debugging Metrics**

While CLEval is the primary metric, the following are often logged during training for debugging purposes:

* **Total Loss:** The combined loss value from the model (e.g., DBLoss). This is monitored to ensure the model is learning and converging.
* **Loss Components:** Individual components of the loss function (e.g., bce_loss, dice_loss) are logged separately to help diagnose specific training issues.
