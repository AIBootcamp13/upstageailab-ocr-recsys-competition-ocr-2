# **Experiment: DBNet++ vs CRAFT Curved Text Detection Analysis**

* **Date:** 2025-10-04
* **Author:** @wchoi189
* **Status:** Completed

## **1. Objective**

*Compare DBNet++ and CRAFT architectures for curved text detection on receipt images. Investigate sudden validation performance drop at step 2049, particularly in recall and hmean metrics. DBNet++ showed slightly higher performance overall.*

## **2. Configuration**

* **Base Config:** train.yaml (Implicit)
* **Key Overrides:**
  ```
  exp_name: dbnetpp_vs_craft_curved_text
  model.architecture_name: dbnetpp,craft
  model.encoder.model_name: resnet50
  model.head.postprocess.use_polygon: true
  trainer.max_epochs: 15
  data.batch_size: 8
  ```
* **Full Command:**

  ```bash
  cd /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2 && python runners/train.py exp_name="dbnetpp_vs_craft_curved_text" model.architecture_name=dbnetpp,craft model.encoder.model_name=resnet50 model.head.postprocess.use_polygon=true trainer.max_epochs=15 data.batch_size=8 -m
  ```

* **W&B Run:** wchoi189_dbnetpp-resnet18-unet-db-head-db-loss-bs8-lr1e-3_hmean0.898 (DBNet++ run)

## **3. Results**

| Metric | Value |
| :---- | :---- |
| Best val/hmean | 0.9027 (step 2869) |
| Final val/hmean | 0.8985 |
| Final val/recall | 0.8457 |
| Final val/precision | 0.9668 |
| Performance Drop at Step 2049 | Recall: 0.847 → 0.738 (-0.109), Hmean: 0.894 → 0.831 (-0.063), Precision: 0.956 → 0.972 (+0.016) |
| Training Time | ~XXm (multi-run) |

* **W&B Link:** [Link to W&B run]
* **Analysis Tools:** `scripts/analyze_experiment.py` - Automated script to process Wandb CSV exports, detect performance anomalies, and generate filtered summaries for AI analysis.

## **4. Analysis & Learnings**

* DBNet++ achieved slightly higher performance than CRAFT, with best hmean of 0.903 vs CRAFT's results (need to compare from multi-run).
* **Sudden Performance Drop:** At step 2049, val/recall dropped sharply from ~0.85 to 0.74, causing hmean to drop from 0.89 to 0.83. Precision actually improved slightly. This suggests data quality issues in the validation batch at that step.
* **Suspected Cause:** Receipt images dataset (~4000 images) likely contains ~200 rotated or corrupted images. The drop frequency (~2 times per 3000 steps) indicates periodic exposure to problematic batches.
* **Tracing the Drop:** To identify specific batches, implement per-batch logging of metrics and image paths. Log validation metrics per batch and flag batches with recall < 0.8.
* **AI Analysis Requirements:** Provide key metrics (precision, recall, hmean per step), anomaly detection (drops >0.1 in recall), model config, and data insights. Avoid logging raw images or excessive per-sample data.
* **Automation Options:** Use Python scripts to process Wandb CSV exports, detect anomalies using threshold-based rules, and generate filtered summaries. Libraries like pandas for data processing and matplotlib for visualization.

## **5. Next Steps**

* [ ] **Implement Batch-Level Logging:** Add code to log per-validation-batch metrics and image paths to identify problematic batches.
* [ ] **Data Quality Audit:** Create script to scan dataset for rotated/corrupted images using OpenCV checks (aspect ratio, text orientation).
* [ ] **Anomaly Detection Script:** Develop Python script to automatically detect performance drops from Wandb data and generate reports.
* [ ] **Compare CRAFT Results:** Analyze the CRAFT run from the multi-run to confirm DBNet++ superiority.
* [ ] **Fix Data Issues:** Remove or correct identified bad images and retrain to validate improvement.</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/ai_handbook/04_experiments/DBNetPP_vs_CRAFT_Curved_Text_Analysis.md
