# **filename: docs/ai_handbook/04_experiments/TEMPLATE.md**

# **Experiment: [Brief, Descriptive Title]**

* **Date:** YYYY-MM-DD
* **Author:** @username
* **Status:** [Completed / In-Progress / Failed]

## **1. Objective**

*What was the primary goal or hypothesis of this experiment? (e.g., "Test the performance impact of the new PAN decoder against the UNet baseline on receipt data.")*

## **2. Configuration**

* **Base Config:** configs/train.yaml

* # **Key Overrides:**   **```yaml**   **Paste the most important command-line overrides here**   **model:**   **decoder: pan_decoder**   **data:**   **batch_size: 16**   **```**

* Full Command:
  ```bash
  uv run python runners/train.py --config-name train model.decoder=pan_decoder data.batch_size=16
  ```

## **3. Results**

| Metric | Value |
| :---- | :---- |
| Validation F1-Score | 0.000 |
| Validation Precision | 0.000 |
| Validation Recall | 0.000 |
| Training Time | 0h 0m |

* **W&B Link:** [suspicious link removed]
* **Checkpoints:** outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/

## **4. Analysis & Learnings**

*What did we learn from this? Was the hypothesis correct? Were there any surprises?*

* Analysis point 1.
* Analysis point 2.

## **5. Next Steps**

* [ ] Follow-up action 1.
* [ ] Follow-up action 2.
