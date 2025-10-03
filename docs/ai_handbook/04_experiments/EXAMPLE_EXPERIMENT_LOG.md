# **Experiment: Initial Run Analysis & High Precision / Low Recall**

* **Date:** 2025-10-01
* **Author:** @wchoi189
* **Status:** Completed

## **1. Objective**

*This run serves as the baseline for the OCR text detection model. The primary goal was to establish initial performance metrics using the default DBNet configuration with a ResNet34 backbone.*

## **2. Configuration**

* **Base Config:** train.yaml (Implicit)
* **Key Overrides:**
  ```
  exp_name: ocr_training-dbnet-pan_decoder-resnet34
  model.encoder.model_name: resnet34
  model.component_overrides.decoder.name: pan_decoder
  model.optimizer.lr: 0.001
  dataloaders.train_dataloader.batch_size: 4
  trainer.max_epochs: 29
  ```
* **Full Command:**

  ```/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/runners/train.py exp_name=ocr_training-dbnet-pan_decoder-resnet34 logger.wandb.enabled=true model.architecture_name=dbnet model/architectures=dbnet model.encoder.model_name=resnet34 model.component_overrides.decoder.name=pan_decoder model.component_overrides.head.name=db_head model.component_overrides.loss.name=db_loss model/optimizers=adam model.optimizer.lr=0.001 model.optimizer.weight_decay=0.0001 dataloaders.train_dataloader.batch_size=4 dataloaders.val_dataloader.batch_size=4 trainer.max_epochs=29 trainer.accumulate_grad_batches=1 trainer.gradient_clip_val=5.0 trainer.precision=32 seed=42 data=default
  ```

## **3. Results**

| Metric | Value |
| :---- | :---- |
| test/hmean | 0.5920 |
| test/precision | 0.9916 |
| test/recall | 0.4367 |
| Training Time | ~63m |

* **W&B Link:** (Link to your W&B run would go here)
* **Checkpoints:** outputs/ocr_training-dbnet-pan_decoder-resnet34/checkpoints/

## **4. Analysis & Learnings**

* The model has learned to be extremely conservative. The **precision is nearly perfect (0.99)**, meaning that when it predicts a text box, it is almost always correct.
* However, the **recall is very low (0.43)**, indicating the model is failing to identify more than half of the text on the receipts. This is the primary bottleneck for performance.
* This imbalance suggests an issue with the confidence threshold in post-processing rather than a fundamental problem with the learned features. The model is likely producing correct probability maps, but we are too strict when converting them into final bounding boxes.
* The StepLR learning rate scheduler was configured with step_size: 100, but the training only ran for 29 epochs, so the learning rate never decayed. This is suboptimal.

## **5. Next Steps**

* [x] **Hypothesis 1: Lower the Bounding Box Threshold.**
  * **Action:** Run an experiment with model.head.postprocess.box_thresh=0.3 to encourage the model to produce more bounding boxes, directly targeting the low recall.
* [ ] **Hypothesis 2: Improve the LR Scheduler.**
  * **Action:** After analyzing the threshold change, run an experiment replacing StepLR with CosineAnnealingLR to provide a more dynamic learning rate schedule. Set model.scheduler.T_max=${trainer.max_epochs}.
* [ ] **Hypothesis 3: Increase Augmentation.**
  * **Action:** If recall remains an issue, introduce more aggressive augmentations like albumentations.RandomBrightnessContrast and albumentations.GaussNoise to the training transform pipeline.
