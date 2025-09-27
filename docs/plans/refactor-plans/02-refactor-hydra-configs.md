# Hydra Config Refactoring Plan

This plan outlines the steps to refactor the Hydra configuration to align with the `lightning-hydra-template` structure, fix existing errors, and improve modularity.

## Todo (Likely COMPLETE)

1.  **[In Progress] Fix Model Channel Mismatch:**
    -   **Goal:** Resolve the `RuntimeError` caused by a channel mismatch between the UNet decoder and the DBNet head.
    -   **Action:** Modify `configs/preset/models/decoder/unet.yaml` to set `output_channels: 256` to match the head's expected `in_channels`.

2.  **[Not Started] Refactor Data and Dataloader Configs:**
    -   **Goal:** Decompose the monolithic `configs/preset/datasets/db.yaml` into separate, modular configuration files.
    -   **Action:** Create `configs/data/default.yaml` containing `datasets`, `transforms`, and `collate_fn`.
    -   **Action:** Create `configs/dataloaders/default.yaml` containing the `train_dataloader`, `val_dataloader`, etc., using variables like `${data.batch_size}`.

3.  **[Not Started] Update Base Config Defaults:**
    -   **Goal:** Integrate the new modular configs into the main configuration.
    -   **Action:** Add `data: default` and `dataloaders: default` to the `defaults` list in `configs/base.yaml`.

4.  **[Not Started] Clean Up Experiment Configs:**
    -   **Goal:** Remove redundant, hardcoded settings from the main experiment files.
    -   **Action:** Delete the `dataloaders` section from `configs/train.yaml`, `configs/test.yaml`, and `configs/predict.yaml`.

5.  **[Not Started] Validate All Changes:**
    -   **Goal:** Ensure the refactoring is successful and the training pipeline runs without errors.
    -   **Action:** Execute the training script with appropriate overrides to confirm the new configuration is loaded and works correctly.
