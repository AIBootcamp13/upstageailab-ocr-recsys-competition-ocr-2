# Polygon Pre-Processing Implementation Plan

This document provides detailed instructions to refactor the data pipeline by replacing ineffective on-the-fly polygon caching with offline pre-processing. This will achieve significant performance gains (estimated 5-8x speedup in validation).

**Important:** This is a major refactoring task requiring architectural changes. It is not recommended for automated agents; execute manually or with expert assistance.

## Prerequisites and Assumptions

- **Dataset Preparation**: Ensure the dataset is fully prepared and accessible at `data/datasets/` before running Phase 1.
- **Environment Setup**: Python environment with Hydra, PyTorch, and other dependencies installed via `uv`.
- **Hardware**: Sufficient disk space for generated `.npz` files (estimated 2-5GB per dataset).
- **Assumptions**: The current `OCRDataset` returns NumPy arrays before `ToTensor` transforms. If not, minor adjustments may be needed in Phase 1.
- **Version Control**: Work on a feature branch (e.g., `feature/offline-preprocessing`) with commits per phase for easy reversion.

## Filenaming Conventions

To maintain consistency and organization, follow these conventions when creating new documents and scripts:

**For Markdown Documents (.md):**
- **Format:** `YYYY-MM-DD_NN_descriptive_name.md` (aligned with Post-Debugging Session Framework)
- `YYYY-MM-DD`: Current date (e.g., 2025-10-08)
- `NN`: Sequential number (01, 02, 03...) within the session or phase
- `descriptive_name`: Clear, specific name without abbreviations
- **Index Assignment Rules:**
  - `00`: Session summary/overview
  - `01-09`: Primary artifacts (scripts, configs)
  - `10-49`: Secondary artifacts (logs, data dumps)
  - `50-79`: Analysis and documentation
  - `80-99`: Archives and deprecated files
- **Directory:** `docs/` (e.g., `2025-10-08_50_preprocessing_guide.md`)

**For Scripts (.py):**
- **Format:** snake_case with descriptive names
- **Directory:** `scripts/` (e.g., `preprocess_maps.py`)
- **Prefix:** No specific prefix unless testing (see below)

**For Test Scripts (.py):**
- **Format:** `test_module_name.py` or `test_feature_name.py`
- **Directory:** `tests/` (e.g., `test_preprocess_maps.py`)

**For Configuration Files (.yaml, .json):**
- **Format:** Use existing patterns in `configs/` subdirectories
- **Examples:** `configs/data/base.yaml`, `configs/model/default.yaml`

**General Guidelines:**
- Use lowercase, underscores for spaces
- Be descriptive and avoid abbreviations
- Avoid special characters except underscores and hyphens
- Follow project structure conventions

## Risk Assessment and Rollback Plan

- **Risks**: Data corruption during pre-processing could break training; shape mismatches in tensors; performance regressions.
- **Mitigation**: Back up original datasets before Phase 1. Test incrementally after each phase.
- **Rollback**: Revert to original caching by restoring deleted files from Git and re-enabling `polygon_cache` in configs.

## Timeline and Milestones

- **Phase 1-4**: Complete in 1-2 weeks, with daily commits and testing.
- **Phases 5-7**: Optional optimizations; evaluate after core refactor.
- **Milestones**: Phase 4 validation passing; full training run with improved metrics.

## Validation and Testing Guidelines

Before and after implementation, use limited datasets for rapid iteration:

- **Dataset Limits**: Use ~500 train samples and ~50 validation samples (from ~3500 train and 400 val total).
- **Epochs**: Limit to 1 epoch for testing.
- **Skip Tests**: Focus on validation; skip full test runs if possible.
- **Expected Performance**: H-mean ≥ 0.3 for 1 epoch with small subsets.
- **Testing Integration**: Add unit tests for new components (e.g., `DBCollateFN`) in `tests/`. Use Qwen Coder in --yolo mode for parallel test generation (see below).

Example train command:
```bash
uv run python runners/train.py trainer.limit_train_batches=1000 trainer.limit_val_batches=100 trainer.max_epochs=1
```

Monitor for cache stats showing low hit rates (e.g., 9-15%) pre-refactor, and faster performance post-refactor.

## Parallel Development with Qwen Coder

To accelerate development, use Qwen Coder in --yolo mode for automated unit test generation. This allows parallel work: develop code while Qwen generates tests.

- **Setup**: Ensure `qwen` is installed and in PATH.
- **Usage**: After creating/modifying code (e.g., scripts or classes), run:
  ```bash
  cat <file_path> | qwen --yolo --prompt "Generate comprehensive unit tests for the provided Python code using pytest. Include edge cases and assertions."
  ```
  Capture output and integrate into `tests/`.
- **Integration Points**: Use after Phase 1.1 (script creation), Phase 2.1-2.2 (dataset/collate changes), and Phase 3.1 (config updates).

-----

### **Phase 1: Create the Offline Pre-processing Script**

**Goal:** Create a standalone script to generate and save the probability and threshold maps for the entire dataset once.

**Effort Estimate:** 2-4 hours.
**Dependencies:** None.
**Risks:** Memory issues if dataset is large; incorrect map generation. Mitigation: Test with subset first. Rollback: Delete generated files.

**Action Item 1.1: Write Script Skeleton and Logic**
Create `scripts/preprocess_maps.py` with error handling and logging.

**Action Item 1.2: Add Validation and Config Handling**
Include sanity checks for `.npz` shapes and Hydra config usage (base: `configs/train.yaml`).

**Action Item 1.3: Test with Sample Data**
Run on a small subset to verify output.

- ✅ 2025-10-08: `uv run python scripts/preprocess_maps.py data.train_num_samples=20 data.val_num_samples=10` now generates `.npz` maps for the sampled training set (stored in `data/datasets/images/train_maps`). Added channel dimensions to saved maps and filtered degenerate polygons to keep PyClipper stable.
- ⚠️ Validation dataset path `data/datasets/images/images_val_canonical` currently resolves to zero images, so the script emits a warning and skips generation. Sync or relink the validation assets before running the full job.
- Root fixes: avoid re-transposing `ToTensorV2` outputs (keep CHW tensors) and update `DBTransforms` to drop polygons with <3 keypoints after Albumentations removes out-of-frame points.
- ✅ 2025-10-08 (train sanity check): Hardened `OCRDataset` to drop degenerate polygons (width/height < 1px or <3 points) after transforms. Short run `uv run python runners/train.py trainer.limit_train_batches=2 trainer.limit_val_batches=1 trainer.max_epochs=1` finishes without the previous QHull warning (metrics still 0.0 on the tiny slice, as expected).

**Qwen Integration:** After 1.1, generate unit tests:
```bash
cat scripts/preprocess_maps.py | qwen --yolo --prompt "Generate unit tests for the preprocess_maps.py script using pytest. Focus on preprocess function and error cases."
```

**Action Item 1.1: Create New File: `scripts/preprocess_maps.py`**
This script will be the new entry point for the heavy computation.


**Action Item 1.2: Ensure `OCRDataset` Can Cooperate**
The script above assumes the `OCRDataset` returns a NumPy array before the `ToTensor` transform. If it doesn't, a small modification might be needed, or the script can handle the conversion. For now, assume it works.
        np.savez_compressed(output_filename, prob_map=prob_map, thresh_map=thresh_map)

@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    # This allows us to run preprocessing for both train and val sets
    print("--- Pre-processing Training Data ---")
    preprocess(cfg.datasets.train_dataset)

    print("n--- Pre-processing Validation Data ---")
    preprocess(cfg.datasets.val_dataset)

    print("nPreprocessing complete.")


if __name__ == "__main__":
    main()

**Action Item 1.2: Ensure `OCRDataset` Can Cooperate**
The script above assumes the `OCRDataset` returns a NumPy array before the `ToTensor` transform. If it doesn't, a small modification might be needed, or the script can handle the conversion. For now, assume it works.

-----

### **Phase 2: Refactor the Data Loading Pipeline**

**Goal:** Modify the `Dataset` and `CollateFN` to use the pre-processed `.npz` files instead of calculating maps on the fly.

**Effort Estimate:** 3-5 hours.
**Dependencies:** Phase 1 completion.
**Risks:** Shape mismatches; missing files. Mitigation: Add error handling for missing `.npz`. Rollback: Restore original methods.

**Action Item 2.1: Update `ocr/datasets/ocr_dataset.py`**
Modify `__getitem__` to load maps and remove polygons. Add fallback if maps missing.

**Action Item 2.2: Simplify `ocr/datasets/db_collate_fn.py`**
Remove cache logic; update `__call__` for pre-loaded maps. Ensure shapes: `[batch_size, 1, H, W]`.

**Action Item 2.3: Delete Obsolete Files**
Remove `polygon_cache.py` and `test_polygon_caching.py`. Commit deletions.

**Qwen Integration:** After 2.1-2.2, generate tests:
```bash
cat ocr/datasets/ocr_dataset.py ocr/datasets/db_collate_fn.py | qwen --yolo --prompt "Generate unit tests for the updated OCRDataset and DBCollateFN classes, focusing on map loading and tensor stacking."
```

**Action Item 2.1: Modify `ocr/datasets/ocr_dataset.py`**
Update the `__getitem__` method to load the `.npz` files.

  * Locate the `__getitem__` method.
  * After loading the image and annotations, **add the following logic**:
      * Construct the path to the `.npz` map file based on the image filename.
      * Load the `.npz` file using `np.load()`.
      * Extract `prob_map` and `thresh_map`.
      * **Remove the `polygons` key from the returned dictionary.**
      * Add the `prob_map` and `thresh_map` to the returned dictionary.

**Action Item 2.2: Simplify `ocr/datasets/db_collate_fn.py`**
This class becomes a simple collator.

  * **Remove** the `cache` argument from `__init__`.
  * **Remove** the `make_prob_thresh_map` method entirely.
  * **Remove** the `distance` method entirely.
  * Modify the `__call__` method:
      * Remove the `polygons` list creation.
      * Remove the entire loop that iterates to create `prob_maps` and `thresh_maps`.
      * Directly get the pre-loaded maps from the batch items.
      * Stack them into tensors.

The new `__call__` method should look roughly like this:

```python
def __call__(self, batch):
    images = [item["image"] for item in batch]
    # ... other metadata ...
    prob_maps = [torch.from_numpy(item["prob_map"]) for item in batch]
    thresh_maps = [torch.from_numpy(item["thresh_map"]) for item in batch]

    collated_batch = OrderedDict(
        images=torch.stack(images, dim=0),
        # ... other metadata ...
        prob_maps=torch.stack(prob_maps, dim=0).unsqueeze(1), # Add channel dim
        thresh_maps=torch.stack(thresh_maps, dim=0).unsqueeze(1), # Add channel dim
    )
    return collated_batch
```

**Action Item 2.3: Delete Obsolete Caching Files**
The on-the-fly caching mechanism is now fully replaced.

  * **Delete** the file `ocr/datasets/polygon_cache.py`.
  * **Delete** the test file `tests/performance/test_polygon_caching.py`.

-----

### **Phase 3: Update Configuration and Documentation**

**Goal:** Align configuration files with the new data pipeline and document the new workflow.

**Effort Estimate:** 1-2 hours.
**Dependencies:** Phases 1-2.
**Risks:** Config errors breaking training. Mitigation: Validate configs with dry run. Rollback: Restore backups.

**Action Item 3.1: Clean Up Hydra Configuration**
Remove `polygon_cache` sections and update `collate_fn`. Check all configs for references.

**Action Item 3.2: Create Documentation**
Write `docs/preprocessing_guide.md` with usage, troubleshooting, and examples. Update `README.md` and add changelog.

**Qwen Integration:** After 3.1, generate config tests:
```bash
cat configs/data/base.yaml | qwen --yolo --prompt "Generate validation tests for the updated data config, ensuring no missing keys and correct types."
```

**Action Item 3.1: Clean Up Hydra Configuration (`.yaml` files)**

  * In `configs/data/base.yaml` and any other data configs, **remove the entire `polygon_cache` section**.
  * In `configs/data/base.yaml`, ensure the `collate_fn.cache` argument is removed. It should look like this:
    ```yaml
    collate_fn:
      _target_: ocr.datasets.DBCollateFN
      shrink_ratio: 0.4
      thresh_min: 0.3
      thresh_max: 0.7
    ```
  * **Delete** the file `configs/data/cache.yaml`.

**Action Item 3.2: Create New Documentation**
Create a file `docs/preprocessing_guide.md` to explain the new workflow to future developers.

````markdown
# Data Pre-processing Guide

## Overview

To accelerate training and validation, the project uses an offline pre-processing step to generate the probability and threshold maps required by the DBNet model. This avoids calculating these maps on-the-fly for every batch, which was a major performance bottleneck.

This process must be run once after the dataset is prepared.

## How to Run Pre-processing

The pre-processing logic is handled by the `scripts/preprocess_maps.py` script. It uses the project's Hydra configuration to ensure that all data transformations are consistent with the training and validation pipelines.

To run the script, execute the following command from the root of the project:

```bash
uv run python scripts/preprocess_maps.py
```

## How It Works

The script performs the following steps for both the training and validation datasets:

1.  **Initializes** the `OCRDataset` and `DBCollateFN` using the main `config.yaml`.
2.  **Creates Output Directories**: It creates new directories (e.g., `data/datasets/images_train_maps/`) to store the generated map files.
3.  **Iterates Through Dataset**: It loops through every sample in the dataset.
4.  **Generates Maps**: For each sample, it calls the `make_prob_thresh_map` method to generate the `prob_map` and `thresh_map`.
5.  **Saves Maps**: The generated NumPy arrays are saved to a compressed `.npz` file, named to correspond with the original image file (e.g., `image_001.npz`).

During training and validation, the modified `OCRDataset` class now loads these `.npz` files directly, bypassing the expensive on-the-fly computation.

## Troubleshooting

- **Memory Issues**: If pre-processing fails due to RAM, reduce batch size in the script or process subsets.
- **Missing Files**: Ensure datasets are prepared; check paths in configs.
- **Shape Errors**: Verify map shapes match expectations (e.g., prob_map: [1, H, W]).

## Expected Output

Post-preprocessing, directories like `data/datasets/images_train_maps/` will contain `.npz` files. Training should show faster validation times.
````````markdown
# docs/preprocessing_guide.md

# Data Pre-processing Guide

## Overview

To accelerate training and validation, the project uses an offline pre-processing step to generate the probability and threshold maps required by the DBNet model. This avoids calculating these maps on-the-fly for every batch, which was a major performance bottleneck.

This process must be run once after the dataset is prepared.

## How to Run Pre-processing

The pre-processing logic is handled by the `scripts/preprocess_maps.py` script. It uses the project's Hydra configuration to ensure that all data transformations are consistent with the training and validation pipelines.

To run the script, execute the following command from the root of the project:

```bash
uv run python scripts/preprocess_maps.py
````

---

### **Phase 4: Validation Plan**

**Goal:** Verify that the new pipeline works correctly and achieves the expected performance improvement.

**Effort Estimate:** 2-3 hours.
**Dependencies:** Phases 1-3.
**Risks:** Performance regressions. Mitigation: Compare metrics quantitatively. Rollback: Revert commits.

**Action Item 4.1: Run Pre-processing and Test**
Execute script and run limited training. Ensure no errors.

**Action Item 4.2: Conduct Benchmark**
Measure epoch time, CPU usage, H-mean. Use profiling tools. Expect 5-8x speedup.

**Action Item 4.3: Regression Test**
Compare predictions on held-out set to ensure accuracy.

**Action Item 4.1: Run the Pre-processing Script**
Execute the new script to generate the map files for the training and validation sets.

```bash
uv run python scripts/preprocess_maps.py
```

Verify that the `_maps` directories are created and populated with `.npz` files.

**Action Item 4.2: Run an End-to-End Test**
Perform a short training run to ensure the new data pipeline is wired correctly and the model trains without errors.

```bash
uv run python runners/train.py trainer.limit_train_batches=500 trainer.limit_val_batches=50 trainer.max_epochs=1
```

**Success Criterion:** The run completes without any shape mismatches or data loading errors. Expect H-mean ≥ 0.3.

**Action Item 4.3: Conduct a Performance Benchmark**
Run a validation epoch with limited data to measure the performance gain.

```bash
uv run python runners/train.py trainer.limit_train_batches=500 trainer.limit_val_batches=50 trainer.max_epochs=1
```

Compare epoch time and H-mean against pre-refactor runs. Expect 5-8x speedup and improved metrics.

### **Phase 5: Parallelize Pre-processing (Tier 1 Optimization)**

**Goal:** Accelerate the offline pre-processing step using multi-core parallelism.

**Estimated Additional Gain:** Reduces pre-processing time from hours to minutes.

**Action Item 5.1: Modify `scripts/preprocess_maps.py`**
Implement multiprocessing to process multiple images simultaneously.

- Use `multiprocessing.Pool` with `cpu_count()` workers.
- Create a global dataset and config for workers to avoid pickling issues.
- Process training and validation sets in parallel.

**Success Criterion:** Pre-processing completes in a fraction of the original time.

### **Phase 6: Implement WebDataset Format or RAM Caching (Tier 2 Optimization)**

**Goal:** Choose between WebDataset for scalable I/O or full RAM caching for small datasets.

**Estimated Additional Gain:** 2-5x faster data loading.

**Option A: WebDataset (for large datasets or cloud training)**
- **Action Item 6.1A:** Install WebDataset: `pip install webdataset`
- **Action Item 6.2A:** Create `scripts/convert_to_webdataset.py` to pack images and maps into `.tar` files.
- **Action Item 6.3A:** Update `OCRDataset` to use `webdataset.WebDataset`.

**Option B: Full RAM Caching (if dataset fits in RAM)**
- **Action Item 6.1B:** Add `cache_in_ram: true` to `configs/data/base.yaml`.
- **Action Item 6.2B:** Modify `OCRDataset.__init__` to load all data into `self.data` list if `cache_in_ram=True`.
- **Action Item 6.3B:** Update `__getitem__` to return `self.data[idx]`.

**Decision Criteria:** Use RAM caching if total dataset size < 32GB RAM. Otherwise, use WebDataset.

**Success Criterion:** Data loading becomes near-instantaneous.

### **Phase 7: Integrate NVIDIA DALI (Tier 3 Optimization)**

**Goal:** Offload data loading and augmentation to GPU for maximum throughput.

**Estimated Additional Gain:** Eliminates CPU bottlenecks, maximizes GPU utilization.

**Action Item 7.1: Install DALI**
Add `nvidia-dali-cuda110` or appropriate version to dependencies.

**Action Item 7.2: Create DALI Pipeline**
Implement a DALI pipeline to handle reading, decoding, and augmenting data on GPU.

**Action Item 7.3: Replace DataLoader**
Use `DALI DALIGenericIterator` instead of PyTorch DataLoader.

**Action Item 7.4: Reimplement Transforms**
Convert `albumentations` transforms to DALI operators.

**Success Criterion:** CPU usage drops significantly during training, with higher throughput.

**Note:** This is the most complex phase. Start with earlier phases and validate before proceeding.

## Tools and Automation

- **Dependency Management:** Use `uv` for packages.
- **Testing:** `pytest` for unit tests; generate with Qwen as above.
- **Profiling:** `cProfile` or PyTorch profiler for benchmarks.
- **CI/CD:** Consider adding pre-processing to GitHub Actions.

## Final Notes

This refined plan is more actionable with modular steps, risk mitigation, and parallel development via Qwen. Focus on Phases 1-4 first for core benefits. If issues arise, iterate with small fixes.

**Action Item 7.2: Create DALI Pipeline**
Implement a DALI pipeline to handle reading, decoding, and augmenting data on GPU.

**Action Item 7.3: Replace DataLoader**
Use `DALI DALIGenericIterator` instead of PyTorch DataLoader.

**Action Item 7.4: Reimplement Transforms**
Convert `albumentations` transforms to DALI operators.

**Success Criterion:** CPU usage drops significantly during training, with higher throughput.

**Note:** This is the most complex phase. Start with earlier phases and validate before proceeding.
