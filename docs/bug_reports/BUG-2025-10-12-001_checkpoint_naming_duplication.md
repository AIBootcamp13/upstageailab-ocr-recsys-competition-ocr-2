## 🐛 Bug Report Template

**Bug ID:** BUG-2025-10-12-001
**Date:** October 12, 2025
**Reporter:** Development Team
**Severity:** High
**Status:** Fixed

### Summary
Checkpoint filenames contain duplicate labels and incorrect epoch numbering, causing automatic deletion of valid checkpoints.

### Environment
- **Pipeline Version:** OCR Training Pipeline
- **Components:** PyTorch Lightning ModelCheckpoint, UniqueModelCheckpoint
- **Configuration:** `filename: "epoch_{epoch:02d}_step_{step:06d}"`, `auto_insert_metric_name: True`

### Steps to Reproduce
1. Run training with the current checkpoint configuration
2. Observe checkpoint filenames have duplicated prefixes
3. Note that all checkpoints show `epoch_00` regardless of training progress

### Expected Behavior
Checkpoint filenames should be `epoch_XX_step_XXXXXX_YYYYMMDD_HHMMSS.ckpt` with correct epoch numbers.

### Actual Behavior
Checkpoint filenames are `epoch_epoch_00_step_step_000103_20251012_025102.ckpt` with epoch always 00.

### Root Cause Analysis
**Duplicate Labels:** The metrics passed to `format_checkpoint_name` contain pre-formatted strings with prefixes ("epoch_00", "step_000103") instead of raw numbers. When the template `"epoch_{epoch:02d}_step_{step:06d}"` is formatted with these strings, it produces `"epoch_epoch_00_step_step_000103"`.

**Incorrect Epoch Labeling:** The epoch counter is not incrementing properly, always remaining at 0.

**Code Path:**
```
1. Trainer calls ModelCheckpoint.format_checkpoint_name with metrics containing formatted strings
2. UniqueModelCheckpoint calls super().format_checkpoint_name, which formats the template with the pre-formatted strings
3. Resulting filename has duplicated labels
```

### Resolution
Replaced the `format_checkpoint_name` method in `UniqueModelCheckpoint` with a robust implementation that manually constructs filenames using the trainer's authoritative state:

```python
def format_checkpoint_name(self, metrics: dict | None = None, filename: str | None = None) -> str:
    """
    Formats the checkpoint name robustly using the trainer's state.
    """
    trainer = getattr(self, "trainer", None)
    if trainer is None:
        return super().format_checkpoint_name(metrics or {}, filename)

    # Get authoritative epoch and step directly from the trainer
    epoch = trainer.current_epoch
    step = trainer.global_step

    # Build the core filename string
    stem = f"epoch_{epoch:02d}_step_{step:06d}"

    # Add the monitored metric value if enabled
    if self.auto_insert_metric_name and metrics and self.monitor:
        metric_val = metrics.get(self.monitor)
        if isinstance(metric_val, torch.Tensor):
            metric_name_clean = self.monitor.replace("/", "_")
            stem = f"{stem}_{metric_name_clean}_{metric_val.item():.4f}"

    # Add unique identifiers (model info, timestamp)
    dirpath = self.dirpath or "."
    is_best_checkpoint = "best" in (filename or "").lower()

    if is_best_checkpoint:
        stem = f"best_{stem}"

    model_info = self._get_model_info()
    if model_info:
        stem = f"{stem}_{model_info}"

    if self.add_timestamp:
        stem = f"{stem}_{self.timestamp}"

    # Combine and return the final path
    final_name = f"{stem}{self.FILE_EXTENSION}"
    return os.path.join(dirpath, final_name)
```

Updated the YAML configuration to use a simplified filename template for best checkpoints:

```yaml
filename: "{val/hmean:.4f}-best"
```

### Testing
- [x] Code compiles without errors
- [x] New `format_checkpoint_name` method implemented
- [x] YAML configuration updated
- [ ] Verify checkpoint names are correct in next training run (expected: `epoch_XX_step_XXXXXX_val_hmean_0.XXXX_modelinfo_YYYYMMDD_HHMMSS.ckpt`)
- [ ] Verify epoch numbers increment properly
- [ ] Verify no checkpoint overwrites occur

### Prevention
- Add validation for metrics passed to format_checkpoint_name
- Ensure epoch counter increments correctly
- Add unit tests for checkpoint naming

### Risk Assessment
High risk: Checkpoint overwrites can cause loss of training progress and make it difficult to resume training from the best checkpoint.

---
