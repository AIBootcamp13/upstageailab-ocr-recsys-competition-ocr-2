# 2025-10-06 Per Batch Image Logging Configuration

## Summary
Made the per batch image logging feature configurable through external configuration instead of hardcoded values.

## Changes Made

### Configuration Externalization
- **File:** `configs/logger/wandb.yaml`
- **Added:** `per_batch_image_logging` section with `enabled` and `recall_threshold` options
- **Default:** `enabled: true`, `recall_threshold: 0.8`

### Code Updates
- **File:** `ocr/lightning_modules/ocr_pl.py`
- **Modified:** Validation step to use configurable threshold instead of hardcoded `0.8`
- **Logic:** Only logs problematic batch images when both `enabled=true` and `recall < threshold`

- **File:** `runners/test.py`
- **Fixed:** WandB config serialization to properly handle Hydra interpolations
- **Issue:** Config.json not appearing in WandB overview due to serialization failures
- **Solution:** Added try/except block to fall back from `resolve=True` to `resolve=False`

### Documentation
- **File:** `docs/ai_handbook/03_references/06_wandb_integration.md`
- **Added:** Complete "Per Batch Image Logging" section explaining:
  - How the feature works
  - Configuration options
  - Usage examples
  - What gets logged
  - Use cases and performance considerations

## Benefits
- **Flexibility:** Can enable/disable without code changes
- **Tunability:** Adjustable recall threshold for different error analysis needs
- **Environment-specific:** Different settings for dev/prod
- **Backwards compatible:** Maintains previous behavior by default

## Usage
```bash
# Enable with custom threshold
python runners/train.py logger.per_batch_image_logging.recall_threshold=0.7

# Disable completely
python runners/train.py logger.per_batch_image_logging.enabled=false
```
