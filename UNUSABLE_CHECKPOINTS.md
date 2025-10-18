# Unusable Checkpoints Report

**Generated on:** October 18, 2025
**Total Checkpoints Scanned:** 42
**Usable Checkpoints:** 24 (57.1%)
**Unusable Checkpoints:** 18 (42.9%)

## Summary

This document lists checkpoints that cannot be used for inference due to missing or incompatible configuration files. These checkpoints were created before the filesystem refactoring that saves resolved configurations alongside checkpoint files.

### Why These Checkpoints Are Unusable

The inference system requires either:
1. A `.config.json` file alongside the checkpoint (new filesystem refactoring approach)
2. Legacy configuration files in expected locations (Hydra `.hydra/config.yaml` or project-level configs)

These checkpoints lack both, making them incompatible with the current inference pipeline.

### Migration Status

- **24 checkpoints** were successfully migrated and now have `.config.json` files
- **18 checkpoints** could not be migrated due to missing source configuration files
- Migration script: `scripts/agent_tools/generate_checkpoint_configs.py`

## Unusable Checkpoints

### October 17-18, 2025 (Recent Training Runs)

| Checkpoint Path | Size (MB) | Created | Type |
|----------------|-----------|---------|------|
| `outputs/ocr_training-dbnet-pan_decoder-mobilenetv3_small_050-unknown_training_20251017_215423/checkpoints/last.ckpt` | 202.3 | 2025-10-17 22:00:16 | Last checkpoint |
| `outputs/ocr_training-dbnet-pan_decoder-mobilenetv3_small_050-unknown_training_20251017_215423/checkpoints/best.ckpt` | 202.3 | 2025-10-17 22:00:15 | Best checkpoint |
| `outputs/ocr_training-dbnet-pan_decoder-mobilenetv3_small_050_33-unknown_training_20251018_023219/checkpoints/best-v2.ckpt` | 202.3 | 2025-10-18 04:23:25 | Best checkpoint v2 |
| `outputs/ocr_training-dbnet-pan_decoder-mobilenetv3_small_050_33-unknown_training_20251018_023219/checkpoints/last.ckpt` | 202.3 | 2025-10-18 04:29:34 | Last checkpoint |
| `outputs/ocr_training-dbnet-pan_decoder-mobilenetv3_small_050_33-unknown_training_20251018_023219/checkpoints/best.ckpt` | 202.3 | 2025-10-18 04:29:33 | Best checkpoint |
| `outputs/ocr_training-dbnet-pan_decoder-mobilenetv3_small_050_33-unknown_training_20251018_023219/checkpoints/best-v1.ckpt` | 202.3 | 2025-10-18 03:59:06 | Best checkpoint v1 |
| `outputs/debug_mobilenetv3_training-unknown_training_20251017_230640/checkpoints/best-v2.ckpt` | 20.9 | 2025-10-17 23:06:51 | Best checkpoint v2 |
| `outputs/debug_mobilenetv3_training-unknown_training_20251017_230640/checkpoints/last.ckpt` | 20.9 | 2025-10-17 23:07:00 | Last checkpoint |
| `outputs/debug_mobilenetv3_training-unknown_training_20251017_230640/checkpoints/best.ckpt` | 20.9 | 2025-10-17 23:06:45 | Best checkpoint |
| `outputs/debug_mobilenetv3_training-unknown_training_20251017_230640/checkpoints/best-v1.ckpt` | 20.9 | 2025-10-17 23:06:48 | Best checkpoint v1 |

### October 15, 2025 (Earlier Training Runs)

| Checkpoint Path | Size (MB) | Created | Type |
|----------------|-----------|---------|------|
| `outputs/test_full_commit_2c6f1f1-unknown_training_20251015_233734/checkpoints/last.ckpt` | 188.8 | 2025-10-15 23:38:50 | Last checkpoint |
| `outputs/test_full_commit_2c6f1f1-unknown_training_20251015_233734/checkpoints/best.ckpt` | 188.8 | 2025-10-15 23:38:50 | Best checkpoint |
| `outputs/test_data_format_change-unknown_training_20251015_231857/checkpoints/last.ckpt` | 188.8 | 2025-10-15 23:20:13 | Last checkpoint |
| `outputs/test_data_format_change-unknown_training_20251015_231857/checkpoints/best.ckpt` | 188.8 | 2025-10-15 23:20:13 | Best checkpoint |
| `outputs/test_ocr_pl_revert-unknown_training_20251015_233406/checkpoints/last.ckpt` | 188.8 | 2025-10-15 23:35:22 | Last checkpoint |
| `outputs/test_ocr_pl_revert-unknown_training_20251015_233406/checkpoints/best.ckpt` | 188.8 | 2025-10-15 23:35:22 | Best checkpoint |
| `outputs/test_baseline_working-unknown_training_20251015_231508/checkpoints/last.ckpt` | 188.8 | 2025-10-15 23:16:25 | Last checkpoint |
| `outputs/test_baseline_working-unknown_training_20251015_231508/checkpoints/best.ckpt` | 188.8 | 2025-10-15 23:16:25 | Best checkpoint |

## Recommendations

### For Future Training
- All new training runs will automatically create usable checkpoints with `.config.json` files
- The filesystem refactoring ensures inference compatibility

### For Existing Unusable Checkpoints
- **Option 1**: Delete these checkpoints to free up disk space (~3.8 GB total)
- **Option 2**: Keep for reference but exclude from inference UI
- **Option 3**: Manual migration if source configs can be located

### Disk Space Impact
- **Unusable checkpoints**: ~3.8 GB across 18 files
- **Usable checkpoints**: ~4.2 GB across 24 files (including .config.json overhead)
- **Total checkpoint storage**: ~8.0 GB

## Technical Details

### Migration Failure Reasons
- Missing Hydra `.hydra/config.yaml` directories
- No project-level configuration files in expected locations
- Incompatible config formats from early development stages

### File Format
- All unusable checkpoints are standard PyTorch Lightning `.ckpt` files
- They contain model weights but lack configuration metadata
- Inference requires knowing model architecture, preprocessing settings, etc.

### Resolution Status
- ✅ **Inference engine updated** to prioritize `.config.json` files
- ✅ **Checkpoint catalog updated** for new config resolution
- ✅ **Migration script available** for future use
- ✅ **Backward compatibility maintained** for legacy configs where available</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/UNUSABLE_CHECKPOINTS.md
