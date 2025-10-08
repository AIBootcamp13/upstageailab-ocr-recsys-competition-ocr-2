# Session Handover: OCR Performance Regression Resolution

**Date:** 2025-10-08
**From:** Previous Debugging Session
**To:** Next Developer/Team Member

## Context Summary

The OCR training pipeline experienced a critical performance regression where validation metrics dropped to hmean=0.000 with "Missing predictions for ground truth" warnings. Through systematic debugging using git bisect, the root cause was identified as commit `bbf30088b941da7a5d8ab7770ebbfdbfeccd99e1` which introduced performance optimization features that interfered with the core model detection pipeline.

## Current Status

### ✅ Resolved Issues
- **Performance Regression Fixed:** Training now achieves hmean ≈ 0.85, meeting baseline requirements (≥0.6 for baseline, ≥0.8 for canonical datasets at epoch 0)
- **Training Stability:** Pipeline completes without crashes
- **Callback Compatibility:** UniqueModelCheckpoint updated for PyTorch Lightning 2.5.5
- **Config Interpolations:** Fixed `${data.batch_size}` → `${batch_size}` issues

### ⚠️ Remaining Issues to Address

#### 1. Missing Prediction GT Label Warnings
**Problem:** Despite good metrics, training logs show numerous warnings:
```
Missing predictions for ground truth file 'drp.en_ko.in_house.selectstar_XXXXX.jpg'
```

**Impact:** Non-blocking (metrics are good), but indicates potential data processing issues

**Investigation Points:**
- Check if warnings correlate with specific image types or preprocessing failures
- Verify polygon coordinate transformations in DBTransforms
- Examine DBCollateFN polygon handling
- Test with num_workers=0 for better error traces
  - Use 1 or 5% of dataset
- Review validation data loading pipeline

**Expected Outcome:** Eliminate warnings while maintaining metric performance

#### 2. Performance Optimization Re-implementation Assessment

**Problem:** Original performance features (5-8x validation speedup) were reverted due to incompatibility

**Features to Re-assess:**
- **Polygon Caching:** Intended to cache polygon computations for faster validation
- **Performance Callbacks:**
  - `resource_monitor`: System resource tracking
  - `throughput_monitor`: Training speed monitoring
  - `profiler`: Performance profiling
- **Component Overrides:** Custom encoder/decoder/head configurations

**Assessment Requirements:**
- **Isolation Testing:** Test each feature individually in clean environment
- **Compatibility Verification:** Ensure no interference with model forward pass
- **Performance Measurement:** Quantify actual speedups vs overhead
- **Integration Strategy:** Determine safe rollout approach (feature flags, gradual enablement)

**Expected Outcome:** Working performance optimizations that don't break core functionality

## Priority Tasks

### High Priority
1. **Fix Missing Predictions Warning**
   - Investigate warning sources
   - Implement fixes without affecting metrics
   - Update validation criteria

### Medium Priority
2. **Performance Features Assessment**
   - Create isolated test environment
   - Test polygon caching implementation
   - Test performance callbacks individually
   - Document compatibility requirements

### Low Priority
3. **Regression Prevention**
   - Add automated tests for performance features
   - Implement config validation
   - Update documentation

## Technical Details

### Code Changes Made
- Reverted commit `bbf30088b941da7a5d8ab7770ebbfdbfeccd99e1`
- Updated `ocr/lightning_modules/callbacks/unique_checkpoint.py` format_checkpoint_name signature
- Fixed dataloader config interpolations in `configs/dataloaders/default.yaml`

### Current Working Config
- Model: DBNet with ResNet18 encoder, FPN decoder, DB head/loss
- Batch size: 12
- Optimizer: AdamW (lr=1e-3, wd=0.0001)
- Training completes successfully with good metrics

### Testing Commands
```bash
# Quick validation test
uv run python runners/train.py trainer.max_epochs=1

# Debug with single worker
uv run python runners/train.py trainer.max_epochs=1 dataloaders.train_dataloader.num_workers=0 dataloaders.val_dataloader.num_workers=0
```

## Debugging Artifacts and Rolling Logs

### Agent Instructions for Debugging

When investigating issues, the agent should generate comprehensive debugging artifacts and maintain rolling logs to ensure reproducible debugging and proper documentation.

#### Required Debugging Artifacts

1. **Debug Log Directory Structure**
   ```
   logs/debugging_sessions/YYYY-MM-DD_HH-MM-SS_debug/
   ├── debug.jsonl                 # Main structured log
   ├── training.log               # Raw training output
   ├── config_dump.yaml           # Full resolved config
   ├── git_status.txt             # Git state snapshot
   ├── system_info.txt            # Environment details
   └── artifacts/                 # Additional files
       ├── profiler_traces/       # Performance traces
       ├── model_outputs/         # Sample predictions
       └── data_samples/          # Input data examples
   ```

2. **Structured Debug Logging**
   - Use JSONL format for machine-readable logs
   - Include timestamps, log levels, and structured data
   - Capture all hypotheses tested and results
   - Document config changes and their effects

3. **Rolling Log Management**
   - Maintain last 10 debug sessions in `logs/debugging_sessions/`
   - Auto-cleanup old sessions after 30 days
   - Compress archived logs to save space

#### Artifact Generation Commands

```bash
# Generate debug session directory
mkdir -p logs/debugging_sessions/$(date +%Y-%m-%d_%H-%M-%S)_debug

# Capture system state
echo "=== Git Status ===" > git_status.txt
git status >> git_status.txt
git log --oneline -10 >> git_status.txt

echo "=== Environment ===" > system_info.txt
python -c "import torch, lightning; print(f'PyTorch: {torch.__version__}'); print(f'Lightning: {lightning.__version__}')" >> system_info.txt
nvidia-smi >> system_info.txt 2>/dev/null || echo "No GPU" >> system_info.txt

# Config dump
python -c "import hydra; from omegaconf import OmegaConf; cfg = hydra.compose(config_name='train'); OmegaConf.save(cfg, 'config_dump.yaml')"

# Performance profiling (if needed)
python -c "
import torch
from torch.profiler import profile, record_function, ProfilerActivity
# Add profiling code here
"
```

#### Log Analysis Commands

```bash
# Search for specific patterns in logs
grep "Missing predictions" logs/debugging_sessions/*/*.jsonl

# Analyze training metrics over time
python -c "
import json
import glob
logs = glob.glob('logs/debugging_sessions/*/*.jsonl')
for log in logs[-5:]:  # Last 5 sessions
    with open(log) as f:
        for line in f:
            data = json.loads(line)
            if 'metrics' in data:
                print(f'{log}: {data[\"metrics\"]}')
"

# Compare configs between sessions
diff <(head -20 logs/debugging_sessions/2025-10-08_15-00-00_debug/config_dump.yaml) \
     <(head -20 logs/debugging_sessions/2025-10-07_10-00-00_debug/config_dump.yaml)
```

## Handover Notes

- Regression was caused by performance features interfering with model detection pipeline
- Revert strategy was effective for immediate fix
- Missing warnings may indicate edge cases in data processing
- Performance features need careful re-implementation with proper testing

## Contact
For questions about this handover, reference the debug log: `logs/debugging_sessions/2025-10-08_15-00-00_debug.jsonl`
