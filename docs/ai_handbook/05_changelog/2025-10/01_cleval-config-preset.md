# 2025-10-01 CLEval Config Preset

## Summary
Hydra now manages the CLEval metric defaults through `configs/metrics/cleval.yaml`, enabling command-line overrides instead of source edits.

## Changes Made

### **Configuration**
- Added a `metrics` config group with a `cleval` preset that exposes the full constructor signature.
- Updated the Lightning module factory to inject the configured metric into `OCRPLModule` and pass the same options to the parallel evaluator workers.

### **Documentation**
- Refreshed the evaluation reference to highlight the new Hydra-driven knobs and demonstrate CLI overrides.

### **Testing**
- Ran `pytest tests/test_metrics.py tests/test_lightning_module.py` to confirm metric behaviour and Lightning wiring remain stable.

## Impact
- Teams can toggle case sensitivity, granularity penalties, or scale-wise reporting from the command line without editing `ocr/lightning_modules/ocr_pl.py`.
- Batch evaluation now mirrors the configured settings, making offline analyses consistent with training logs.

## Next Steps
- Consider adding curated presets (e.g., `metrics: cleval_scale_wise`) for common evaluation scenarios.
