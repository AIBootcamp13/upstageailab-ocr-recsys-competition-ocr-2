# 2025-10-01 Evaluation Metrics Doc Refresh

## Summary
Clarified how CLEval is wired into the training stack and documented the configurable knobs exposed by `ocr.metrics.cleval_metric.CLEvalMetric`. The reference now spells out the metric outputs, scale-wise options, and the supporting utilities that feed leaderboard submissions.

## Changes Made

### **Documentation**
- Expanded `docs/ai_handbook/03_references/04_evaluation_metrics.md` with implementation tables, configuration defaults, and practical usage guidance.
- Added inline example code for ad-hoc CLEval runs and pointed readers to the existing `tests/test_metrics.py` safety net.

### **Testing**
- Documentation-only update; no automated tests were executed for this change.

## Impact
- Engineers now have a single reference explaining how to tune CLEval penalties, enable scale-wise reporting, and interpret the additional counters exposed by the metric.
- Lowers ramp-up time for contributors who need to debug evaluation discrepancies or extend the metric for future competitions.

## Next Steps
- Consider exposing CLEval options through Hydra presets so experimentation can happen configuration-first.
- Audit the evaluation orchestration docs in `04_experiments/` to ensure they reference the refreshed metric guidance.
