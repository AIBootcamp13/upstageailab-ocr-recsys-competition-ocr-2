## 🐛 Bug Report: Run ID Confusion in Performance Benchmarking

**Bug ID:** BUG-2025-003
**Date:** October 14, 2025
**Reporter:** Development Team
**Severity:** Medium
**Status:** Fixed

### Summary
Performance benchmarking script incorrectly compared wrong runs due to run ID confusion, leading to initial misinterpretation of optimization effectiveness.

### Environment
- **Pipeline Version:** Phase 6C
- **Components:** Performance benchmarking scripts, WandB run comparison
- **Configuration:** Benchmark comparison between baseline and optimized runs
- **Tools:** `compare_baseline_vs_optimized.py` script

### Steps to Reproduce
1. Run performance benchmark with mixed precision + caching optimizations
2. Run baseline benchmark with 32-bit precision + no caching
3. Compare runs using script with potentially swapped run IDs
4. Observe that "optimized" run appears slower and less accurate than expected

### Expected Behavior
Benchmark comparison should correctly identify which run is baseline vs optimized and report accurate performance differences.

### Actual Behavior
```python
# Initial incorrect comparison (run IDs swapped)
Baseline run (9evam0xb): precision=32-true, hmean=0.8839
Optimized run (b1bipuoz): precision=16-mixed, hmean=0.7816

# Actually should be:
Baseline run (9evam0xb): precision=32-true, hmean=0.8839  ✅
Optimized run (b1bipuoz): precision=16-mixed, hmean=0.7816 ❌ (worse than baseline)
```

### Root Cause Analysis
**Run ID Mislabeling:** Performance benchmarking relied on manual run ID assignment without validation of actual configurations:

- **Run 9evam0xb**: Actually baseline (32-bit, no caching) - correctly labeled
- **Run b1bipuoz**: Actually optimized (16-bit, caching) but performed worse due to mixed precision issues

**Code Path:**
```
User provides run IDs manually
├── Script fetches WandB runs
├── No validation of run configurations
├── Incorrect assumptions about optimization settings
└── Misleading performance comparison results
```

### Resolution
```python
# Fixed in compare_baseline_vs_optimized.py
def validate_run_configurations(baseline_metrics, optimized_metrics):
    """Validate that runs match expected baseline vs optimized configurations."""
    baseline_config = baseline_metrics['config']['trainer']['precision']
    optimized_config = optimized_metrics['config']['trainer']['precision']

    if baseline_config == '16-mixed' and optimized_config == '32-true':
        print("⚠️  Run IDs appear to be swapped!")
        return False
    return True
```

### Testing
- [x] Run configuration validation added to comparison script
- [x] Correct run ID assignment verified
- [x] Performance comparison now accurate
- [x] Mixed precision degradation properly identified

### Prevention
- Add automatic configuration validation in benchmarking scripts
- Implement run metadata verification before comparisons
- Document expected configuration patterns for baseline vs optimized runs
- Add checksums or hashes of critical config parameters for validation</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/bug_reports/BUG_2025_003_RUN_ID_CONFUSION.md
