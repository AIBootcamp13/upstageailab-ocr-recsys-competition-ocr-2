## 🐛 Bug Report: Mixed Precision Performance Degradation

**Bug ID:** BUG-2025-002
**Date:** October 14, 2025
**Reporter:** Development Team
**Severity:** High
**Status:** Open

### Summary
Mixed precision training (FP16) causes severe performance degradation in OCR model, with H-mean dropping from 0.8839 to 0.5530 (37% reduction) when using `trainer.precision=16-mixed`.

### Environment
- **Pipeline Version:** Phase 6C
- **Components:** PyTorch Lightning training, DBNet OCR model
- **Configuration:** `trainer.precision=16-mixed`, DBNet with ResNet18 backbone
- **Hardware:** GPU training with CUDA

### Steps to Reproduce
1. Configure training with `trainer.precision=16-mixed`
2. Train DBNet model on OCR dataset for 3 epochs
3. Run validation and observe H-mean metrics
4. Compare with identical configuration using `trainer.precision=32-true`

### Expected Behavior
Mixed precision should maintain similar performance to full precision while providing training speedup, with H-mean degradation <5%.

### Actual Behavior
```python
# Baseline (32-bit precision)
val/hmean: 0.8839

# Mixed precision (16-bit)
val/hmean: 0.5530  # 37% performance drop

# Combined with caching
val/hmean: 0.7816  # 11.6% performance drop (still significant)
```

### Root Cause Analysis
**Numerical Instability in FP16:** Mixed precision training introduces significant numerical errors in:

1. **Polygon coordinate calculations** - DBNet's geometric operations require high precision
2. **Loss function accumulation** - DB loss combines multiple weighted terms that accumulate FP16 errors
3. **Gradient computations** - Reduced precision affects gradient flow and optimization

**Evidence:**
- Diagnostic test shows 37% H-mean drop with FP16 alone
- Individual batch metrics show higher variance and lower scores
- Loss components remain similar, indicating gradient/optimization issues

**Code Path:**
```
trainer.precision=16-mixed
├── Automatic Mixed Precision (AMP) enabled
├── Forward pass: FP16 computations
├── Loss calculation: Accumulates in FP16
├── Backward pass: Gradient scaling issues
└── Validation: Poor convergence, low H-mean
```

### Resolution
**Immediate Fix:**
```yaml
# Disable mixed precision until resolved
trainer:
  precision: 32-true  # Instead of 16-mixed
```

**Long-term Solutions:**
1. **Gradient Scaling:** Implement proper gradient scaling for FP16
2. **Loss Function Tuning:** Adjust DB loss weights for FP16 stability
3. **Selective Precision:** Use FP16 for backbone, FP32 for critical heads
4. **Gradient Monitoring:** Add FP16-specific gradient clipping and monitoring

### Testing
- [x] Baseline performance confirmed (32-bit: 0.8839 H-mean)
- [x] Mixed precision degradation reproduced (16-bit: 0.5530 H-mean)
- [x] Caching impact isolated (minimal additional degradation)
- [ ] Gradient scaling solution implemented
- [ ] Performance recovery verified

### Prevention
- Add precision-specific performance regression tests
- Implement gradient monitoring for mixed precision training
- Document precision requirements for geometric models
- Add automated performance validation in CI/CD pipeline

### Impact Assessment
- **Severity:** High - Core functionality (training performance) severely impacted
- **Scope:** Affects all future training runs using mixed precision
- **Workaround:** Use 32-bit precision (slower but correct)
- **Timeline:** Requires investigation of gradient scaling solutions

### Related Issues
- Performance optimization blocked until resolved
- Mixed precision benefits (2x speedup) cannot be utilized
- May affect other geometric computer vision models in pipeline

---

### Investigation Notes

**Diagnostic Results:**
- Mixed precision alone: -37% H-mean (0.8839 → 0.5530)
- With caching: -11.6% H-mean (0.8839 → 0.7816)
- Speed improvement: +1.1x faster validation (19.23s → 17.63s)

**Affected Components:**
- DBNet model architecture
- DB loss function (prob_map, binary_map, thresh_map losses)
- Polygon coordinate processing
- Gradient-based optimization

**Recommended Next Steps:**
1. Implement gradient scaling with `torch.cuda.amp.GradScaler()`
2. Test selective precision (FP16 backbone, FP32 heads)
3. Monitor gradient norms during FP16 training
4. Consider loss function modifications for FP16 stability</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/bug_reports/BUG_2025_002_MIXED_PRECISION_PERFORMANCE.md
