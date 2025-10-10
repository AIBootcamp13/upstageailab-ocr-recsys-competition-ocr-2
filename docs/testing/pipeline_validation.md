# Testing Strategy: Pipeline Validation Guide

**Purpose**: Comprehensive testing strategy to validate data contracts and prevent regression bugs in the OCR pipeline.

**Audience**: Developers implementing new features, QA engineers, and CI/CD pipelines.

---

## 📋 Testing Pyramid

```
┌─────────────────────────────────┐
│ End-to-End Integration Tests    │ ← Catch contract violations
│ (test_collate_integration.py)   │
├─────────────────────────────────┤
│ Component Integration Tests     │ ← Validate component interactions
│ (test_transform_pipeline_*.py)  │
├─────────────────────────────────┤
│ Unit Tests with Contracts       │ ← Validate individual functions
│ (test_*_contracts.py)          │
├─────────────────────────────────┤
│ Shape/Type Validation Tests     │ ← Prevent shape mismatches
│ (test_*_shapes.py)             │
└─────────────────────────────────┘
```

---

## 🧪 Test Categories

### 1. Data Contract Validation Tests

**Purpose**: Ensure all pipeline stages maintain data contracts.

**Files**:
- `tests/ocr/datasets/test_polygon_filtering.py` - Polygon shape contracts
- `tests/integration/test_collate_integration.py` - End-to-end contracts
- `tests/ocr/datasets/test_transform_pipeline_contracts.py` - Transform contracts

**Coverage**:
- ✅ Input validation for each pipeline stage
- ✅ Output format verification
- ✅ Shape consistency across stages
- ✅ Type safety (numpy vs torch, PIL vs numpy)

### 2. Edge Case and Error Handling Tests

**Purpose**: Test system robustness with invalid or edge case inputs.

**Test Cases**:
- Empty polygon lists
- Images with no text regions
- Corrupted map files
- Variable batch sizes
- Extreme image dimensions
- Invalid polygon shapes

### 3. Performance Regression Tests

**Purpose**: Detect performance degradation from contract violations.

**Metrics**:
- Transform throughput (> 100 images/sec)
- Memory usage (< 8GB per GPU)
- Training stability (loss convergence)
- Inference latency (< 50ms per batch)

---

## 🔄 CI/CD Integration

### Pre-commit Hooks

```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-data-contracts
        name: Validate Data Contracts
        entry: python scripts/validate_pipeline_contracts.py
        language: system
        files: \.(py)$
        pass_filenames: false

      - id: run-contract-tests
        name: Run Contract Tests
        entry: python -m pytest tests/ -k "contract" -x
        language: system
        files: \.(py)$
        pass_filenames: false
```

### GitHub Actions Workflow

```yaml
# .github/workflows/pipeline-validation.yml
name: Pipeline Contract Validation

on:
  push:
    paths:
      - 'ocr/datasets/**'
      - 'ocr/models/**'
      - 'tests/**'

jobs:
  validate-contracts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run contract tests
        run: |
          python -m pytest tests/ -k "contract" --tb=short
      - name: Validate data contracts
        run: |
          python scripts/validate_pipeline_contracts.py
```

---

## 🛠️ Testing Utilities

### Contract Validation Script

```python
# scripts/validate_pipeline_contracts.py
import torch
from ocr.datasets import OCRDataset, DBCollateFN
from ocr.datasets.transforms import DBTransforms

def validate_full_pipeline():
    """Validate complete pipeline maintains contracts."""
    # Create test dataset
    dataset = OCRDataset(...)

    # Test dataset contract
    sample = dataset[0]
    validate_dataset_contract(sample)

    # Test transform contract
    transformed = transforms(sample)
    validate_transform_contract(transformed)

    # Test collate contract
    batch = collate_fn([transformed])
    validate_collate_contract(batch)

    print("✅ All contracts validated")

if __name__ == "__main__":
    validate_full_pipeline()
```

### Shape Debugging Helper

```python
# debug/shape_debugger.py
def debug_tensor_shapes(data, prefix=""):
    """Print tensor shapes for debugging."""
    if isinstance(data, dict):
        for key, value in data.items():
            debug_tensor_shapes(value, f"{prefix}.{key}")
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            debug_tensor_shapes(item, f"{prefix}[{i}]")
    elif hasattr(data, 'shape'):
        print(f"{prefix}: {data.shape}")
    else:
        print(f"{prefix}: {type(data)}")
```

---

## 📊 Test Coverage Metrics

### Required Coverage

- **Data Contract Tests**: > 95% of contract rules
- **Error Handling**: All documented error conditions
- **Edge Cases**: All boundary conditions
- **Integration Paths**: All pipeline combinations

### Coverage Report

```bash
# Generate coverage report
python -m pytest tests/ --cov=ocr/ --cov-report=html

# Check contract-specific coverage
python -m pytest tests/ -k "contract" --cov=ocr/ --cov-report=term-missing
```

---

## 🚨 Failure Analysis

### Common Test Failures

1. **Shape Mismatch Errors**
   ```
   ValueError: Target size (2, 1, 224, 224) must be the same as input size (2, 1, 896, 896)
   ```
   **Cause**: Head upscale factor not matched in ground truth
   **Fix**: Update ground truth dimensions: `gt = F.interpolate(gt, scale_factor=4)`

2. **Channel Count Errors**
   ```
   RuntimeError: expected input[2, 3, 224, 224] to have 256 channels, but got 3
   ```
   **Cause**: Raw images passed to head instead of decoder features
   **Fix**: Use full pipeline or mock decoder output

3. **Type Errors**
   ```
   AttributeError: 'Image' object has no attribute 'shape'
   ```
   **Cause**: PIL Image where numpy array expected
   **Fix**: Convert with `np.array(pil_image)`

### Debugging Workflow

1. **Identify Failure Point**
   - Run with `--tb=long` for full traceback
   - Check tensor shapes: `print(x.shape for x in [input1, input2, ...])`

2. **Validate Contracts**
   - Run `scripts/validate_pipeline_contracts.py`
   - Check against `docs/pipeline/data_contracts.md`

3. **Isolate Component**
   - Test individual pipeline stages
   - Use mock data to isolate issues

4. **Fix and Retest**
   - Apply contract-compliant fix
   - Run full test suite
   - Update contracts if needed

---

## 📈 Continuous Improvement

### Test Evolution

- **Add tests** for new features that affect data flow
- **Update contracts** when data formats change
- **Expand coverage** for uncovered edge cases
- **Performance benchmarks** for regression detection

### Quality Gates

- [ ] All contract tests pass
- [ ] No shape-related runtime errors
- [ ] Performance within acceptable ranges
- [ ] Documentation updated for changes

---

**Last Updated**: October 11, 2025
**Version**: 1.0</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/testing/pipeline_validation.md
