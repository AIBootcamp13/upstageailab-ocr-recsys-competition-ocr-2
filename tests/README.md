# Tests Directory

**Generated:** October 13, 2025
**Last Updated:** October 14, 2025 (OCR Dataset Modular Refactor)
**Test Framework:** pytest 8.4.2
**Coverage:** Comprehensive unit, integration, and regression testing

## Overview

This directory contains the complete test suite for the OCR Receipt Text Detection competition project. The tests are organized by type and scope, providing comprehensive validation of the OCR pipeline from data loading to model inference.

## Test Organization

### Directory Structure

```
tests/
├── unit/                    # Fast, isolated unit tests
├── integration/            # End-to-end pipeline tests
├── smoke/                  # Quick sanity checks
├── performance/            # Performance regression tests
├── regression/             # Bug fix validation tests
├── debug/                  # Debugging utilities and data analysis
├── ui/                     # UI component tests
├── ocr/                    # OCR-specific test organization
│   ├── callbacks/          # Lightning callback tests
│   ├── datasets/           # Dataset implementation tests
│   ├── metrics/            # Evaluation metric tests
│   ├── models/             # Model component tests
│   └── utils/              # OCR utility tests
├── scripts/                # Test automation scripts
├── manual/                 # Manual testing procedures
├── demos/                  # Demonstration tests
├── wandb/                  # Weights & Biases integration tests
├── conftest.py             # pytest fixtures and configuration
└── pytest.ini             # pytest configuration
```

## Test Categories

### 🔬 Unit Tests (`tests/unit/`)

**Purpose:** Test individual components in isolation with mocked dependencies.

**Key Test Files:**
- `test_validation_models.py` - Pydantic v2 data validation models (1,172 lines, 61 tests)
- `test_dataset.py` - Dataset implementations and data loading
- `test_lightning_module.py` - PyTorch Lightning module logic
- `test_metrics.py` - Evaluation metrics and CLEvalMetric integration
- `test_architecture.py` - Model architecture components
- `test_config_utils.py` - Hydra configuration utilities
- `test_image_processor.py` - Image preprocessing and augmentation
- `test_evaluator.py` - Model evaluation pipeline

**Testing Patterns:**
- Comprehensive mocking to avoid file I/O
- Property-based testing with hypothesis
- Edge case validation
- Error condition testing

### 🔗 Integration Tests (`tests/integration/`)

**Purpose:** Test complete workflows and component interactions.

**Key Test Files:**
- `test_ocr_lightning_predict_integration.py` - Full prediction pipeline
- `test_ocr_pipeline_integration.py` - End-to-end OCR processing
- `test_collate_integration.py` - Data collation and batching
- `test_dataloader_batching.py` - DataLoader batch processing
- `test_inference_service.py` - Inference API validation
- `test_exif_orientation_smoke.py` - EXIF orientation handling
- `test_hydra_config_validation.py` - Configuration validation

**Testing Scope:**
- Real file I/O with temporary directories
- Actual model components (not mocked)
- End-to-end data flow validation
- Performance benchmarking

### 🚀 Smoke Tests (`tests/smoke/`)

**Purpose:** Quick sanity checks for critical functionality.

**Key Test Files:**
- `test_command_builder_smoke.py` - Command line interface validation

**Testing Focus:**
- Fast execution (< 1 second per test)
- Critical path validation
- Deployment readiness checks

### 📊 Performance Tests (`tests/performance/`)

**Purpose:** Detect performance regressions and validate optimizations.

**Key Test Files:**
- `test_regression.py` - Performance regression detection
- `baselines/` - Historical performance baselines

**Testing Metrics:**
- Training throughput (images/second)
- Memory usage patterns
- Inference latency
- GPU utilization

### 🔄 Regression Tests (`tests/regression/`)

**Purpose:** Ensure bug fixes remain fixed and prevent regressions.

**Key Test Files:**
- `test_regression_validation_fix.py` - Validation bug fixes

**Testing Approach:**
- Reproduce historical bugs
- Validate fix effectiveness
- Prevent reintroduction of known issues

### 🐛 Debug Tests (`tests/debug/`)

**Purpose:** Development-time debugging utilities and data analysis.

**Key Files:**
- `data_analyzer.py` - Dataset analysis and visualization
- `generate_offline_samples.py` - Test data generation

**Usage:** Not part of CI pipeline, used for development debugging.

### 🎨 UI Tests (`tests/ui/`)

**Purpose:** Test Streamlit UI components and visualization.

**Key Test Files:**
- `test_visualization_helpers.py` - UI helper functions
- `test_visualize_annotations.py` - Annotation visualization

## Test Configuration

### pytest Configuration (`pytest.ini`)

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --strict-config
    --disable-warnings
    --tb=short
    -v
    -W ignore::UserWarning:pkg_resources
    -W ignore::SentryHubDeprecationWarning
    -W ignore::PydanticDeprecatedSince20
    --ignore=logs/
    --ignore=DEPRECATED/
    --ignore=_archive/
    --ignore=_llm_code_gen_workflow_exp/
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    performance: marks tests as performance tests
```

### Test Fixtures (`conftest.py`)

**Global Fixtures:**
- `temp_path` - Temporary directory for file I/O tests
- `sample_image_tensor` - Mock image tensor (1, 3, 224, 224)
- `sample_batch_images` - Batch of images (4, 3, 224, 224)
- `sample_prediction_maps` - Mock prediction maps
- `sample_target_maps` - Mock target maps
- `sample_polygons` - Sample polygon annotations
- `mock_config` - Hydra configuration mock

## Running Tests

### All Tests
```bash
uv run pytest
```

### Specific Categories
```bash
# Unit tests only
uv run pytest tests/unit/

# Integration tests only
uv run pytest tests/integration/

# With markers
uv run pytest -m "unit"
uv run pytest -m "integration"
uv run pytest -m "performance"
```

### Coverage Report
```bash
uv run pytest --cov=ocr --cov-report=html
```

### Performance Tests
```bash
uv run pytest tests/performance/ -m "performance"
```

## Test Development Guidelines

### Unit Test Patterns

1. **Mock External Dependencies**
   ```python
   from unittest.mock import patch, MagicMock

   @patch('ocr.utils.file_utils.Path.exists')
   def test_component_with_file_io(self, mock_exists):
       mock_exists.return_value = True
       # Test logic here
   ```

2. **Use Fixtures for Common Data**
   ```python
   def test_with_sample_data(self, sample_image_tensor, sample_polygons):
       # Test implementation
   ```

3. **Test Edge Cases**
   ```python
   def test_empty_input(self):
       with pytest.raises(ValueError):
           process_empty_data()

   def test_invalid_format(self):
       with pytest.raises(ValidationError):
           PolygonArray(points=invalid_data)
   ```

### Integration Test Patterns

1. **Use Temporary Directories**
   ```python
   def test_full_pipeline(self, temp_path):
       # Create test data in temp_path
       # Run full pipeline
       # Assert results
   ```

2. **Test Realistic Data**
   ```python
   def test_with_real_annotations(self):
       # Use actual annotation format
       # Test parsing and processing
   ```

## Test Evolution History

### October 2025 - Major Refactoring Period

**OCR Dataset Modular Refactor (2025-10-14):**
- Added comprehensive tests for new utility modules
- `test_cache_manager.py` - CacheManager validation (20 tests)
- `test_image_utils.py` - Image processing utilities (15 tests)
- `test_polygon_utils.py` - Polygon validation (14 tests)
- Total: 49 new unit tests for modular refactor

**Data Contract Implementation (2025-10-12):**
- Added 61 comprehensive unit tests for Pydantic v2 models
- Enhanced validation testing across all data boundaries
- Runtime contract validation testing

**OCR Dataset Migration (2025-10-13):**
- Migration from legacy OCRDataset to ValidatedOCRDataset
- Enhanced data validation and error handling tests
- Backward compatibility testing

### Key Testing Principles Established

1. **Comprehensive Mocking Strategy**
   - Avoid file I/O in unit tests
   - Mock external services and APIs
   - Use fixtures for consistent test data

2. **Multi-Layer Testing Approach**
   - Unit tests for isolated components
   - Integration tests for workflows
   - Performance tests for regressions
   - Smoke tests for deployment validation

3. **Data Validation Focus**
   - Extensive Pydantic model testing
   - Edge case validation
   - Error condition handling

4. **CI/CD Integration**
   - Fast unit tests in every commit
   - Integration tests in merge validation
   - Performance tests in nightly builds

## Test Maintenance

### Adding New Tests

1. **Unit Tests:** Place in `tests/unit/test_<component>.py`
2. **Integration Tests:** Place in `tests/integration/test_<workflow>.py`
3. **Follow Naming:** `test_<functionality>.py`
4. **Use Markers:** `@pytest.mark.unit`, `@pytest.mark.integration`

### Updating Fixtures

Add new fixtures to `conftest.py` for common test data patterns.

### Performance Baselines

Update `tests/performance/baselines/` when making performance improvements.

## Troubleshooting

### Common Issues

1. **Import Errors:** Ensure `PYTHONPATH` includes project root
2. **CUDA Errors:** Use `pytest --tb=short` for cleaner output
3. **Memory Issues:** Run tests individually: `pytest tests/unit/test_large_component.py::TestClass::test_method`

### Debug Mode

```bash
# Verbose output
uv run pytest -v -s

# Stop on first failure
uv run pytest --tb=short -x

# Run specific test
uv run pytest tests/unit/test_validation_models.py::TestPolygonArray::test_valid_polygon_array
```

## AI Context Preservation

This test suite was developed with AI assistance and follows modern testing practices:

- **Comprehensive Coverage:** Tests validate both happy paths and error conditions
- **Mocking Strategy:** Extensive use of unittest.mock for isolated testing
- **Fixture Reuse:** Common test data provided through pytest fixtures
- **Marker System:** Tests categorized by type and execution speed
- **Documentation:** Extensive docstrings and comments for maintainability

The test structure reflects the modular architecture of the OCR system, with dedicated test files for each major component and utility module.
