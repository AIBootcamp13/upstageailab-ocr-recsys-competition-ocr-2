# Refactor Plan (Draft, Unimplemented due to time)

## Executive Summary

This document outlines a comprehensive plan to refactor the current OCR framework from its existing structure to a modern, plug-and-play architecture. The refactoring will enable easy experimentation with different OCR architectures while maintaining clean, maintainable code.

## Current State Analysis

### Existing Structure (DEPRECATED)
```
project_root/
├── ocr/                          # Current implementation
│   ├── __init__.py
│   ├── datasets/
│   ├── lightning_modules/
│   ├── metrics/
│   ├── models/
│   └── utils/
├── configs/                      # Configuration files
├── data/                         # Data and datasets
├── runners/                      # Training/inference scripts
└── pyproject.toml               # Project configuration
```

### Issues Identified
1. **Tight Coupling**: Components are tightly coupled, making it difficult to swap architectures
2. **No Abstract Interfaces**: Missing abstract base classes for component interchangeability
3. **Configuration Complexity**: Hydra configs are scattered and not well-organized
4. **Testing Gaps**: Limited test coverage, especially for integration scenarios
5. **Import Issues**: Relative imports and circular dependencies
6. **No Registry System**: No centralized way to register and discover components

## Target Architecture

### New Structure (src layout)
```
src/
├── ocr_framework/
│   ├── __init__.py
│   ├── core/                     # Abstract base classes
│   │   ├── __init__.py
│   │   ├── base_encoder.py
│   │   ├── base_decoder.py
│   │   ├── base_head.py
│   │   ├── base_loss.py
│   │   └── base_metric.py
│   ├── architectures/            # Architecture implementations
│   │   ├── __init__.py
│   │   ├── registry.py           # Component registry
│   │   ├── dbnet/
│   │   │   ├── __init__.py
│   │   ├── east/
│   │   │   ├── __init__.py
│   │   └── craft/
│   │       ├── __init__.py
│   ├── models/                   # Model assembly
│   │   ├── __init__.py
│   │   ├── factory.py            # Model factory
│   │   └── composite_model.py    # Composite model
│   ├── datasets/                 # Data handling
│   │   ├── __init__.py
│   │   ├── base_dataset.py
│   │   ├── transforms/
│   │   └── collate_fns/
│   ├── training/                 # Training components
│   │   ├── __init__.py
│   │   ├── lightning_modules/
│   │   └── callbacks/
│   ├── evaluation/               # Metrics and evaluation
│   │   ├── __init__.py
│   │   ├── metrics/
│   │   └── evaluators/
│   ├── utils/                    # Utilities
│   │   ├── __init__.py
│   │   ├── logging/
│   │   ├── visualization/
│   │   └── config_utils.py
│   └── config/                   # Configuration management
│       ├── __init__.py
│       ├── schemas/
│       └── validators/
├── tests/                        # Comprehensive test suite
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   ├── manual/
│   ├── debug/
│   └── wandb/
├── configs/                      # Hydra configurations
│   ├── architectures/
│   ├── components/
│   ├── datasets/
│   ├── experiments/
│   └── defaults.yaml
├── scripts/                      # Utility scripts
├── outputs/                      # Experiment outputs
├── docs/                         # Documentation
└── pyproject.toml               # Updated project config
```

## Phase Breakdown

### Phase 1: Foundation (Week 1-2)
**Goal**: Establish core abstractions and project structure

#### Tasks:
1. **Create src layout structure**
   - Create `src/ocr_framework/` directory
   - Set up all subdirectories
   - Update `pyproject.toml` with new source layout

2. **Implement abstract base classes**
   - `BaseEncoder` with abstract methods
   - `BaseDecoder` with abstract methods
   - `BaseHead` with abstract methods
   - `BaseLoss` with abstract methods
   - `BaseMetric` with abstract methods

3. **Create component registry**
   - `ArchitectureRegistry` class
   - Registration methods for all component types
   - Discovery methods for component lookup

4. **Set up configuration management**
   - Reorganize Hydra configs into logical groups
   - Create configuration validators
   - Implement configuration schemas

#### Deliverables:
- ✅ New directory structure
- ✅ Abstract base classes
- ✅ Component registry
- ✅ Reorganized configurations

### Phase 2: Component Migration (Week 3-4)
**Goal**: Migrate existing components to new architecture

#### Tasks:
1. **Migrate DBNet components**
   - Refactor `TimmBackbone` to inherit from `BaseEncoder`
   - Refactor `UNetDecoder` to inherit from `BaseDecoder`
   - Refactor `DBNetHead` to inherit from `BaseHead`
   - Refactor loss functions to inherit from `BaseLoss`

2. **Update model assembly**
   - Create `CompositeModel` class
   - Implement `ModelFactory` for component instantiation
   - Update `OCRLightningModule` to use new architecture

3. **Migrate data components**
   - Refactor `OCRDataset` to new structure
   - Update transforms and collate functions
   - Ensure compatibility with new model interface

4. **Update metrics and evaluation**
   - Refactor `CLEvalMetric` to inherit from `BaseMetric`
   - Update evaluation pipeline
   - Maintain backward compatibility

#### Deliverables:
- ✅ Migrated DBNet components
- ✅ Updated model assembly
- ✅ New data pipeline
- ✅ Updated evaluation system

### Phase 3: Testing Infrastructure (Week 5-6)
**Goal**: Implement comprehensive testing suite

#### Tasks:
1. **Set up test structure**
   - Create `tests/` directory with subdirectories
   - Configure pytest with markers and fixtures
   - Set up test configuration files

2. **Implement unit tests**
   - Test all abstract base classes
   - Test component registry functionality
   - Test individual migrated components

3. **Implement integration tests**
   - Test complete training pipeline
   - Test inference pipeline
   - Test data loading and processing

4. **Add manual and debug tests**
   - Visualization tests
   - Data validation tests
   - Reproducibility tests

#### Deliverables:
- ✅ Complete test suite
- ✅ CI/CD pipeline configuration
- ✅ Test coverage reporting

### Phase 4: New Architectures (Week 7-8)
**Goal**: Implement additional OCR architectures

#### Tasks:
1. **Implement EAST architecture** (SKIP)
   - Create EAST encoder, decoder, head, loss
   - Register components in architecture registry
   - Create EAST configuration files

2. **Implement CRAFT architecture**
   - Create CRAFT components
   - Register in architecture registry
   - Create configuration files

3. **Add additional encoders**
   - EfficientNet backbone
   - Custom CNN implementations
   - Update registry with new options

#### Deliverables:
- ✅ EAST architecture implementation
- ✅ CRAFT architecture implementation
- ✅ Additional encoder options

### Phase 5: Advanced Features (Week 9-10)
**Goal**: Add advanced experimentation features

#### Tasks:
1. **Implement logging system**
   - Rich-based logging utilities
   - Icecream debugging integration
   - Structured logging for experiments

2. **Create synthetic data generation**
   - Module for generating augmented datasets
   - Integration with training pipeline
   - Configuration for synthetic data parameters

3. **Add experiment management**
   - Experiment tracking utilities
   - Result comparison tools
   - Hyperparameter optimization support

#### Deliverables:
- ✅ Advanced logging system
- ✅ Synthetic data generation
- ✅ Experiment management tools

### Phase 6: Documentation and Validation (Week 11-12)
**Goal**: Finalize documentation and validate implementation

#### Tasks:
1. **Update all documentation**
   - API reference for new architecture
   - Usage examples for different architectures
   - Migration guide for existing code

2. **Performance validation**
   - Benchmark different architectures
   - Validate accuracy preservation
   - Performance profiling and optimization

3. **Final integration testing**
   - End-to-end testing of all architectures
   - Cross-architecture compatibility testing
   - Production readiness validation

#### Deliverables:
- ✅ Complete documentation
- ✅ Performance benchmarks
- ✅ Production-ready codebase

## Risk Assessment

### High Risk Items:
1. **Breaking Changes**: Migration may break existing workflows
   - *Mitigation*: Maintain backward compatibility where possible, provide migration scripts

2. **Performance Regression**: New architecture may impact performance
   - *Mitigation*: Comprehensive benchmarking, gradual rollout

3. **Configuration Complexity**: Hydra config reorganization may introduce errors
   - *Mitigation*: Extensive validation, gradual migration

### Medium Risk Items:
1. **Testing Coverage**: Ensuring comprehensive test coverage
   - *Mitigation*: Start with critical path testing, expand gradually

2. **Component Compatibility**: Ensuring all components work together
   - *Mitigation*: Integration testing from early phases

### Low Risk Items:
1. **New Architecture Implementation**: Adding EAST/CRAFT
   - *Mitigation*: Implement incrementally, test thoroughly

## Dependencies and Prerequisites

### Required Dependencies:
- Python 3.9+
- PyTorch 1.12+
- PyTorch Lightning 1.8+
- Hydra 1.2+
- Timm (for backbones)
- Albumentations (for augmentations)
- CLEval (for metrics)

### New Dependencies:
- Rich (for logging)
- Icecream (for debugging)
- Additional testing libraries

### Development Tools:
- Black (code formatting)
- Isort (import sorting)
- Flake8 (linting)
- MyPy (type checking)
- Pre-commit hooks

## Success Criteria

### Functional Requirements:
- [ ] All existing functionality preserved
- [ ] DBNet architecture produces identical results
- [ ] New architectures (EAST, CRAFT) functional
- [ ] Plug-and-play component swapping works
- [ ] Configuration-driven architecture selection

### Quality Requirements:
- [ ] 90%+ test coverage
- [ ] All tests passing
- [ ] No performance regression >5%
- [ ] Clean, documented code
- [ ] Type hints throughout codebase

### Usability Requirements:
- [ ] Clear documentation and examples
- [ ] Easy configuration for experiments
- [ ] Reproducible results
- [ ] Efficient debugging and development

## Timeline and Milestones

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Foundation | 2 weeks | Core abstractions complete |
| Migration | 2 weeks | All components migrated |
| Testing | 2 weeks | 90% test coverage achieved |
| New Architectures | 2 weeks | EAST and CRAFT implemented |
| Advanced Features | 2 weeks | Logging and synthetic data ready |
| Documentation | 2 weeks | Complete documentation and validation |

## Resource Requirements

### Team:
- 1-2 Senior Python/ML Engineers
- 1 QA Engineer (for testing phases)
- 1 DevOps Engineer (for CI/CD)

### Infrastructure:
- GPU instances for training/testing
- CI/CD pipeline with GPU support
- Artifact storage for models and results

### Tools and Licenses:
- GitHub Actions (CI/CD)
- Weights & Biases (experiment tracking)
- Code quality tools (already configured)

## Monitoring and Metrics

### Progress Metrics:
- Lines of code migrated
- Test coverage percentage
- Number of architectures supported
- Performance benchmarks

### Quality Metrics:
- Test pass rate
- Code quality scores
- Documentation completeness
- Performance regression tracking

## Contingency Plans

### If Timeline Slips:
1. **Phase 1-2 delay**: Focus on core abstractions first, delay advanced features
2. **Testing delays**: Implement critical path tests first, expand later
3. **Architecture delays**: Complete DBNet migration first, add others iteratively

### If Technical Issues Arise:
1. **Performance issues**: Profile and optimize bottlenecks
2. **Compatibility issues**: Create adapter layers for problematic components
3. **Testing failures**: Debug and fix issues, update test expectations

## Next Steps

1. **Immediate Actions**:
   - Review and approve this plan
   - Set up development environment with new structure
   - Begin Phase 1 implementation

2. **Week 1 Planning**:
   - Create detailed task breakdown for Phase 1
   - Set up development branches
   - Configure CI/CD for new structure

3. **Communication**:
   - Share plan with team
   - Set up regular progress meetings
   - Establish communication channels for blockers

This refactor plan provides a structured approach to modernizing the OCR framework while maintaining functionality and enabling future experimentation.</content>
<parameter name="filePath">/home/vscode/workspace/upstage-receipt-text-detection-dbnet-baseline/refactor-plan.md
