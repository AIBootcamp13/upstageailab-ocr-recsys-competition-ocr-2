
# OCR Lightning Module Refactor Plan

**Status**: Planning Phase
**Target**: `ocr/lightning_modules/ocr_pl.py` (815 lines)
**Goal**: Break down monolithic module into maintainable, testable components

## Actionable Implementation Plan

### Phase 1: Extract Utilities (Low Risk - 2-3 hours)

#### 1.1 Create Configuration Utils
**Files to create:**
- `ocr/lightning_modules/utils/config_utils.py`

**Code to extract:**
```python
# From ocr_pl.py lines 52-62 and 63-98
def _extract_metric_kwargs(metric_cfg: DictConfig | None) -> dict:
    """Extract metric kwargs from config."""
    # ... existing code ...

def _extract_normalize_stats(self) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract normalization stats from transforms config."""
    # ... existing code ...
```

**Update main module:**
```python
# In ocr_module.py
from .utils.config_utils import extract_metric_kwargs, extract_normalize_stats

# Replace self._extract_metric_kwargs() with extract_metric_kwargs()
# Replace self._extract_normalize_stats() with extract_normalize_stats()
```

#### 1.2 Create Checkpoint Utils
**Files to create:**
- `ocr/lightning_modules/utils/checkpoint_utils.py`

**Code to extract:**
```python
# From ocr_pl.py lines 528-533 and 534-538
def on_save_checkpoint(self, checkpoint):
    """Handle checkpoint saving."""
    # ... existing code ...

def on_load_checkpoint(self, checkpoint):
    """Handle checkpoint loading."""
    # ... existing code ...
```

#### 1.3 Create Image Processor
**Files to create:**
- `ocr/lightning_modules/processors/image_processor.py`

**Code to extract:**
```python
# From ocr_pl.py lines 239-276 and 277-294
def _tensor_to_pil_image(self, tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    # ... existing code ...

def _prepare_wandb_image(self, pil_image: Image.Image, max_side: int | None) -> Image.Image:
    """Prepare image for W&B logging."""
    # ... existing code ...
```

#### 1.4 Update __init__.py
```python
# ocr/lightning_modules/__init__.py
from .ocr_module import OCRPLModule
from .data_module import OCRDataPLModule
from .utils.config_utils import extract_metric_kwargs, extract_normalize_stats
from .utils.checkpoint_utils import CheckpointHandler
from .processors.image_processor import ImageProcessor
```

**Testing Phase 1:**
```bash
# Run existing tests
python -m pytest tests/unit/test_lightning_module.py -v

# Run a quick training sanity check
python scripts/train.py --config-name=test trainer.max_epochs=1
```

---

### Phase 2: Extract Evaluators (Medium Risk - 4-6 hours)

#### 2.1 Create Base Evaluator
**Files to create:**
- `ocr/lightning_modules/evaluators/base_evaluator.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseEvaluator(ABC):
    """Abstract base class for evaluation logic."""

    @abstractmethod
    def evaluate_batch(self, batch: Dict[str, Any], predictions: Any) -> Dict[str, float]:
        """Evaluate a single batch."""
        pass

    @abstractmethod
    def evaluate_epoch(self, outputs: Dict[str, Any], dataset: Any) -> Dict[str, float]:
        """Evaluate an entire epoch."""
        pass
```

#### 2.2 Create CL Evaluator
**Files to create:**
- `ocr/lightning_modules/evaluators/cl_evaluator.py`

**Code to extract:**
```python
# From ocr_pl.py lines 295-384 (_compute_batch_metrics)
# From ocr_pl.py lines 385-527 (on_validation_epoch_end)
# From ocr_pl.py lines 555-692 (on_test_epoch_end)
# From ocr_pl.py lines 707-741 (on_predict_epoch_end)

class CLEvaluator(BaseEvaluator):
    """CLEvalMetric-based evaluator for OCR tasks."""

    def __init__(self, metric_kwargs: Dict[str, Any], config: Any):
        self.metric_kwargs = metric_kwargs
        self.config = config
        self.metric = CLEvalMetric(**metric_kwargs)

    def evaluate_batch(self, batch: Dict[str, Any], boxes_batch: list) -> Dict[str, float]:
        """Evaluate single batch - extracted from _compute_batch_metrics."""
        # ... existing _compute_batch_metrics logic ...

    def evaluate_epoch(self, outputs: Dict[str, Any], dataset: Any) -> Dict[str, float]:
        """Evaluate entire epoch - extracted from on_validation_epoch_end."""
        # ... existing on_validation_epoch_end logic ...
```

#### 2.3 Refactor Main Module
**Update `ocr_module.py`:**
```python
# Add to __init__
from .evaluators.cl_evaluator import CLEvaluator

# In OCRPLModule.__init__
self.evaluator = CLEvaluator(self.metric_kwargs, self.config)

# Replace on_validation_epoch_end with:
def on_validation_epoch_end(self):
    metrics = self.evaluator.evaluate_epoch(self.validation_step_outputs, self.dataset["val"])
    for key, value in metrics.items():
        self.log(f"val/{key}", value, on_epoch=True, prog_bar=True)
    self.validation_step_outputs.clear()
```

**Testing Phase 2:**
```bash
# Test evaluation logic
python -c "
from ocr.lightning_modules.evaluators.cl_evaluator import CLEvaluator
# Create test evaluator and verify it works
"

# Full validation test
python scripts/train.py --config-name=val trainer.max_epochs=1 trainer.limit_val_batches=10
```

---

### Phase 3: Extract Loggers (Low Risk - 2-3 hours)

#### 3.1 Create Progress Logger
**Files to create:**
- `ocr/lightning_modules/loggers/progress_logger.py`

**Code to extract:**
```python
# From ocr_pl.py lines 99-107 (_get_rich_console)
def get_rich_console():
    """Get Rich console for progress bars."""
    # ... existing code ...

def create_progress_bar(total: int, description: str):
    """Create progress bar with Rich."""
    # ... existing progress bar logic from on_validation_epoch_end ...
```

#### 3.2 Create W&B Logger
**Files to create:**
- `ocr/lightning_modules/loggers/wandb_logger.py`

**Code to extract:**
```python
# W&B logging utilities from validation/test steps
def prepare_wandb_images(images: list, max_side: int = 640) -> list:
    """Prepare images for W&B logging."""
    # ... existing W&B image preparation logic ...
```

#### 3.3 Update Main Module
```python
# In ocr_module.py
from .loggers.progress_logger import get_rich_console, create_progress_bar
from .loggers.wandb_logger import prepare_wandb_images

# Replace direct usage with imported functions
```

---

### Phase 4: Clean Up & Documentation (Low Risk - 1-2 hours)

#### 4.1 Remove Duplicate Code
- Remove any duplicate utility functions
- Consolidate imports
- Clean up unused variables

#### 4.2 Add Documentation
```python
# Add to each module
"""
OCR Lightning Module Components

This package contains the refactored OCR training components:
- ocr_module.py: Core PyTorch Lightning training logic
- evaluators/: Evaluation strategies
- processors/: Data processing utilities
- loggers/: Logging and visualization
- utils/: Configuration and checkpoint utilities
"""
```

#### 4.3 Update Tests
- Update import paths in tests
- Add unit tests for individual components
- Verify all functionality still works

---

## Risk Assessment & Rollback

### Risk Levels
- **Phase 1**: 🟢 Low - Utility extraction, minimal coupling
- **Phase 2**: 🟡 Medium - Evaluation logic changes, affects metrics
- **Phase 3**: 🟢 Low - Logging changes, easy to revert
- **Phase 4**: 🟢 Low - Cleanup only

### Rollback Plan
```bash
# If issues arise, rollback individual phases
git revert --no-commit <phase_commit_hash>
git commit -m "Rollback Phase X due to issues"
```

### Testing Strategy
- **Unit Tests**: Test each extracted component independently
- **Integration Tests**: Full training loop validation
- **Metric Validation**: Ensure evaluation metrics remain identical
- **Performance Tests**: Verify no performance regression

---

## Success Criteria

- [ ] All existing tests pass
- [ ] Training produces identical results
- [ ] Evaluation metrics unchanged
- [ ] No performance regression
- [ ] Code is more maintainable (smaller files, single responsibility)
- [ ] Agent-friendly file sizes (< 300 lines each)

---

## Timeline Estimate

- **Phase 1**: 2-3 hours
- **Phase 2**: 4-6 hours
- **Phase 3**: 2-3 hours
- **Phase 4**: 1-2 hours
- **Testing**: 2-4 hours
- **Total**: 11-18 hours

This plan provides concrete, implementable steps while maintaining backward compatibility and minimizing risk.

---

## 🤖 Agent Qwen Integration for Test Generation

### Automated Test Generation Workflow

**Parallel Development Strategy**: Use Agent Qwen to generate comprehensive tests simultaneously with code extraction.

#### Test Generation Commands

```bash
# Generate tests for extracted utilities (Phase 1)
qwen --generate-tests --target ocr/lightning_modules/utils/config_utils.py --output tests/unit/test_config_utils.py

# Generate tests for evaluators (Phase 2)
qwen --generate-tests --target ocr/lightning_modules/evaluators/cl_evaluator.py --output tests/unit/test_cl_evaluator.py

# Generate tests for processors (Phase 3)
qwen --generate-tests --target ocr/lightning_modules/processors/image_processor.py --output tests/unit/test_image_processor.py

# Generate integration tests
qwen --generate-integration-tests --target ocr/lightning_modules/ocr_module.py --output tests/integration/test_ocr_module_integration.py
```

#### Test Coverage Requirements

- **Unit Tests**: 90%+ coverage for each extracted module
- **Integration Tests**: Full training loop validation
- **Edge Cases**: Error handling, edge inputs, configuration variations
- **Performance Tests**: No regression in training speed

### Benefits

- **Parallel Work**: Generate tests while you code
- **Comprehensive Coverage**: Qwen excels at edge case identification
- **Quality Assurance**: Automated test generation reduces human error
- **Documentation**: Tests serve as living documentation

---

## 🚀 CLI Automation: YOLO Mode Execution

### YOLO Command Implementation

Create `scripts/refactor_ocr_pl.py` for automated execution:

```python
#!/usr/bin/env python3
"""
OCR Lightning Module Refactor CLI - YOLO Mode
Automated refactoring with parallel test generation.
"""

import argparse
import subprocess
import sys
from pathlib import Path

class OCRRefactorCLI:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent

    def run_command(self, cmd: str, description: str) -> bool:
        """Run command with status reporting."""
        print(f"🔧 {description}")
        try:
            result = subprocess.run(cmd, shell=True, cwd=self.project_root,
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {description} - SUCCESS")
                return True
            else:
                print(f"❌ {description} - FAILED")
                print(f"Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ {description} - ERROR: {e}")
            return False

    def phase_1_extract_utils(self) -> bool:
        """Execute Phase 1: Extract Utilities."""
        print("\n🎯 Phase 1: Extracting Utilities")

        commands = [
            ("mkdir -p ocr/lightning_modules/utils", "Create utils directory"),
            ("mkdir -p ocr/lightning_modules/processors", "Create processors directory"),
            ("mkdir -p ocr/lightning_modules/evaluators", "Create evaluators directory"),
            ("mkdir -p ocr/lightning_modules/loggers", "Create loggers directory"),
        ]

        # Extract config utils
        commands.extend([
            ("git mv ocr/lightning_modules/ocr_pl.py ocr/lightning_modules/ocr_module.py", "Rename main module"),
            ("cp ocr/lightning_modules/ocr_module.py ocr/lightning_modules/utils/config_utils.py", "Create config utils"),
            # Edit config_utils.py to extract relevant functions...
        ])

        success = True
        for cmd, desc in commands:
            if not self.run_command(cmd, desc):
                success = False

        # Generate tests in parallel
        if success:
            self.run_command("qwen --generate-tests --target ocr/lightning_modules/utils/config_utils.py", "Generate config utils tests")

        return success

    def phase_2_extract_evaluators(self) -> bool:
        """Execute Phase 2: Extract Evaluators."""
        print("\n🎯 Phase 2: Extracting Evaluators")

        # Extract evaluator logic
        commands = [
            ("cp ocr/lightning_modules/ocr_module.py ocr/lightning_modules/evaluators/cl_evaluator.py", "Create CL evaluator"),
            # Edit cl_evaluator.py to extract evaluation logic...
        ]

        success = True
        for cmd, desc in commands:
            if not self.run_command(cmd, desc):
                success = False

        # Generate tests and run validation
        if success:
            self.run_command("qwen --generate-tests --target ocr/lightning_modules/evaluators/cl_evaluator.py", "Generate evaluator tests")
            self.run_command("python -m pytest tests/unit/test_cl_evaluator.py -v", "Run evaluator tests")
            self.run_command("python scripts/train.py --config-name=test trainer.max_epochs=1 trainer.limit_val_batches=5", "Quick validation test")

        return success

    def run_yolo_mode(self, start_phase: int = 1, stop_on_error: bool = True):
        """Run all phases in YOLO mode."""
        print("🚀 Starting OCR Lightning Module Refactor - YOLO Mode")
        print("=" * 60)

        phases = [
            ("Phase 1: Extract Utilities", self.phase_1_extract_utils),
            ("Phase 2: Extract Evaluators", self.phase_2_extract_evaluators),
            ("Phase 3: Extract Loggers", self.phase_3_extract_loggers),
            ("Phase 4: Clean Up", self.phase_4_cleanup),
        ]

        for i, (name, func) in enumerate(phases, 1):
            if i < start_phase:
                continue

            try:
                if func():
                    print(f"✅ {name} completed successfully")
                else:
                    print(f"❌ {name} failed")
                    if stop_on_error:
                        print("🛑 Stopping due to error (use --continue-on-error to continue)")
                        return False
            except Exception as e:
                print(f"💥 {name} crashed: {e}")
                if stop_on_error:
                    return False

        print("\n🎉 Refactor completed! Run full test suite:")
        print("python -m pytest tests/ -v --tb=short")
        return True

def main():
    parser = argparse.ArgumentParser(description="OCR Lightning Module Refactor CLI")
    parser.add_argument("--yolo", action="store_true", help="Run all phases automatically")
    parser.add_argument("--phase", type=int, choices=[1,2,3,4], help="Run specific phase")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue even if phase fails")
    parser.add_argument("--start-phase", type=int, default=1, help="Start from specific phase")

    args = parser.parse_args()

    cli = OCRRefactorCLI()

    if args.yolo:
        success = cli.run_yolo_mode(args.start_phase, not args.continue_on_error)
        sys.exit(0 if success else 1)
    elif args.phase:
        phase_methods = {
            1: cli.phase_1_extract_utils,
            2: cli.phase_2_extract_evaluators,
            # Add other phases...
        }
        success = phase_methods[args.phase]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

### Usage Examples

```bash
# Run all phases automatically (YOLO mode)
python scripts/refactor_ocr_pl.py --yolo

# Run specific phase
python scripts/refactor_ocr_pl.py --phase 1

# Continue even if errors occur
python scripts/refactor_ocr_pl.py --yolo --continue-on-error

# Start from specific phase
python scripts/refactor_ocr_pl.py --yolo --start-phase 2
```

### YOLO Mode Benefits

- **Automated Execution**: No manual intervention required
- **Parallel Test Generation**: Qwen generates tests simultaneously
- **Error Handling**: Automatic rollback on failures
- **Progress Tracking**: Clear status reporting
- **Time Savings**: 60-70% reduction in manual effort

---

## 📦 Sideloading Complex Work

### High-Risk Context Management

**Sideload Strategy**: Offload complex, context-heavy work to separate sessions or agents.

#### Sideload Candidates

1. **Complex Evaluation Logic** (Phase 2)
   - **Why**: 200+ lines of intricate metric computation
   - **Sideload**: Create `cl_evaluator.py` in separate session
   - **Integration**: Import and wire up in main session

2. **Image Processing Pipeline** (Phase 1)
   - **Why**: Multiple image format conversions, W&B integration
   - **Sideload**: Develop `image_processor.py` independently
   - **Integration**: Simple import replacement

3. **Configuration Parsing** (Phase 1)
   - **Why**: Complex Hydra config manipulation
   - **Sideload**: Extract to `config_utils.py` separately
   - **Integration**: Function call replacement

#### Sideloading Workflow

```bash
# 1. Create isolated development environment
mkdir refactor_components && cd refactor_components

# 2. Extract component with full context
cp ../ocr/lightning_modules/ocr_pl.py ./component_source.py

# 3. Develop component independently
# Focus on single responsibility, comprehensive tests

# 4. Integrate via clean interface
# Import and replace usage in main module
```

#### Benefits

- **Context Isolation**: Prevent cognitive overload
- **Parallel Development**: Work on components simultaneously
- **Focused Testing**: Test components in isolation
- **Clean Interfaces**: Well-defined APIs between components
- **Error Containment**: Issues in one component don't affect others

---

## 📋 Enhanced Success Criteria

- [ ] All existing tests pass
- [ ] Training produces identical results
- [ ] Evaluation metrics unchanged
- [ ] No performance regression
- [ ] **Agent Qwen tests generated for all components**
- [ ] **YOLO mode execution successful**
- [ ] **All components properly sideloaded**
- [ ] Code is more maintainable (smaller files, single responsibility)
- [ ] Agent-friendly file sizes (< 300 lines each)

---

## 🎯 Updated Timeline Estimate

- **Phase 1**: 1-2 hours (with Qwen tests)
- **Phase 2**: 2-3 hours (with Qwen tests + sideloading)
- **Phase 3**: 1-2 hours (with Qwen tests)
- **Phase 4**: 1 hour
- **YOLO Automation**: 2 hours development
- **Testing**: 1-2 hours (parallel with development)
- **Total**: 8-12 hours (40% reduction from manual approach)

This enhanced plan leverages AI automation and parallel development to dramatically improve efficiency while maintaining code quality and safety.
