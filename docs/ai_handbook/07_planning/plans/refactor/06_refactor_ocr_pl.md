
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
