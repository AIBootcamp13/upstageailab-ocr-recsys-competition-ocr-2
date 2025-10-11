
# **Executable Refactor Plan: Decouple the Lightning Module**

**Status**: Ready for Execution
**Target**: `ocr/lightning_modules/ocr_pl.py` (845 lines)
**Goal**: Extract evaluation logic into a dedicated service, then modularize utilities
**Date**: October 11, 2025
**Branch**: 08_refactor/ocr_pl

## **Overview**

This plan merges the decoupling approach from `03_decouple_lightning_module_plan.md` with the modular extraction strategy from the original plan. We'll start with the high-impact evaluation decoupling, then extract supporting utilities.

### **Key Changes from Original Plan**
- **Simplified Phases**: Focus on evaluation decoupling first (highest impact)
- **Updated Paths**: Use `ocr/evaluation/` instead of `ocr/lightning_modules/evaluators/`
- **Executable Code**: Provide complete code snippets and commands
- **Current State Aware**: Verified file structure and existing code

---

## **Phase 1: Create Dedicated Evaluation Service** (High Impact - 2-3 hours)

### **Step 1.1: Create Evaluation Directory and Base Evaluator**
```bash
mkdir -p /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/evaluation
```

Create `/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/evaluation/__init__.py`:
```python
# ocr/evaluation/__init__.py
from .evaluator import CLEvalEvaluator

__all__ = ["CLEvalEvaluator"]
```

Create `/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/evaluation/evaluator.py`:
```python
# ocr/evaluation/evaluator.py
from collections import OrderedDict, defaultdict
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from tqdm import tqdm

from ocr.metrics import CLEvalMetric
from ocr.utils.orientation import remap_polygons

try:
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class CLEvalEvaluator:
    """Dedicated evaluator for CLEvalMetric-based OCR evaluation."""

    def __init__(self, dataset, metric_cfg: Optional[Dict[str, Any]] = None, mode: str = "val"):
        """
        Initialize the evaluator.

        Args:
            dataset: The dataset to evaluate against
            metric_cfg: Configuration for the metric
            mode: 'val' or 'test' for logging prefixes
        """
        self.dataset = dataset
        self.mode = mode
        self.metric_cfg = metric_cfg or {}
        self.metric = CLEvalMetric(**self.metric_cfg)
        self.predictions: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def update(self, filenames: List[str], predictions: List[Dict[str, Any]]) -> None:
        """Update the evaluator state with predictions from a single batch."""
        for filename, pred_data in zip(filenames, predictions):
            self.predictions[filename] = pred_data

    def _get_rich_console(self):
        """Get Rich console for progress bars."""
        try:
            from rich.console import Console
            return Console()
        except ImportError:
            return None

    def _compute_metrics_for_file(self, gt_filename: str, entry: Dict[str, Any]) -> Tuple[float, float, float]:
        """Compute metrics for a single file."""
        gt_words = self.dataset.anns[gt_filename] if hasattr(self.dataset, "anns") else self.dataset.dataset.anns[gt_filename]

        pred_polygons = entry.get("boxes", [])
        orientation = entry.get("orientation", 1)
        raw_size = entry.get("raw_size")

        # Handle raw size determination
        if raw_size is None:
            image_path = entry.get("image_path")
            if image_path is None:
                image_path = getattr(self.dataset, 'image_path', None)
                if image_path:
                    image_path = image_path / gt_filename
            try:
                with Image.open(image_path) as pil_image:
                    raw_width, raw_height = pil_image.size
            except Exception:
                raw_width, raw_height = 0, 0
        else:
            raw_width, raw_height = map(int, raw_size) if isinstance(raw_size, (list, tuple)) else (0, 0)

        # Prepare detection quads
        det_quads = [polygon.reshape(-1).tolist() for polygon in pred_polygons if polygon.size > 0]

        # Prepare ground truth quads
        canonical_gt = []
        if gt_words is not None and len(gt_words) > 0:
            if raw_width > 0 and raw_height > 0:
                canonical_gt = remap_polygons(gt_words, raw_width, raw_height, orientation)
            else:
                canonical_gt = [np.asarray(poly, dtype=np.float32) for poly in gt_words]
        gt_quads = [
            np.asarray(poly, dtype=np.float32).reshape(-1).tolist() for poly in canonical_gt if np.asarray(poly).size > 0
        ]

        # Compute metrics
        self.metric.reset()
        self.metric(det_quads, gt_quads)
        result = self.metric.compute()

        return (
            result["recall"].item(),
            result["precision"].item(),
            result["f1"].item()
        )

    def compute(self) -> Dict[str, float]:
        """Compute the final metrics after an epoch."""
        # Log cache statistics if available
        if hasattr(self.dataset, "log_cache_statistics"):
            self.dataset.log_cache_statistics()

        cleval_metrics = defaultdict(list)

        # Get filenames to evaluate (handle Subset datasets)
        if hasattr(self.dataset, "indices") and hasattr(self.dataset, "dataset"):
            filenames_to_check = [list(self.dataset.dataset.anns.keys())[idx] for idx in self.dataset.indices]
        else:
            filenames_to_check = list(self.dataset.anns.keys())

        # Only evaluate files that have predictions
        processed_filenames = [gt_filename for gt_filename in filenames_to_check if gt_filename in self.predictions]

        if not processed_filenames:
            logging.warning(f"No {self.mode} predictions found. This may indicate a data loading or prediction issue.")
            recall = precision = hmean = 0.0
        else:
            # Progress bar setup
            iterator = processed_filenames
            if RICH_AVAILABLE:
                console = self._get_rich_console()
                if console:
                    from rich.progress import Progress
                    with Progress(
                        TextColumn("[bold red]{task.description}"),
                        BarColumn(bar_width=50, style="red"),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TextColumn("•"),
                        TextColumn("[progress.completed]{task.completed}/{task.total}"),
                        TextColumn("•"),
                        TimeElapsedColumn(),
                        console=console,
                        refresh_per_second=2,
                    ) as progress:
                        task = progress.add_task("Evaluation", total=len(processed_filenames))
                        for gt_filename in processed_filenames:
                            entry = self.predictions[gt_filename]
                            recall, precision, hmean = self._compute_metrics_for_file(gt_filename, entry)
                            cleval_metrics["recall"].append(recall)
                            cleval_metrics["precision"].append(precision)
                            cleval_metrics["hmean"].append(hmean)
                            progress.advance(task)
                else:
                    iterator = tqdm(processed_filenames, desc="Evaluation", colour="red")
            else:
                iterator = tqdm(
                    processed_filenames,
                    desc="Evaluation",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                    colour="red",
                )

            if not RICH_AVAILABLE or not console:
                for gt_filename in iterator:
                    entry = self.predictions[gt_filename]
                    recall, precision, hmean = self._compute_metrics_for_file(gt_filename, entry)
                    cleval_metrics["recall"].append(recall)
                    cleval_metrics["precision"].append(precision)
                    cleval_metrics["hmean"].append(hmean)

            # Calculate averages
            recall = float(np.mean(cleval_metrics["recall"])) if cleval_metrics["recall"] else 0.0
            precision = float(np.mean(cleval_metrics["precision"])) if cleval_metrics["precision"] else 0.0
            hmean = float(np.mean(cleval_metrics["hmean"])) if cleval_metrics["hmean"] else 0.0

        return {
            f"{self.mode}/recall": recall,
            f"{self.mode}/precision": precision,
            f"{self.mode}/hmean": hmean,
        }

    def reset(self) -> None:
        """Reset the internal state for a new epoch."""
        self.predictions.clear()
```

### **Step 1.2: Integrate Evaluator into Lightning Module**

Update `/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/lightning_modules/ocr_pl.py`:

First, add the import at the top (after line 25):
```python
from ocr.evaluation import CLEvalEvaluator
```

In the `__init__` method (around line 26), replace the step_outputs with evaluators:
```python
# Replace these lines:
self.validation_step_outputs: OrderedDict[str, Any] = OrderedDict()
self.test_step_outputs: OrderedDict[str, Any] = OrderedDict()
self.predict_step_outputs: OrderedDict[str, Any] = OrderedDict()

# With:
self.valid_evaluator = CLEvalEvaluator(self.dataset["val"], self.metric_kwargs, mode="val")
self.test_evaluator = CLEvalEvaluator(self.dataset["test"], self.metric_kwargs, mode="test")
self.predict_step_outputs: OrderedDict[str, Any] = OrderedDict()
```

Update `validation_step` (around line 300-350, find the method):
```python
def validation_step(self, batch):
    pred = self.model(return_loss=False, **batch)

    boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)
    predictions = []
    for idx, boxes in enumerate(boxes_batch):
        normalized_boxes = [np.asarray(box, dtype=np.float32).reshape(-1, 2) for box in boxes]
        pred_data = {
            "boxes": normalized_boxes,
            "orientation": batch.get("orientation", [1])[idx] if "orientation" in batch else 1,
            "raw_size": tuple(batch.get("raw_size", [(0, 0)])[idx]) if "raw_size" in batch else None,
            "canonical_size": tuple(batch.get("canonical_size", [None])[idx]) if "canonical_size" in batch else None,
            "image_path": batch.get("image_path", [None])[idx] if "image_path" in batch else None,
        }
        predictions.append(pred_data)

    self.valid_evaluator.update(batch["image_filename"], predictions)
    return pred
```

Replace `on_validation_epoch_end` (lines 400-550):
```python
def on_validation_epoch_end(self):
    metrics = self.valid_evaluator.compute()
    for key, value in metrics.items():
        self.log(key, value, on_epoch=True, prog_bar=True)

    # Store final metrics for checkpoint saving
    self._checkpoint_metrics = {
        "recall": metrics.get("val/recall", 0.0),
        "precision": metrics.get("val/precision", 0.0),
        "hmean": metrics.get("val/hmean", 0.0),
    }

    self.valid_evaluator.reset()
```

Update `test_step` and `on_test_epoch_end` similarly (lines 550-700):
```python
def test_step(self, batch):
    pred = self.model(return_loss=False, **batch)

    boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)
    predictions = []
    for idx, boxes in enumerate(boxes_batch):
        normalized_boxes = [np.asarray(box, dtype=np.float32).reshape(-1, 2) for box in boxes]
        pred_data = {
            "boxes": normalized_boxes,
            "orientation": batch.get("orientation", [1])[idx] if "orientation" in batch else 1,
            "raw_size": tuple(batch.get("raw_size", [(0, 0)])[idx]) if "raw_size" in batch else None,
            "canonical_size": tuple(batch.get("canonical_size", [None])[idx]) if "canonical_size" in batch else None,
            "image_path": batch.get("image_path", [None])[idx] if "image_path" in batch else None,
        }
        predictions.append(pred_data)

    self.test_evaluator.update(batch["image_filename"], predictions)
    return pred

def on_test_epoch_end(self):
    metrics = self.test_evaluator.compute()
    for key, value in metrics.items():
        self.log(key, value, on_epoch=True, prog_bar=True)
    self.test_evaluator.reset()
```

### **Step 1.3: Test Phase 1**
```bash
# Run existing unit tests
cd /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2
python -m pytest tests/unit/test_lightning_module.py -v

# Run a smoke test
python -c "
from ocr.lightning_modules.ocr_pl import OCRPLModule
from ocr.evaluation import CLEvalEvaluator
print('Import successful')
"

# Quick training validation
python runners/train.py trainer.fast_dev_run=true
```

---

## **Phase 2: Extract Configuration and Utility Functions** (Low Risk - 2-3 hours)

### **Step 2.1: Create Utils Directory**
```bash
mkdir -p /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/lightning_modules/utils
```

Create `/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/lightning_modules/utils/__init__.py`:
```python
# ocr/lightning_modules/utils/__init__.py
from .config_utils import extract_metric_kwargs, extract_normalize_stats
from .checkpoint_utils import CheckpointHandler

__all__ = ["extract_metric_kwargs", "extract_normalize_stats", "CheckpointHandler"]
```

### **Step 2.2: Extract Config Utils**
Create `/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/lightning_modules/utils/config_utils.py`:
```python
# ocr/lightning_modules/utils/config_utils.py
from typing import Tuple
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf


def extract_metric_kwargs(metric_cfg: DictConfig | None) -> dict:
    """Extract metric kwargs from config."""
    if metric_cfg is None:
        return {}

    cfg_dict = OmegaConf.to_container(metric_cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        return {}

    cfg_dict.pop("_target_", None)
    return cfg_dict


def extract_normalize_stats(config) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Extract normalization stats from transforms config."""
    transforms_cfg = getattr(config, "transforms", None)
    if transforms_cfg is None:
        return None, None

    sections: list[ListConfig] = []
    for attr in ("train_transform", "val_transform", "test_transform", "predict_transform"):
        section = getattr(transforms_cfg, attr, None)
        if section is None:
            continue
        transforms = getattr(section, "transforms", None)
        if isinstance(transforms, ListConfig):
            sections.append(transforms)

    for transforms in sections:
        for transform in transforms:
            transform_dict = OmegaConf.to_container(transform, resolve=True)
            if not isinstance(transform_dict, dict):
                continue
            target = transform_dict.get("_target_")
            if target != "albumentations.Normalize":
                continue
            mean = transform_dict.get("mean")
            std = transform_dict.get("std")
            if mean is None or std is None:
                continue
            return np.array(mean), np.array(std)

    return None, None
```

### **Step 2.3: Extract Checkpoint Utils**
Create `/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/lightning_modules/utils/checkpoint_utils.py`:
```python
# ocr/lightning_modules/utils/checkpoint_utils.py
from typing import Any, Dict


class CheckpointHandler:
    """Handle checkpoint saving and loading of additional metrics."""

    @staticmethod
    def on_save_checkpoint(module, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Save additional metrics in the checkpoint."""
        if hasattr(module, "_checkpoint_metrics"):
            checkpoint["cleval_metrics"] = module._checkpoint_metrics
        return checkpoint

    @staticmethod
    def on_load_checkpoint(module, checkpoint: Dict[str, Any]) -> None:
        """Restore metrics from checkpoint (optional)."""
        if "cleval_metrics" in checkpoint:
            module._checkpoint_metrics = checkpoint["cleval_metrics"]
```

### **Step 2.4: Update Lightning Module**
In `ocr_pl.py`, add imports:
```python
from .utils.config_utils import extract_metric_kwargs, extract_normalize_stats
from .utils.checkpoint_utils import CheckpointHandler
```

Replace method calls:
```python
# In __init__:
self.metric_kwargs = extract_metric_kwargs(metric_cfg)
# ...
self._normalize_mean, self._normalize_std = extract_normalize_stats(self.config)

# Replace on_save_checkpoint and on_load_checkpoint:
def on_save_checkpoint(self, checkpoint):
    return CheckpointHandler.on_save_checkpoint(self, checkpoint)

def on_load_checkpoint(self, checkpoint):
    CheckpointHandler.on_load_checkpoint(self, checkpoint)
```

### **Step 2.5: Test Phase 2**
```bash
# Test utils
python -c "
from ocr.lightning_modules.utils.config_utils import extract_metric_kwargs, extract_normalize_stats
from ocr.lightning_modules.utils.checkpoint_utils import CheckpointHandler
print('Utils import successful')
"

# Run tests
python -m pytest tests/unit/test_lightning_module.py -v
python runners/train.py trainer.fast_dev_run=true
```

---

## **Phase 3: Extract Image Processing and Logging** (Low Risk - 2-3 hours)

### **Step 3.1: Create Processors Directory**
```bash
mkdir -p /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/lightning_modules/processors
```

Create processors for image handling and logging utilities.

### **Step 3.2: Extract Image Processor**
Create `/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/lightning_modules/processors/image_processor.py`:
```python
# ocr/lightning_modules/processors/image_processor.py
from typing import Tuple
import torch
from PIL import Image


class ImageProcessor:
    """Handle image processing utilities."""

    @staticmethod
    def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        # Denormalize if needed (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Unnormalize
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)

        # Convert to PIL
        to_pil = transforms.ToPILImage()
        return to_pil(tensor)

    @staticmethod
    def prepare_wandb_image(pil_image: Image.Image, max_side: int | None = 640) -> Image.Image:
        """Prepare image for W&B logging."""
        if max_side is None:
            return pil_image

        width, height = pil_image.size
        if width > height:
            new_width = max_side
            new_height = int(height * max_side / width)
        else:
            new_height = max_side
            new_width = int(width * max_side / height)

        return pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
```

### **Step 3.3: Extract Progress Logger**
Create `/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/lightning_modules/loggers/progress_logger.py`:
```python
# ocr/lightning_modules/loggers/progress_logger.py
try:
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def get_rich_console():
    """Get Rich console for progress bars."""
    if RICH_AVAILABLE:
        return Console()
    return None
```

### **Step 3.4: Update Lightning Module**
Add imports and use the extracted utilities where applicable.

### **Step 3.5: Test Phase 3**
```bash
# Test processors
python -c "
from ocr.lightning_modules.processors.image_processor import ImageProcessor
from ocr.lightning_modules.loggers.progress_logger import get_rich_console
print('Processors import successful')
"

# Full test
python -m pytest tests/unit/ -k lightning_module -v
python runners/train.py trainer.max_epochs=1 trainer.limit_val_batches=5
```

---

## **Phase 4: Final Cleanup and Documentation** (Low Risk - 1 hour)

### **Step 4.1: Update __init__.py Files**
Update `/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ocr/lightning_modules/__init__.py`:
```python
# ocr/lightning_modules/__init__.py
from .ocr_pl import OCRPLModule
from .utils import extract_metric_kwargs, extract_normalize_stats, CheckpointHandler
from .processors.image_processor import ImageProcessor
from .loggers.progress_logger import get_rich_console

__all__ = [
    "OCRPLModule",
    "extract_metric_kwargs",
    "extract_normalize_stats",
    "CheckpointHandler",
    "ImageProcessor",
    "get_rich_console"
]
```

### **Step 4.2: Add Documentation**
Add module docstrings and update README if needed.

### **Step 4.3: Final Testing**
```bash
# Run full test suite
python -m pytest tests/ -x --tb=short

# Performance test
python runners/train.py trainer.max_epochs=1 data.batch_size=4
```

---

## **Success Criteria**

- [ ] All existing tests pass
- [ ] Training produces identical results (±0.001 tolerance)
- [ ] Evaluation metrics unchanged
- [ ] No performance regression (<5% slowdown)
- [ ] Code maintainability improved (file sizes <400 lines)
- [ ] Clean separation of concerns
- [ ] Backward compatibility maintained

## **Rollback Plan**

If issues arise:
```bash
# Revert all changes
git checkout HEAD~1 -- ocr/lightning_modules/ocr_pl.py
git checkout HEAD~1 -- ocr/evaluation/
git checkout HEAD~1 -- ocr/lightning_modules/utils/
# etc.
```

## **Timeline**

- **Phase 1**: 2-3 hours (evaluation decoupling)
- **Phase 2**: 2-3 hours (config/utils extraction)
- **Phase 3**: 2-3 hours (processors/logging)
- **Phase 4**: 1 hour (cleanup)
- **Total**: 7-10 hours

This plan is executable, reflects current project structure, and prioritizes the highest-impact changes first.
