# **Actionable Refactor Plan: Decouple the Lightning Module**

**Objective:** Decouple the metric calculation and evaluation logic from the main OCRPLExporter in ocr/lightning_modules/ocr_pl.py to improve modularity, testability, and performance.

### **Phase 1: Create a Dedicated Evaluation Service**

The goal of this phase is to extract the complex evaluation loop into a self-contained, testable service.

Action 1: Create the new service file.
Create a new file at ocr/evaluation/evaluator.py.
Action 2: Define the CLEvalEvaluator class.
In the new file, add the following class structure. It will be responsible for managing the state of the evaluation across an epoch.
```python
# ocr/evaluation/evaluator.py
from collections import OrderedDict, defaultdict
import numpy as np
from tqdm import tqdm
from ocr.metrics import CLEvalMetric

class CLEvalEvaluator:
    def __init__(self, dataset, metric_cfg):
        self.dataset = dataset
        self.metric = CLEvalMetric(**(metric_cfg or {}))
        self.predictions = OrderedDict()

    def update(self, filenames, predictions):
        """Update the evaluator state with predictions from a single batch."""
        for filename, boxes in zip(filenames, predictions):
            self.predictions[filename] = boxes

    def compute(self):
        """Compute the final metrics after an epoch."""
        cleval_metrics = defaultdict(list)

        # This is the logic moved from on_validation_epoch_end
        for gt_filename, gt_words in tqdm(self.dataset.anns.items(), desc="Evaluation"):
            if gt_filename not in self.predictions:
                # Handle missing predictions
                cleval_metrics["recall"].append(0.0)
                cleval_metrics["precision"].append(0.0)
                cleval_metrics["hmean"].append(0.0)
                continue

            pred = self.predictions[gt_filename]
            det_quads = [[point for coord in polygons for point in coord] for polygons in pred]
            gt_quads = [item.squeeze().reshape(-1) for item in gt_words]

            self.metric.reset()
            self.metric(det_quads, gt_quads)
            result = self.metric.compute()

            cleval_metrics["recall"].append(result["recall"].item())
            cleval_metrics["precision"].append(result["precision"].item())
            cleval_metrics["hmean"].append(result["f1"].item())

        recall = float(np.mean(cleval_metrics["recall"])) if cleval_metrics["recall"] else 0.0
        precision = float(np.mean(cleval_metrics["precision"])) if cleval_metrics["precision"] else 0.0
        hmean = float(np.mean(cleval_metrics["hmean"])) if cleval_metrics["hmean"] else 0.0

        return {"val/recall": recall, "val/precision": precision, "val/hmean": hmean}

    def reset(self):
        """Reset the internal state for a new epoch."""
        self.predictions.clear()
```

### **Phase 2: Integrate the New Service into the Lightning Module**

Now, refactor ocr_pl.py to use the new service, simplifying its responsibilities.

**Action 1: Modify ocr/lightning_modules/ocr_pl.py.**

* Import the new CLEvalEvaluator.
* In the OCRPLModule's __init__ method, instantiate the evaluator:
  ```python
  from ocr.evaluation.evaluator import CLEvalEvaluator
  # ... inside OCRPLModule.__init__
  self.valid_evaluator = CLEvalEvaluator(self.dataset["val"], self.metric_cfg)
  self.test_evaluator = CLEvalEvaluator(self.dataset["test"], self.metric_cfg)
  ```

* Remove the self.validation_step_outputs and self.test_step_outputs attributes.

**Action 2: Simplify the validation and test steps.**

* Modify validation_step to call the evaluator's update method.
  ```python
   # ... inside validation_step
  boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)
  self.valid_evaluator.update(batch["image_filename"], boxes_batch)
  return pred
  ```

* Modify on_validation_epoch_end to use the evaluator. **Replace the entire method** with this simplified version:
  ```python
    def on_validation_epoch_end(self):
      metrics = self.valid_evaluator.compute()
      self.log_dict(metrics, on_epoch=True, prog_bar=True)
      self.valid_evaluator.reset()
  ```

* Repeat the same simplification for test_step and on_test_epoch_end using self.test_evaluator.

### **Phase 3: Validation**

Ensure the refactoring did not introduce any regressions.

Action 1: Run a smoke test.
Execute a single training and validation step to verify the new wiring.
```bash
uv run python runners/train.py trainer.fast_dev_run=true
```
Confirm it runs without errors.

Action 2: Run a full validation epoch.
Run a short training job to confirm the metrics are calculated and logged correctly.
```bash
uv run python runners/train.py trainer.max_epochs=1 data.batch_size=4
```
Compare the logged val/hmean with a previous run to ensure correctness.

### **Prompt for Agentic AI (Next Session)**


Objective: Execute the refactoring plan to decouple the Lightning Module. Follow the three phases outlined in `refactor_plan_decouple_lightning_module.md`. Create the new evaluation service, integrate it into the `OCRPLModule`, and run the validation commands to ensure correctness.
