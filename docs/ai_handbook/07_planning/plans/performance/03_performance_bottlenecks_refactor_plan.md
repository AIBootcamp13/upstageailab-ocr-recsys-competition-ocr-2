# **Actionable Refactor Plan: Address Performance Bottlenecks**

**Objective:** Systematically identify and resolve the two primary performance bottlenecks: inefficient metric calculation during validation and suboptimal DataLoader configuration.

### **Task 1: Optimize Metric Calculation**

**Analysis:** The current implementation in ocr_pl.py re-iterates the entire validation dataset in on_validation_epoch_end. This is highly inefficient.

**Solution:** Integrate the CLEvalMetric with torchmetrics to perform batch-wise updates and efficient epoch-end aggregation.

**This task is the primary goal of the "Decouple the Lightning Module" refactoring plan.** Therefore, executing refactor_plan_decouple_lightning_module.md will resolve this bottleneck. No separate action is needed other than ensuring the new CLEvalEvaluator is used correctly.

### **Task 2: Tune DataLoader num_workers**

**Analysis:** The num_workers is hardcoded to a high value (12) which may not be optimal for all hardware and can cause CPU bottlenecks.

**Solution:** Use profiling to find the optimal num_workers value for the target environment and update the default configuration.

Phase 1: Instrument the Training Loop with the PyTorch Profiler.
Modify runners/train.py to include profiling, allowing us to measure the time spent in data loading vs. computation.

* **Action:** Add the profiler context manager to the training_step of OCRPLModule in ocr/lightning_modules/ocr_pl.py.
  ```
  # Add this to imports
  from torch.profiler import profile, record_function, ProfilerActivity

  # Wrap the training_step content
  def training_step(self, batch, batch_idx):
      with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
          with record_function("model_train_step"):
              # ... existing training_step logic ...

      # At the end of an epoch or after a few steps, export the trace
      if self.global_step % 100 == 0:
          prof.export_chrome_trace(f"trace_{self.global_step}.json")

      return pred

  ```

  *Note: A more robust implementation would use a PyTorch Lightning Profiler callback.*

Phase 2: Run an Ablation Study on num_workers.
Execute short training runs with different num_workers values to gather performance data.

* **Action:** Use Hydra's multirun feature to sweep over num_workers.
  uv run python runners/train.py --multirun
      trainer.max_steps=200
      data.batch_size=8
      dataloaders.train_dataloader.num_workers=0,2,4,8,12

Phase 3: Analyze Profiler Traces.
Use Chrome's tracing tool (chrome://tracing) or another profiler viewer to analyze the generated .json traces.

* **Action:**
  1. Load the trace_*.json files for each run.
  2. Look for the DataLoader and model_train_step sections.
  3. Identify the num_workers value where the time spent waiting for data (gaps between model_train_step calls) is minimized, and the CPU is not overloaded. This is the optimal value.

Phase 4: Update the Default Configuration.
Once the optimal num_workers is found, update the default configuration file.

* **Action:** Modify configs/dataloaders/default.yaml with the new, empirically determined value for num_workers.

### **Prompt for Agentic AI (Next Session)**

Objective: Execute the performance tuning plan for the DataLoader. Follow the four phases outlined in `refactor_plan_performance_bottlenecks.md` for Task 2. Instrument the training loop with the PyTorch Profiler, run an ablation study sweeping over `num_workers`, analyze the results to find the optimal value, and update the default configuration.
