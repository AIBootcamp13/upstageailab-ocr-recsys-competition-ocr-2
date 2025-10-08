# **filename: docs/ai_handbook/02_protocols/13_training_protocol.md**

<!-- ai_cue:priority=high -->

<!-- ai_cue:use_when=training,experimentation -->

# **Protocol: Training & Experimentation**

This protocol provides the single source of truth for planning, executing, analyzing, and iterating on model training experiments. Following this workflow ensures that all runs are reproducible, learnings are captured, and context is efficiently passed between human and AI collaborators.

## **The Principle: Plan, Run, Analyze, Iterate**

Every experiment is treated as a cycle. We form a hypothesis, test it with a training run, analyze the results to draw conclusions, and use those conclusions to form the hypothesis for the next cycle. This creates a clear, traceable chain of reasoning.

## **The Workflow**

### **Step 1: Plan the Experiment in a Log File**

Before running any code, you must define what you are trying to achieve.

1. **Copy the Template**: Duplicate docs/ai_handbook/04_experiments/TEMPLATE.md to a new file. Use the naming convention YYYY-MM-DD_objective.md (e.g., 2025-10-01_learning-rate-analysis.md).
2. **State the Objective**: In the "Objective" section, write a clear, testable hypothesis.
   * *Good Example:* "Hypothesis: The current box_thresh of 0.4 is too high, causing low recall. This experiment will test a lower threshold of 0.3 to increase recall without significantly harming precision."
3. **Define the Configuration**: In the "Configuration" section, paste the full, runnable command-line invocation, including all key overrides. This ensures perfect reproducibility.

### **Step 2: Start a Structured Context Log**

For full auditability, all actions taken during the experiment should be logged.

1. **Start the Log**: make context-log-start LABEL="<experiment-name>"
2. **Log the Plan**: The first action logged should be a reference to the experiment plan file created in Step 1.

### **Step 3: Execute and Monitor**

1. **Run Training**: Execute the exact command from your experiment plan.
2. **Monitor**: Use Weights & Biases (W&B) to monitor the run in real-time.

### **Step 4: Analyze and Summarize**

Once the run is complete, transform the raw results into knowledge.

1. **Link the W&B Run**: Add the direct URL to the W&B run dashboard in the "Results" section of your experiment log.
2. **Record Key Metrics**: Populate the results table with the final, most important metrics from the W&B summary (e.g., val/hmean, test/recall, test/precision).
3. **Write the Analysis**: This is the most critical part. In the "Analysis & Learnings" section, interpret the results in the context of your hypothesis.
   * Was the hypothesis validated or invalidated?
   * What was surprising? (e.g., "Recall increased as expected, but precision dropped more than anticipated.")
   * What do the validation images show?
4. **Summarize Actions**: Run make context-log-summarize LOG=<path_to_your_log.jsonl> to generate a clean summary of the session's actions.
5. **Capture the Sweep Output (Optional)**: When you run ablation sweeps, automate the result roll-up so future runs are comparable:
    * **Collect W&B runs:**

       ```bash
       uv run python ablation_study/collect_results.py --project "receipt-text-recognition-ocr-project" --tag lr_scan --output outputs/ablation/lr_scan_results.csv
       ```

       *Key flags*: `--entity` if your runs live under a shared team account, `--metrics` to restrict the summary output, and `--group-by` to aggregate by any config column (for example `model.optimizer.lr`).

    * **Generate publishable tables:**

       ```bash
       uv run python ablation_study/generate_ablation_table.py --input outputs/ablation/lr_scan_results.csv --ablation-type learning_rate --metric val/hmean --output-md docs/ablation/lr_scan.md
       ```

       Add `--output-latex docs/ablation/lr_scan.tex` if you need LaTeX, or `--columns` for custom column selections. The script auto-sorts by your primary metric and formats the sweeping parameter for quick comparison.

### **Step 5: Define the Next Iteration**

Close the loop by planning the next logical step.

1. **Form a New Hypothesis**: Based on your analysis, what should be tested next?
2. **Define Next Steps**: In the "Next Steps" section of your experiment log, create a clear, actionable checklist for the subsequent experiment. This becomes the input for the next cycle, starting again at Step 1.
3. **Auto-Propose the Follow-up Command (Optional)**: If you have a canonical W&B run ID, generate a draft Hydra command with the command:

   ```bash
   uv run python scripts/agent_tools/propose_next_run.py <wandb_run_id>
   ```

   The helper inspects the original configuration and returns a prefilled suggestion you can refine before committing to the next experiment.
