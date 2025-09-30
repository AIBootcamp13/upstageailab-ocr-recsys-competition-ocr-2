# **filename: docs/ai_handbook/02_protocols/08_hydra_config_refactoring.md**

# **Protocol: Hydra Configuration Refactoring**

This protocol provides a safe and systematic process for refactoring the project's Hydra configuration structure. Due to the declarative and compositional nature of Hydra, this process is fundamentally different from refactoring Python code and requires a deliberate, plan-first approach.

## **The Challenge: Why Hydra Refactoring is Hard**

* **Lack of Intermediate States:** A partially refactored Hydra config is often completely broken. Unlike Python, you cannot easily test one small change at a time. The entire structure must be valid for the application to even start.
* **"Silent" Errors:** Incorrect defaults or path issues can lead to the wrong configuration being composed without any explicit error, leading to confusing behavior downstream.
* **Context Contamination:** Iterative trial-and-error attempts quickly pollute the conversation history with failed approaches, making it difficult to track the intended state.

## **The Strategy: Plan, Execute, Validate**

This protocol follows a strict three-phase process to mitigate these risks. You must complete each phase before proceeding to the next.

### **Phase 1: Offline Planning & Analysis**

**Do not modify any live configuration files in this phase.** All work must be done in a single, comprehensive planning document.

1. **Define the Goal:** State the objective clearly.
   * *Good Example:* "Decompose the monolithic db.yaml into separate data and dataloader config groups to improve modularity, following the lightning-hydra-template pattern."
2. **Audit the Current State:**
   * **List Affected Config Files:** List every `.yaml` file that will be created, moved, modified, or deleted.
   * **Identify Code References:** Search the codebase (especially runners/ and scripts/) to find how the main experiment configs (e.g., train.yaml) are invoked. This identifies the primary entry points.
   * **Map Interdependencies:** Make a best effort to map how the defaults lists in your main configs connect to the various config groups.
3. **Design the Target State:** This is the core of your plan.
   * **Visualize the Migration:** Create a Mermaid diagram that shows the configs/ directory structure *before* and *after* the refactor.

     ```mermaid
     graph TD
         subgraph "Before"
             A["configs/"] --> A1["preset/datasets/db.yaml"];
             A --> A2["train.yaml (contains dataloader logic)"];
         end
         subgraph "After"
             B["configs/"] --> B1["data/default.yaml"];
             B --> B2["dataloaders/default.yaml"];
             B --> B3["train.yaml (references new groups)"];
         end
     ```

   * **Write New File Contents & Define Headers:** For every new or modified file, write its complete contents. You **must** follow the convention for package headers to ensure Hydra's resolver works correctly.
     * **For files in the root of a config group (e.g., configs/data/default.yaml):** Use the _global_ package header. This makes its contents available at the top level (e.g., as data).

       ```yaml
       # @package _global_

       # Main data configuration
       # ...
       ```
     * **For files inside a config group (e.g., configs/model/decoder/unet.yaml):** Use a group-specific package header. This makes the file selectable from the defaults list.

       ```yaml
        # @package _group_.model.decoder

       # UNet decoder configuration
       # ...
       ```
   * **Define Post-Refactor Usage:** Clearly state how the new configuration will be used.
   * **New defaults list:** Write the exact defaults list for the main experiment config (e.g., train.yaml).

     ```yaml
      defaults:
       - _self_
       - data: default
       - dataloaders: default
       - model: dbnet
       # ...
     ```
     * **New Invocation:** Provide an example of the new, cleaner command-line invocation if it has changed.
4. **Establish a Validation Command:** Identify a single, simple command that is known to work with the *current* configuration and which you will use to verify the *new* configuration.
   * *Example:* `uv run python runners/train.py --config-name train data.limit_val_batches=1`

### **Phase 2: Transactional Execution**

Now, you will execute the plan you created in Phase 1. This should be done as a single, focused set of actions within a dedicated git branch.

1. **Create a New Branch:** `git checkout -b feature/refactor-hydra-configs`
2. **Execute Your Plan:** Create, modify, and move all files exactly as defined in your plan from Phase 1.

### **Phase 3: Validation & Debugging**

Verify the new structure using your validation command and Hydra's built-in tools.

1. **Run the Validation Command:** Execute the command you identified in Phase 1.
   * **If it succeeds:** Your refactor was successful. Proceed to the Finalize step.
   * **If it fails:** Do not guess the fix. Proceed to the debugging steps below.
2. **Debugging with Hydra Tools:**
   * **To See the Final Config (--cfg job):** This is your most powerful tool. It prints the fully composed configuration your application would receive.
     ```bash
      uv run python runners/train.py --config-name train data.limit_val_batches=1 --cfg job
     ```
     Compare this output to your plan. The discrepancy will reveal the error.
   * **To See How Hydra is Working (hydra.verbose):** If the composition is unexpected, this will show you Hydra's search path and exactly which files it is composing.
     ```bash
      uv run python runners/train.py --config-name train hydra.verbose=true
     ```
3. **Finalize and Document:**
   * Once the validation command succeeds, commit your changes.
   * Update the Command Registry if any script invocations have changed.
   * Create a Changelog entry detailing the refactor.
