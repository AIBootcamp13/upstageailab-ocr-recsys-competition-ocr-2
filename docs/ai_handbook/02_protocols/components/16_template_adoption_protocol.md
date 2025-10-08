<!-- ai_cue:priority=high -->

<!-- ai_cue:use_when=refactor,architecture,onboarding -->

# **Protocol: Template Adoption & Best Practices**

This protocol governs how to reference and adopt patterns from external template repositories, specifically the lightning-hydra-template. The goal is to systematically align our project with community best practices for structure, configuration, and scripting.

## **1. The Reference Template**

* **Primary Source:** lightning-hydra-template by @ashleve
* **GitHub URL:** https://github.com/ashleve/lightning-hydra-template
* **Local Documentation Path:** docs/external/lightning-hydra-template/ (template project) and docs/external/lightning-hydra-template/configs (configuration examples)

This local directory contains an offline copy of the template's official documentation, added by following the **External Documentation Protocol**.

## **2. Agent Workflow for Proposing Refactors**

When tasked with improving the project structure or aligning with best practices, an agent must follow this workflow.

### **Step 1: Analyze the Template's Structure**

1. **Load Key Documentation:** The agent must first read the core documentation files from the local copy of the template.
   * docs/external/lightning-hydra-template/README.md (Project Structure, Main Configs, Experiment Configs sections)
   * docs/external/lightning-hydra-template/configs/ (for configuration examples)
   * docs/external/lightning-hydra-template/src/ (for code structure examples)
2. **Identify Key Patterns:** The agent should identify the major structural patterns from the template, such as:
   * The src/ layout for all Python code.
   * The organization of configs into callbacks/, datamodule/, model/, etc.
   * The use of a centralized train.py and eval.py in the project root.
   * The structure of the PyTorch Lightning LitModule (the model).

### **Step 2: Compare with Our Current Project**

1. **Analyze Local Structure:** The agent must list the current directory structure of our project's ocr/ and configs/ directories.
2. **Identify Deltas (Differences):** The agent will perform a "diff" between the template's structure and our own, noting key differences.
   * *Example Delta:* "The template places all callbacks in configs/callbacks, whereas our project defines them inside configs/train.yaml. Adopting the template's pattern would improve modularity."

### **Step 3: Propose an Incremental Change**

The agent should **not** propose a full, "big bang" refactor. It must propose a single, incremental, and safe change based on its analysis.

1. **Formulate a Hypothesis:** State the proposed change and its expected benefit.
   * *Example Hypothesis:* "By refactoring our callback configurations to match the lightning-hydra-template pattern, we can make them more reusable and simplify the main train.yaml file."
2. **Create a Refactor Plan:** The proposal must include a concrete plan, referencing the **Modular Refactoring Protocol**.
   * **Files to Create:** configs/callbacks/default.yaml, configs/callbacks/early_stopping.yaml
   * **Files to Modify:** configs/train.yaml (to update the defaults list).
   * **Files to Delete:** None.
3. **Request Approval:** The agent presents this incremental plan to the user for approval before taking any action.

This systematic approach ensures that we can leverage the wisdom of popular templates to improve our project in a controlled and deliberate manner.
