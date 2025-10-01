# **filename: docs/ai_handbook/index.md**

# **AI Agent Handbook: OCR Project**

Version: 1.2 (2025-09-30)
Status: This handbook is the single source of truth for AI agents. The latest project status can be found in [our most recent changelog entry](./05_changelog/2025-09-29_legacy-ui-cleanup.md).

## **1. Project Overview**

This project develops a high-performance, modular OCR system for receipt text detection. Our architecture is built on PyTorch Lightning and Hydra, enabling flexible experimentation with various model components through a custom registry system.

## **2. ⚡ Quick Links & Context Bundles**

Use these curated "context bundles" to load the most relevant files for common tasks without overloading your context window.

👉 **AI cue markers:** Protocol files embed comments like `<!-- ai_cue:priority=high -->`. When an agent detects these markers, load high-priority docs first for the relevant `use_when` scenarios (debugging, automation, etc.).

#### **For a New Feature or Model Component:**

1. **Protocol:** [Coding Standards & Workflow](./02_protocols/01_coding_standards.md)
2. **Reference:** [Architecture](./03_references/01_architecture.md)
3. **Reference:** [Hydra & Component Registry](./03_references/02_hydra_and_registry.md)

#### **For Debugging a Training Run:**

1. **Protocol:** [Debugging Workflow](./02_protocols/03_debugging_workflow.md)
2. **Reference:** [Command Registry](./02_protocols/02_command_registry.md)
3. **Experiments:** [Experiment Logs](./04_experiments/) (Find a similar past run)

#### **For Adding a New Utility Function:**

1. **Protocol:** [Utility Adoption Guide](./02_protocols/04_utility_adoption.md)
2. **Reference:** [Existing Utility Functions](./03_references/03_utility_functions.md)

#### **For Launching Streamlit Apps:**

1. **Protocol:** [Command Registry](./02_protocols/02_command_registry.md)
2. **Runner:** [`run_ui.py`](../run_ui.py) commands `evaluation_viewer`, `inference`, `command_builder`, `resource_monitor`
3. **Doc Bundle:** `uv run python scripts/agent_tools/get_context.py --bundle streamlit-maintenance`

#### **For Managing Agent Context Logs:**

1. **Protocol:** [Context Logging & Summarization](./02_protocols/06_context_logging.md)
2. **Start Log:** `make context-log-start LABEL="<task>"`
3. **Summarize Log:** `make context-log-summarize LOG=logs/agent_runs/<file>.jsonl`

#### **For Planning the Next Training Run:**

1. **Protocol:** [Training & Experimentation](./02_protocols/13_training_protocol.md)
2. **Command:** `uv run python scripts/agent_tools/propose_next_run.py <wandb_run_id>`
3. **Extras:** `collect_results.py` and `generate_ablation_table.py` usage is documented inside the protocol for quick sweep analysis.

#### **For Updating Handbook Metadata & Bundles:**

1. **Manifest:** [`docs/ai_handbook/index.json`](./index.json)
2. **Validator:** `uv run python scripts/agent_tools/validate_manifest.py`
3. **Bundle Preview:** `uv run python scripts/agent_tools/get_context.py --list-bundles`

## **3. 🤖 Command Registry**

For safe, autonomous execution of tasks, refer to the [**Command Registry**](./02_protocols/02_command_registry.md). It contains a list of approved scripts, their functions, and examples.

## **4. 📚 Table of Contents**

### **01. Onboarding**

* [Project & Environment Setup](./01_onboarding/01_setup_and_tooling.md)
* [Data Overview](./01_onboarding/02_data_overview.md)

### **02. Protocols (How-To Guides)**

* [Coding Standards & Workflow](./02_protocols/01_coding_standards.md)
* [**Command Registry**](./02_protocols/02_command_registry.md)
* [Training & Experimentation](./02_protocols/13_training_protocol.md)
* [Debugging Workflow](./02_protocols/03_debugging_workflow.md)
* [Utility Adoption Guide](./02_protocols/04_utility_adoption.md)
* [Modular Refactoring Guide](./02_protocols/05_modular_refactor.md)
* [Context Logging and Summarization](./02_protocols/06_context_logging.md)
* [Iterative Debugging and Root Cause Analysis](./02_protocols/07_iterative_debugging.md)
* [Context Checkpointing & Restoration](./02_protocols/08_context_checkpointing.md)
* [Hydra Configuration Refactoring](./02_protocols/09_hydra_config_refactoring.md)
* [Refactoring Guide (redirect)](./02_protocols/10_refactoring_guide.md)
* [Template Adoption & Best Practices](./02_protocols/16_template_adoption_protocol.md)

### **03. References (Factual Information)**

* [Architecture](./03_references/01_architecture.md)
* [Hydra & Component Registry](./03_references/02_hydra_and_registry.md)
* [Utility Functions](./03_references/03_utility_functions.md)
* [Evaluation Metrics](./03_references/04_evaluation_metrics.md)

### **04. Experiments**

* [Experiment Log Template](./04_experiments/TEMPLATE.md)
* [View All Experiments](./04_experiments/)

### **05. Changelog**

* [View Project Changelog](./05_changelog/)
