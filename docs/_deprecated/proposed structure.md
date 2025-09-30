```
docs/
└── ai_handbook/
├── index.md # The "Hub": Main entry point and index for AI agents.
├── 01_onboarding/ # High-level introductory materials.
├── 02_protocols/ # Step-by-step guides for recurring tasks.
├── 03_references/ # Stable, factual information about the project.
├── 04_experiments/ # Logs and analysis of specific training runs.
└── 05_changelog/ # Project status updates and versioning.
```

## **2. Detailed Breakdown**

### **index.md - The Hub**

* **Purpose:** The single entry point for any AI agent.
* **Content:**
  * Brief project overview.
  * "Quick Links & Context Bundles" for common tasks.
  * Link to the Command Registry.
  * A full Table of Contents linking to all other documents.

### **01_onboarding/**

* **Purpose:** Get a new contributor (human or AI) up to speed quickly.
* **Files:**
  * 01_setup_and_tooling.md: Environment setup, uv usage, VS Code config.
  * 02_data_overview.md: Dataset structure, annotation format, common challenges.

### **02_protocols/**

* **Purpose:** Action-oriented playbooks for how to perform specific tasks.
* **Files:**
  * 01_coding_standards.md: Linting, formatting, naming conventions.
  * 02_command_registry.md: The official list of safe-to-run scripts for agents.
  * 03_debugging_workflow.md: Standard procedure for diagnosing issues.
  * 04_utility_adoption.md: How to use and contribute to shared utils.
  * 05_modular_refactor.md: Guide for refactoring components into the registry.

### **03_references/**

* **Purpose:** The "encyclopedia" of the project. Contains facts, not processes.
* **Files:**
  * 01_architecture.md: Describes the plug-and-play code architecture.
  * 02_hydra_and_registry.md: Deep dive into Hydra schemas and the component registry.
  * 03_utility_functions.md: API reference for key utils.
  * 04_evaluation_metrics.md: Explanation of CLEval and other metrics.

### **04_experiments/**

* **Purpose:** A structured log of all significant model training runs and benchmarks.
* **Files:**
  * TEMPLATE.md: A markdown template for logging a new experiment.
  * YYYY-MM-DD_experiment-name.md: Individual log files for each run.

### **05_changelog/**

* **Purpose:** To track the evolution of the project and documentation, ensuring agents use up-to-date information.
* **Files:**
  * YYYY-MM-DD_version-update.md: Snapshot of project state changes.
