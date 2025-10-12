# **filename: docs/ai_handbook/refactor_plan.md**


# **AI Documentation Refactor & Enhancement Plan**

This document outlines a phased plan to refactor the project's documentation into a centralized, efficient, and scalable knowledge base for AI agents, incorporating best practices from both internal and external assessments. Our guiding principle is a systematic, numbered hierarchy for clarity and navigation.

## **Phase 1: Foundation & Structure (1 Week)**

**Goal:** Establish the new, authoritative directory structure and the central index file.

1. **Create New Directory:**
   * Initialize the new documentation root: docs/ai_handbook/.
2. **Build Subdirectories:**
   * Create the following folder structure inside docs/ai_handbook/:
     * 01_onboarding/ (For project overview, architecture, and setup)
     * 02_protocols/ (For task-oriented, step-by-step guides)
     * 03_references/ (For stable, factual information)
     * 04_experiments/ (For logging experiment runs and results)
     * 05_changelog/ (For tracking significant project state changes)
3. **Draft the Central Index (index.md):**
   * Create docs/ai_handbook/index.md. This file is the primary entry point.
   * Populate it with a project overview, a global Table of Contents, and sections for "Context Bundles" and the "Command Registry."

## **Phase 2: Content Migration & Consolidation (1-2 Weeks)**

**Goal:** Migrate existing documentation into the new structure, eliminating redundancy.

1. **Content Migration Map:**

| Old File(s) | New File | Purpose |
| :---- | :---- | :---- |
| docs/copilot/context.md, architecture-overview.md, component-diagrams.md | 03_references/01_architecture.md | The single source of truth for the model and system architecture. |
| docs/copilot/instructions.md, .github/copilot-instructions.md | index.md (Quick Start section) | The new instruction is simple: "Start at index.md." |
| docs/development/coding-standards.md, naming-conventions.md | 02_protocols/01_coding_standards.md | Step-by-step guide for developers on writing and contributing code. |
| docs/copilot/data-context.md | 01_onboarding/02_data_overview.md | High-level context about the dataset, formats, and challenges. |
| docs/development/pydantic-policy.md, registry-compatibility.md | 03_references/02_hydra_and_registry.md | Detailed reference on how Hydra and the component registry work. |
| docs/copilot/quick-reference.md | 03_references/03_utility_functions.md | API-style reference for key reusable functions. |
| docs/maintenance/project-state.md | 05_changelog/YYYY-MM-DD_state.md | Project state updates become entries in a versioned changelog. |
| docs/plans/* | _archive/docs/plans/ | Archive historical planning documents. |

2. **Archive Old Documentation:**
   * Move the old docs/ subdirectories into an _archive/ folder to prevent confusion.

## **Phase 3: Lightweight Tooling & Automation (Ongoing)**

**Goal:** Create lightweight, AI-callable tools for safe, autonomous execution.

1. **Create a Command Registry:**
   * Create 02_protocols/02_command_registry.md.
   * Document a curated list of safe-to-run scripts, including purpose, arguments, expected output, and a note on dry-run modes.
2. **Develop Script Wrappers:**
   * Create a new scripts/agent_tools/ directory.
   * Add simple Python wrappers for common tasks (e.g., validate_config.py, list_checkpoints.py).
   * Ensure all relevant scripts support --dry-run, limit_batches, or similar flags for safe inspection.
3. **Update index.md:**
   * Link to the new command_registry.md from the main index for easy discoverability.

## **Phase 4: Process & Cadence (Ongoing)**

**Goal:** Implement routines to keep the documentation accurate and in sync with the codebase.

1. **Define Context Bundles:**
   * In index.md, create task-specific "Context Bundles" that list the essential documents for recurring workflows.
2. **Establish Versioning Routine:**
   * Formalize the process: When the project state is updated, a new entry is created in 05_changelog/. The index.md is updated to point to the latest entry.
3. **Implement Review Cadence:**
   * Schedule a recurring monthly task to audit the documentation. The checklist includes validating links in index.md, updating the command registry, and pruning obsolete information.
4. **Dedicated Experiment & Run Docs:**
   * Create 04_experiments/TEMPLATE.md to standardize how new experiment results are logged.
   * This ensures that configuration, results, learnings, and links to outputs (W&B, etc.) are captured consistently.



----


### The Prompt

You are an expert senior software engineer tasked with creating a detailed implementation guide for a junior developer (or another AI). I have a high-level **strategic plan** to refactor a complex Python `Dataset` class.

Your task is to expand this strategic plan into a highly detailed **Procedural Blueprint**. This blueprint must be so clear and unambiguous that it eliminates any need for creative interpretation during the coding phase. It should serve as a precise, step-by-step assembly manual for the refactor.

-----

### **Context: The Strategic Plan**

```markdown
# Refactor Plan: OCR Dataset Base (`ocr/datasets/base.py`)

## Objective and Why
Refactor `ocr/datasets/base.py` to establish robust data validation, eliminate the "God Object" anti-pattern, and improve maintainability using Pydantic v2 and separating concerns into dedicated modules like `CacheManager` and utility files.

## Data Contracts
The refactor will use the following Pydantic models defined in `ocr/datasets/schemas.py`:
- `DatasetConfig`: For declarative configuration.
- `ImageMetadata`, `PolygonData`: For structured data representation.
- `DataItem`: The final, validated output object from the `__getitem__` method before being converted to a dictionary.
- `CacheManager`: Configuration for caching.

## Delegated Development Work
The plan involves creating new modules for schemas, utilities, and a cache manager, and then refactoring the main `OCRDataset` class to use them.

## Tree Structure of Proposed Refactor
ocr/
├── datasets/
│   ├── schemas.py          # NEW: Pydantic data models
│   └── base.py             # REFACTORED: ValidatedOCRDataset
└── utils/
    ├── cache_manager.py    # NEW: CacheManager class
    ├── image_utils.py      # NEW: Image processing utilities
    └── polygon_utils.py    # NEW: Polygon processing utilities
```

-----

### **Context: The Target File to Refactor**

```python
# [
#  PASTE THE ENTIRE CONTENT OF THE
#  ocr/datasets/base.py FILE HERE
# ]
```

-----

### **Your Task: Generate the Procedural Blueprint**

Based on the provided context, generate a complete procedural blueprint in a single markdown document. The blueprint must include the following sections:

**1. API Surface Definitions**

Define the precise class and method signatures for the new and refactored components. Include type hints for all arguments and return values.

  * `ocr.utils.cache_manager.CacheManager`: Define its `__init__` and all public methods for getting/setting cached items (images, tensors, maps). Include methods for logging statistics.
  * `ocr.datasets.base.ValidatedOCRDataset`: Define its `__init__`, `__len__`, and `__getitem__` methods. The `__init__` method must accept a single `DatasetConfig` object.

**2. Detailed Implementation Pseudocode**

Provide step-by-step procedural pseudocode inside commented code blocks for the core logic of the `ValidatedOCRDataset` class.

  * **`__init__(self, config: DatasetConfig, transform: Callable)` method:**

      * Detail how to parse the `annotation_path` to build the `self.anns` dictionary.
      * Show how to instantiate the `CacheManager` using the configuration from `config`.
      * Describe the logic for dispatching to preloading methods (`_preload_images`, `_preload_maps`) based on the config.

  * **`__getitem__(self, idx: int)` method:**

      * This is the most critical part. Your pseudocode must meticulously detail the entire data loading and processing pipeline in the correct order:
        1.  **Tensor Cache Check:** Start by checking the `CacheManager` for a fully processed `DataItem` for the given `idx`. If found, return it immediately.
        2.  **Image Loading:** If no tensor is cached, get the filename. Check the `CacheManager` for a cached image. If not found, load from disk using `image_utils`, handle EXIF orientation, convert to a NumPy array, and then cache it using the `CacheManager`.
        3.  **Annotation Processing:** Load raw polygons. If they exist, use `polygon_utils` to remap them for orientation and filter out degenerate polygons.
        4.  **Transformation:** Assemble the image, polygons, and metadata into a `TransformInput` Pydantic model. Pass this single object to the `self.transform` callable.
        5.  **Final Assembly & Validation:** Construct the final `DataItem` Pydantic model from the transformed outputs. This step serves as the final validation of the data contract.
        6.  **Tensor Caching:** Store the validated `DataItem` object in the `CacheManager`.
        7.  **Return Value:** Convert the `DataItem` to a dictionary (`.model_dump()`) and return it.

**3. Data Flow Specification**

Briefly describe the lifecycle of the key data objects. Explain where each Pydantic model (`DatasetConfig`, `TransformInput`, `DataItem`) is created and how it is passed between the dataset, the cache manager, and the transform functions to enforce the data contract at every stage.
