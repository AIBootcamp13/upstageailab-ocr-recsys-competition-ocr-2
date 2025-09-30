# **filename: docs/ai_handbook/02_protocols/05_modular_refactor.md**
<!-- ai_cue:priority=medium -->
<!-- ai_cue:use_when=refactor,architecture -->

# **Protocol: Modular Refactoring**

This protocol provides the authoritative workflow for modular refactors. It now absorbs all guidance that previously lived in `10_refactoring_guide.md`.

## **1. The Goal: Clean Interfaces, Independent Components**

The primary objective of a modular refactor is to increase the long-term health and velocity of the project by:

* **Increasing Cohesion:** Grouping related logic into self-contained modules.
* **Reducing Coupling:** Minimizing dependencies between different parts of the system.
* **Improving Testability:** Making components small enough to be tested in isolation.
* **Enhancing Reusability:** Creating modules that can be reused in different contexts.
* **Improving Readability:** Making the system's architecture easier to understand.

## **2. When to Use This Protocol**

Trigger this protocol whenever you notice "code smells" that indicate low modularity, such as:

* A file or class that has grown too large or mixes unrelated responsibilities.
* Tightly coupled components that must be edited together.
* Duplicated logic across modules.
* Components that are difficult to unit test because dependencies are tangled.

## **3. Core Refactoring Principles**

Align with these guardrails before you touch any code—they keep the project configuration-driven and compatible with our plug-and-play architecture.

1. **Configuration First:** Expose new behaviour through Hydra configs under `configs/` rather than hard-coding paths or parameters.
2. **Respect Base Classes:** Encoders, decoders, heads, and losses must inherit from the abstract base classes in `ocr/models/core/base_classes.py`.
3. **Registry Compliance:** Register new components via `ocr/models/core/registry.py` so the model factory can discover them.
4. **Maintain Testability:** Every extracted unit should be covered by existing tests or new targeted tests. If you cannot test it, reconsider the design.

## **4. The Modular Refactor Workflow**

All modular refactors follow the same four-phase loop. Each phase maps to the checklists that used to live in the separate Refactoring Guide—those steps are now consolidated here.

### **Phase 1: Analyze & Plan**

1. **Clarify the Why:** Capture the motivation (tech debt, new feature, performance regression).
2. **Review Architecture Docs:** Refresh your mental model via [Architecture](../03_references/01_architecture.md) and any relevant component references.
3. **Define Scope & Target Design:** Sketch the desired module boundaries, file layout, and registry/config updates.
4. **Baseline Behaviour:** Run the smallest meaningful verification (unit tests, `uv run python scripts/decoder_benchmark.py --help`, etc.) so you can confirm parity later.

### **Phase 2: Execute Incrementally**

1. **Branch Off:** Work on a dedicated Git branch to keep history clean.
2. **Extract, Don't Rewrite:** Move working code into new homes. Delay clean-up or rewrites until after behaviour is preserved.
3. **Commit & Test Frequently:** After each extraction, run focused tests (or lint) and commit once the step is stable.

### **Phase 3: Validate Continuously**

1. **Run Targeted Tests:** Execute the suites covering the touched modules.
2. **Repeat the Baseline:** Re-run the baseline from Phase 1 before and after major milestones to catch regressions early.

### **Phase 4: Finalise & Document**

1. **Update References & Configs:** Point Hydra configs, registries, and imports at the new modules.
2. **Remove Dead Code:** Once the new path is stable, delete the superseded files/functions.
3. **Document the Change:** Update the AI Handbook references and add a `docs/ai_handbook/05_changelog/` entry summarising the refactor.

## **5. Refactoring Checklist**

Use this checklist as a readiness gate before you declare the refactor complete.

- [ ] **Analysis:** Scope, motivation, and target architecture documented.
- [ ] **Planning:** New module layout + registry/config updates drafted.
- [ ] **Implementation:**
  - [ ] Components inherit from the correct base class.
  - [ ] Registries (`ocr/models/core/registry.py`) updated.
  - [ ] Hydra configs under `configs/` created or adjusted.
- [ ] **Validation:** Unit tests / smoke tests (e.g., `trainer.fast_dev_run=true`) pass.
- [ ] **Documentation:** Handbook + changelog updated, obsolete code removed.
