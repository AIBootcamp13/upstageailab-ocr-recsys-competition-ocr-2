# **filename: docs/ai_handbook/02_protocols/11_streamlit_maintenance_protocol.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=streamlit,maintenance -->

# **Protocol: Streamlit Maintenance**

This protocol codifies the recurring maintenance loop for Streamlit-based UI modules in this repository. Follow it whenever you touch a Streamlit app without a large architectural refactor. The canonical implementation today is the OCR inference experience, but the guardrails below apply to any app living under `ui/apps/`.

---

## **1. Scope & Ownership**

All Streamlit surfaces live in `ui/`. Each runnable app sits under `ui/apps/<app_name>/` with a thin façade (for the inference experience this is `ui/inference_ui.py`). The module consumes:

- UI configuration: `configs/ui/<app_name>.yaml` (e.g., `configs/ui/inference.yaml`).
- Runtime defaults & schemas: `configs/schemas/` (notably `default_model.yaml` and `ui_inference_compat.yaml`).
- UI metadata: `ui_meta/` (naming, copy, experiment storyboards, and sample payloads).
- App-local packages such as `components/`, `services/`, `models/`, and `state.py` within `ui/apps/<app_name>/`.

Maintenance keeps these assets functional, discoverable, and in sync with Agentic AI automation; update paths in this section if a new Streamlit app joins the codebase.

---

## **2. Trigger Conditions**

Run this protocol whenever any of the following occur:

1. Streamlit dependency upgrades or breaking changes (e.g., deprecations like `use_container_width`).
2. Model or checkpoint catalogue changes (new experiments in `outputs/` or schema updates).
3. Configuration changes under `configs/ui/` or `configs/schemas/`.
4. UI regressions reported by QA or notebook agents.
5. Periodic hygiene (recommend monthly or before competition milestones).

---

## **3. Environment Preparation**

1. Activate project environment (Agentic AI uses `uv`):
   - `uv run streamlit --version`
2. Ensure the CUDA device you intend to use is visible (or force CPU via `CUDA_VISIBLE_DEVICES=""`).
3. Confirm required backing assets exist:
   - `outputs/` contains at least one checkpoint for catalogue validation (for inference-style apps).
   - `ui_meta/` assets (copy, examples) are present for the app you are touching.
4. Open two terminals:
   - **App:** run the UI via `uv run streamlit run ui/inference_ui.py --server.port=8504`
   - **Watchdog:** tail the terminal for logs and errors.

---

## **4. Maintenance Checklist**

### **A. Sanity Sweep**

- [ ] `uv run ruff check ui/apps/inference ui/utils/inference_engine.py`
- [ ] `uv run python -m compileall ui` (quick syntax guard)
- [ ] Run the Streamlit app; ensure the landing page renders without warnings.

### **B. Data & Catalogue Health**

- [ ] Evaluate backing data/catalogue as applicable to the app:
   - OCR inference: `uv run python - <<'PY' ... build_catalog(...)`.
   - Other apps: run their service-layer health checks.
- [ ] Verify compatibility schemas (`configs/schemas/ui_inference_compat.yaml` or app-specific equivalents) match new assets; add families if mismatched.
- [ ] Update `configs/schemas/default_model.yaml` if the canonical default model changes.
- [ ] Confirm demo or fallback modes only trigger when catalogue truly lacks assets.

### **C. Real Inference Path**

- [ ] Select at least one ResNet-based and one MobileNet-based checkpoint.
- [ ] Upload a sample image; ensure real inference (not mock) completes with polygons/confidences.
- [ ] If inference falls back to mock data, inspect logs under `ui/utils/inference_engine.py` and retry after fixes.

### **D. Configuration & Schema Integrity**

- [ ] Validate YAML against schema: `uv run python scripts/agent_tools/validate_yaml.py configs/ui/<app_name>.yaml`.
- [ ] Ensure `configs/schemas/*.yaml` stay the single source of truth. Pydantic models in `ui/apps/<app_name>/models/` exist to validate Streamlit session and form state only—never replace Hydra/OmegaConf configs with Pydantic models.
- [ ] Keep docstrings, `ui_meta/` copy, and slider defaults stacked in `configs/ui/<app_name>.yaml` in sync.

### **E. Documentation & Automation**

- [ ] Append a changelog entry (`docs/ai_handbook/05_changelog/`)
- [ ] Notify Agentic AI assistants (update `project-overview.md` summary if behaviour shifts).
- [ ] File or close relevant tracking issues.

---

## **5. Maintenance Playbook**

### **Step 1 – Snapshot Inputs**

1. Record current git status; stash unrelated work.
2. Export most recent `outputs/` metadata: `uv run python scripts/agent_tools/dump_catalog.py` (when available).

### **Step 2 – Apply Changes**

2. Follow the checklist tasks relevant to your change.
3. When editing schemas, run schema validation scripts and regenerate any derived artifacts in `ui_meta/`.
4. For dependency updates, regenerate lockfile (`uv lock`) and run `uv run pre-commit run --all-files` if hooks exist.
5. Keep edits modular—services, components, and configs should change independently when possible.

### **Step 3 – Validate**

1. Re-run the Streamlit app and exercise:
   - Model selector
   - Hyperparameter sliders
   - Upload flows (single + multi-file)
2. Inspect logs for warnings (e.g., compatibility mismatches, missing configs, schema validation failures, torch load fallbacks).
3. Run a targeted UI smoke test (if available) or manual QA with sample artifacts sourced from `ui_meta/`.

### **Step 4 – Sync Documentation Hooks**

1. Verify `AI_DOCS` markers in touched modules still cite the correct sections (search for `# AI_DOCS[`).
2. Run `uv run python scripts/agent_tools/validate_manifest.py` after editing handbook bundles or commands to catch broken links early.
3. If sharing code publicly, strip markers before publishing (`uv run python scripts/agent_tools/strip_doc_markers.py --apply`) and restore afterward.

### **Step 4 – Close the Loop**

1. Capture results in commit message (`streamlit: <summary>`)
2. Update documentation or playbooks touched by the change.
3. Push branch and open PR referencing this protocol.

---

## **6. Maintenance Anti-Patterns**

- Hard-coding paths to checkpoints; rely on `CatalogOptions.from_paths` and configs.
- Bypassing schema validation to “force” a checkpoint—add a family entry instead.
- Editing Streamlit scripts without running lint + manual smoke tests.
- Leaving mock inference fallback as the default path.
- Forgetting to sync UI copy with config/tooltips.

---

## **7. Quick Reference**

| Task | Command/Location |
| --- | --- |
| Launch Streamlit app | `uv run streamlit run ui/<app_entry>.py --server.port=8504` |
| Check OCR catalogue | `uv run python - <<'PY' ... build_catalog(...)` |
| Lint UI package | `uv run ruff check ui/apps/<app_name> ui/utils` |
| UI configuration | `configs/ui/<app_name>.yaml` |
| Core schemas | `configs/schemas/ui_inference_compat.yaml`, `configs/schemas/default_model.yaml` |
| Metadata & copy | `ui_meta/<app_name>/` |
| Entry point | `ui/<app_entry>.py` → `ui/apps/<app_name>/app.py` |

Use this protocol to keep the Streamlit module reliable, discoverable, and ready for Agentic AI automation.
