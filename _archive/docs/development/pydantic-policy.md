# Pydantic Usage Policy (UI Validation Only)

- Hydra remains the single source of truth for configuration composition and runtime parameters.
- Pydantic MUST NOT be used to manage or compose model configuration files.
- Pydantic MAY be used only for:
  - Validating user inputs in Streamlit UIs before producing Hydra overrides.
  - Optionally validating the fully composed Hydra config object (converted to a dict) at runtime.

Rationale: Running two configuration managers in parallel (Hydra + Pydantic models with defaults) leads to duplication and drift. Keep one direction of flow:

UI Inputs -> (Pydantic Validation) -> Hydra Overrides -> Hydra Composition -> (Optional Pydantic validation) -> Runners

Before modifying UI, config, or runners, consult these docs:
- `docs/copilot/context.md`
- `docs/development/` guidelines
- `docs/maintenance/project-state.md`
