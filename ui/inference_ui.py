"""Streamlit entrypoint for the OCR inference experience.

This thin wrapper keeps backwards compatibility with the legacy module path
while delegating the real work to the modular Streamlit app under
``ui.apps.inference``. The heavy lifting now lives in that package; this file
exists so existing launch commands (``streamlit run ui/inference_ui.py``)
continue to operate without modification.

⚠️ Maintain the separation of concerns documented in
``docs/ai_handbook/02_protocols/11_streamlit_maintenance_protocol.md`` and
``docs/ai_handbook/02_protocols/12_streamlit_refactoring_protocol.md``. Consult
those guides—along with the configs in ``configs/ui/`` and schemas in
``configs/schemas/``—before introducing logic here.
"""

from __future__ import annotations

import warnings

# Suppress known Pydantic compatibility warnings
# This warning occurs when dependencies use the old Pydantic v2 parameter name
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:", category=UserWarning)
warnings.filterwarnings("ignore", message="'allow_population_by_field_name' has been renamed to 'validate_by_name'", category=UserWarning)

# Suppress known wandb Pydantic compatibility warnings
# This is a known issue where wandb uses incorrect Field() syntax in Annotated types
# The warnings come from Pydantic when processing wandb's type annotations
warnings.filterwarnings("ignore", message=r"The '(repr|frozen)' attribute.*Field.*function.*no effect", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*(repr|frozen).*Field.*function.*no effect", category=UserWarning)

# Also suppress by category for more reliable filtering
try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except ImportError:
    pass  # In case the warning class is not available in future pydantic versions

from ui.apps.inference.app import main

if __name__ == "__main__":
    main()
