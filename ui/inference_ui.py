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

from ui.apps.inference.app import main

if __name__ == "__main__":
    main()
