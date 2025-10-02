# Registry Compatibility Notes

Keep these guardrails in mind when registering new OCR components:

- **Architecture entries** must map to encoder, decoder, head, and loss identifiers that already exist in the global registry. Ensure the encoder exposes `out_channels` so downstream decoders can infer their `in_channels` automatically.
- **Compatible backbones** should be listed in `configs/ui_meta/architectures/<name>.yaml`. Limit the list to encoders that have been validated with the architecture (shape agreement, normalization, and feature staging).
- **Recommended optimizers** belong in the same UI metadata file. Only include optimizers whose hyper-parameters have been tuned for the architecture; this drives the UI defaults and learning-rate slider bounds.
- **Decoder overrides** are currently locked to the architecture entry. To offer a selectable decoder, add a new architecture preset or extend the registry to accept decoder-specific overrides (see `ocr/models/core/registry.py`).
- Always run `pytest tests/test_architecture.py` after updating registry registrations to confirm component wiring remains valid.
