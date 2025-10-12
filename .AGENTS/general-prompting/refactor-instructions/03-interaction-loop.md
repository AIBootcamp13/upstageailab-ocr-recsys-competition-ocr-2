Here is the latest version of our "Living Refactor Blueprint". Please execute the `[COMMAND]` at the end.

---
# [ PASTE THE ENTIRE BLUEPRINT FROM THE AI's LAST RESPONSE HERE ]
---

[COMMAND]:
Migrate the preprocessing script located at `scripts/preprocess_data.py`. It currently instantiates the old `OCRDataset`. Update it to use `hydra.instantiate` to build the `ValidatedOCRDataset` from a `DatasetConfig` object.
