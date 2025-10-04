
## 4-week incremental roadmap ✅

### Week 1 – Synthetic foundations
- [ ] **Fix internal generator**
  - Patch `SyntheticDatasetGenerator` filename/JSON issues.
  - Add Hydra config `configs/preset/synthetic/base.yaml`.
- [ ] **CLI + preview**
  - Create `runners/generate_synthetic.py` producing samples to `outputs/synthetic/<run_id>`.
  - Export preview grid + metadata manifest.
- [ ] **PaddleOCR/MMOCR adapter**
  - Vendor toolkit files; expose `backend: {custom,paddle,mmocr}`.
  - Normalize annotations to CLEval schema (polygons + transcripts).
  - Document install/runtime instructions.
- [ ] **QA snapshot**
  - Generate 1k receipts from each backend, run overlapping-box/text sanity checks, capture summary in `outputs/synthetic/<run_id>/qa.json`.

### Week 2 – Receipt domain flavor
- [ ] **Dictionaries & templates**
  - Build JSON/YAML dictionaries: merchants, SKUs, taxes, languages.
  - Define layout templates for header/body/footer and multi-column tables.
- [ ] **Assets & rendering**
  - Collect fonts (thermal, dot-matrix, Hangul/CJK) with licenses.
  - Integrate logos, QR, barcodes (python-barcode, qrcode).
  - Extend renderer for dotted leaders, right-aligned totals, coupon blocks.
- [ ] **Validation loop**
  - Produce 5k synthetic receipts, review manually with stakeholders, tweak templates.

### Week 3 – Realism + QA
- [ ] **Augmentations pack**
  - Add thermal fade, streaks, coffee stains, crumple TPS warp, random crop truncation.
  - Assemble background texture library from scans/CC assets.
- [ ] **Automated QA**
  - Implement overlap detection, OCR sanity (tiny recognizer), charset coverage metrics.
  - Wire results into Hydra logging & W&B (if enabled).
- [ ] **Benchmark impact**
  - Train detector on real vs real+synthetic, report CLEval deltas.

### Week 4+ – Scaling & model experiments
- [ ] **Dataset versioning**
  - Archive batches with manifests, checksums, W&B run links.
  - Schedule generation jobs (cron or Makefile targets) with reproducible seeds.
- [ ] **EfficientNet-B7 baseline**
  - Swap TIMM backbone to EfficientNet-B7, adjust decoder channels, tune LR.
  - Run sanity training, record H-mean, speed, GPU mem.
- [ ] **BiFPN upgrade**
  - Implement BiFPN decoder (start with EfficientDet D0 weights), compare vs UNet.
  - Log fusion weight stats and inference timing.
- [ ] **ViT-S + PAN trial (post-synthetic v2)**
  - Introduce Swin/ViT-S backbone, connect to existing PAN decoder.
  - Monitor memory; consider grad checkpointing.
- [ ] **DETR research branch**
  - Prototype RT-DETR/DINO for polygon regression on subset.
  - Decide on evaluation metric (control points vs polygon fit), capture learnings.

### Future / continuous
- [ ] Expand synthetic pipelines as recognition plans solidify (e.g., add transcript QA, per-field labeling).
- [ ] Revisit ViT/PAN & DETR once recognition-ready data proves stable.
- [ ] Integrate results into documentation/readiness checklist for AI hand-off.

## Quality & reporting checkpoints
- Every week: share synthetic QA metrics + sample grids.
- Model experiments: log CLEval (precision/recall/H-mean), FPS, GPU memory, training duration.
- Gate new datasets with QA thresholds (no >1% overlapping polygons, OCR sanity >90%, etc.).
- Maintain a changelog summarizing what’s checked off (use this roadmap as the canonical checklist).
