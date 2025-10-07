#!/bin/bash
# Test script to compare validation with and without cache

CHECKPOINT="/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/outputs/canonical-fix2-dbnet-fpn_decoder-mobilenetv3_small_050/checkpoints/last.ckpt"

echo "========================================="
echo "Running validation WITH cache enabled"
echo "========================================="
uv run python runners/test.py \
  data=canonical \
  checkpoint_path="$CHECKPOINT" \
  callbacks=performance_profiler \
  callbacks.performance_profiler.verbose=true \
  +data.polygon_cache.enabled=true \
  +data.polygon_cache.max_size=1000 \
  project_name=OCR_Performance_Cache \
  exp_name=with_cache_validation

echo ""
echo "========================================="
echo "Validation complete!"
echo "========================================="
echo ""
echo "Check WandB for performance metrics comparison:"
echo "- Without cache (baseline): 16.29s validation time"
echo "- With cache (expected): ~2-3s validation time (5-8x speedup)"
