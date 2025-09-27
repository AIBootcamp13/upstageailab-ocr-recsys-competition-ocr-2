#!/usr/bin/env python3
"""Recommend bucket thresholds & target sizes from image metadata.

Input metadata JSON format (produced by scripts/gen_image_metadata.py):
  {"meta": {"img_0001.jpg": {"w":...,"h":...,"short":...,"long":...,"aspect":...,"area":...}, ...}}

Heuristics:
- Compute short-side quantiles (q25,q50,q75) and propose thresholds just above them.
- Each bucket target size chosen so that average short side within bucket scales to \
  ~desired_short_target (default 640) or higher for tiny.
- Optionally clamp max upscale factor.

Usage:
  python -m scripts.recommend_buckets \
    --metadata data/ICDAR17_full_dataset_tiny_small/metadata.json

Output: YAML snippet printed to stdout you can paste into training config.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument(
        "--desired_short",
        type=int,
        default=640,
        help="Base desired short-side after resize for median bucket",
    )
    ap.add_argument(
        "--tiny_boost",
        type=float,
        default=1.3,
        help="Multiplier for smallest bucket target size",
    )
    ap.add_argument(
        "--max_upscale",
        type=float,
        default=3.0,
        help="Cap on (target_short / original_short)",
    )
    ap.add_argument(
        "--round_to",
        type=int,
        default=32,
        help="Round target sizes to multiple of this",
    )
    args = ap.parse_args()

    with open(args.metadata) as f:
        meta = json.load(f)["meta"]

    skipped_count = 0
    shorts_list = []
    for v in meta.values():
        if "short" in v and isinstance(v["short"], int | float) and v["short"] > 0:
            shorts_list.append(v["short"])
        else:
            skipped_count += 1
    if skipped_count > 0:
        print(f"Skipped {skipped_count} entries with missing or invalid 'short' values.", file=sys.stderr)
    shorts = np.array(shorts_list, dtype=np.float32)
    if shorts.size == 0:
        print("No valid entries.", file=sys.stderr)
        return
    shorts.sort()
    q25, q50, q75 = np.percentile(shorts, [25, 50, 75])
    thresholds = [int(math.ceil(x)) for x in [q25, q50, q75]]
    thresholds = sorted(dict.fromkeys(thresholds))  # unique preserving order

    # Derive per-bucket sets
    edges = thresholds + [10**9]
    buckets = []
    start = 0
    for edge in edges:
        mask = shorts[(shorts >= start) & (shorts < edge)]
        if mask.size == 0:
            start = edge
            continue
        mean_short = float(mask.mean())
        # Determine target size
        if len(buckets) == 0:
            target_short = args.desired_short * args.tiny_boost
        else:
            # Scale down target_short with more buckets, but do not go below 50% of desired_short
            min_target_short = args.desired_short * 0.5
            scale = max(1.0 - 0.1 * (len(buckets) - 1), 0.5)
            target_short = args.desired_short * scale
            target_short = max(target_short, min_target_short)
            # This prevents target_short from becoming too small with many buckets
        # Prevent excessive upscale
        max_allowed = mean_short * args.max_upscale
        # Round up to multiple
        r = int(math.ceil(target_short / args.round_to) * args.round_to)
        target_short = min(r, max_allowed)
        buckets.append((edge, r, mean_short, mask.size))
        start = edge

    # Append final (>= last threshold) bucket if not already capturing stats
    # (Already handled by edge=1e9 loop)

    # Build sizes list
    sizes = [b[1] for b in buckets]
    # thresholds for config exclude sentinel last edge
    cfg_thresholds = [b[0] for b in buckets[:-1]]

    print("# --- Recommended bucket_resize config snippet ---")
    print("bucket_resize:")
    print("  enabled: true")
    print("  metadata_path: <PATH_TO_METADATA_JSON>")
    print(f"  thresholds: {cfg_thresholds}")
    print(f"  sizes: {sizes}")
    print("  epoch_mode: sequential")
    print("  max_samples_per_bucket: null")
    print("# Bucket details (edge,target_short,mean_short,count):")
    for edge, target, mean_short, count in buckets:
        print(f"#   <{edge:4d}: target={target}  mean_orig={mean_short:.1f}  n={count}")


if __name__ == "__main__":
    main()
