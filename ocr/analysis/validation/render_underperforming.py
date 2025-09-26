#!/usr/bin/env python3
"""Regenerate the random sample HTML block for 'Under Performing Predictions' in README.

Usage:
  python scripts/render_underperforming.py --k 4 --folder docs/assets/images/under_performing_predictions

It finds the markers <!-- BEGIN:UNDER_PERFORMING_RANDOM --> ... <!-- END:UNDER_PERFORMING_RANDOM -->
inside README.md and replaces the enclosed block with a newly sampled set of images.

Intended for local use; commit the README after running if you want to freeze a new sample.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

BEGIN_MARK = "<!-- BEGIN:UNDER_PERFORMING_RANDOM -->"
END_MARK = "<!-- END:UNDER_PERFORMING_RANDOM -->"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def build_html_block(img_rel_paths, width=180):
    lines = [BEGIN_MARK, '<p align="center">']
    for p in img_rel_paths:
        lines.append(f'  <img src="{p}" width="{width}"/>')
    lines.append("</p>")
    lines.append(END_MARK)
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--folder", default="docs/assets/images/under_performing_predictions")
    ap.add_argument("--k", type=int, default=4, help="Number of images to sample")
    ap.add_argument("--width", type=int, default=180, help="Image display width")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    img_dir = Path(args.folder)
    assert img_dir.is_dir(), f"Folder not found: {img_dir}"
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    assert images, f"No images found in {img_dir}"

    sample = random.sample(images, k=min(args.k, len(images)))
    # Use relative paths for README portability
    rel_paths = [str(p.as_posix()) for p in sample]

    readme_path = Path(args.readme)
    text = readme_path.read_text(encoding="utf-8")

    if BEGIN_MARK not in text or END_MARK not in text:
        raise RuntimeError("Markers not found in README; ensure they exist before running.")

    prefix, rest = text.split(BEGIN_MARK, 1)
    _old_block, suffix = rest.split(END_MARK, 1)

    new_block = build_html_block(rel_paths, width=args.width)
    new_text = prefix + new_block + suffix
    readme_path.write_text(new_text, encoding="utf-8")
    print(f"Updated {readme_path} with {len(sample)} random images.")
    for p in rel_paths:
        print(" -", p)


if __name__ == "__main__":
    main()
