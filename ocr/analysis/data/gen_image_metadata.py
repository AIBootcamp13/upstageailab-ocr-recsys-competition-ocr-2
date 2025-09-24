#!/usr/bin/env python3
"""Generate per-image geometry metadata for bucketed resizing strategies.

Outputs a JSON mapping image filename -> {w,h,short,long,aspect,area,proposed_bucket}.

Bucket proposal heuristics (modifiable):
  short < 360 -> 768
  360 <= short < 480 -> 704
  480 <= short < 640 -> 640
  else -> 640 (no upscale needed)

Usage:
  python scripts/gen_image_metadata.py --images data/ICDAR17_full_dataset/images --out data/ICDAR17_full_dataset/metadata.json

Later you can refine buckets or add median text height (would require parsing UFO GT).
"""
from __future__ import annotations
import argparse, json, os, math
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def propose_bucket(short_side: int) -> int:
    if short_side < 360:
        return 768
    if short_side < 480:
        return 704
    if short_side < 640:
        return 640
    return 640

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='Directory containing images')
    ap.add_argument('--out', required=True, help='Output JSON path')
    ap.add_argument('--exts', nargs='*', default=['.jpg','.jpeg','.png'])
    args = ap.parse_args()

    img_dir = Path(args.images)
    exts = {e.lower() for e in args.exts}
    files = [p for p in img_dir.rglob('*') if p.is_file() and p.suffix.lower() in exts]
    files.sort()

    meta = {}
    hist = {360:0,480:0,640:0,'>640':0}
    for p in tqdm(files, desc='Scanning'):
        try:
            with Image.open(p) as im:
                w,h = im.size
        except (OSError, Image.UnidentifiedImageError):
            continue
        short_side = min(w,h)
        long_side = max(w,h)
        aspect = long_side/short_side if short_side>0 else None
        bucket = propose_bucket(short_side)
        key_bin = 360 if short_side<360 else 480 if short_side<480 else 640 if short_side<640 else '>640'
        hist[key_bin]+=1
        meta[p.name] = dict(w=w,h=h,short=short_side,long=long_side,aspect=aspect,area=w*h,proposed_bucket=bucket)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path,'w') as f:
        json.dump({'meta':meta,'count':len(meta),'histogram':hist}, f, indent=2)
    print(f'Wrote metadata for {len(meta)} images to {out_path}')
    print('Short side histogram:', hist)

if __name__=='__main__':
    main()
