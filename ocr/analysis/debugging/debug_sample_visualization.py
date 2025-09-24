"""Quick debug utility to visualize a few training samples after augmentation
and the generated EAST score/geo maps. Helps diagnose size mismatches or
coordinate explosions leading to near-zero H-Mean.

Usage (from repo root):
  python -m scripts.debug_sample_visualization --config-name=config_tiny \
      data.input_size=512  # or desired size

It will: (1) instantiate the training dataset with current transforms, (2) pull N samples,
(3) save side-by-side composite images to outputs/debug_samples/.
"""
from __future__ import annotations
import os
import sys
import math
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import cv2
import torch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from ocr.datasets import OCRDataset
from ocr.datasets.transforms import DBTransforms

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    out_dir = os.path.join(os.getcwd(), "outputs", "debug_samples")
    os.makedirs(out_dir, exist_ok=True)
    print("Saving debug samples to:", out_dir)
    # Build a small dataset view at declared input size
    ds = EASTDataset(
        SceneTextDataset(cfg.data, cfg.transforms, is_train=True, size_override=cfg.data.input_size),
        map_scale=0.25,
        shrink_coef=float(getattr(cfg.training, "shrink_coef", 0.3)),
    )
    n = min(4, len(ds))
    for i in range(n):
        img_t, score_map, geo_map, roi_mask = ds[i]
        img = img_t.numpy().transpose(1,2,0)
        img_vis = (np.clip(img * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
        sm = score_map.numpy()[0]
        sm_color = cv2.applyColorMap((sm*255).astype(np.uint8), cv2.COLORMAP_JET)
        rm = roi_mask.numpy()[0]
        rm_color = cv2.applyColorMap((rm*255).astype(np.uint8), cv2.COLORMAP_BONE)
        # Geometry heat (mean of distances)
        d_mean = geo_map.numpy()[:4].mean(axis=0)
        d_m = d_mean.astype(np.float32)
        if d_m.size:
            mn, mx = float(d_m.min()), float(d_m.max())
            if mx - mn < 1e-6:
                d_norm = np.zeros_like(d_m)
            else:
                d_norm = (d_m - mn) / (mx - mn)
        else:
            d_norm = d_m
        d_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        # Compose panels
        h0, w0 = img_vis.shape[:2]
        def resize(x):
            return cv2.resize(x, (w0, h0), interpolation=cv2.INTER_NEAREST)
        comp = np.concatenate(
            [img_vis, resize(sm_color), resize(rm_color), resize(d_color)], axis=1
        )
        cv2.putText(comp, f"idx={i}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"sample_{i}.jpg"), comp)
        print(f"Wrote sample_{i}.jpg")
    print("Done.")

if __name__ == "__main__":
    main()
