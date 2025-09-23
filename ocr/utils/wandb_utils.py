# src/utils/wandb_utils.py
# NEEDS TO BE REPURPOSED FOR TEXT DETECTION

import hashlib
import os

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

import wandb


def generate_run_name(cfg: DictConfig) -> str:
    """Generate a descriptive, stable run name.

    Format (when experiment_tag present):
        <user>_<tag>_<model>-b<bs>-lr<lr>_SCORE_PLACEHOLDER
    Else:
        <user>_<model>-b<bs>-lr<lr>_SCORE_PLACEHOLDER
    """
    user_prefix = cfg.wandb.get("user_prefix", "user")
    model_name = cfg.model.get("name", "model").replace("_", "-")
    batch_size = cfg.data.get("batch_size", "N/A")
    lr_float = cfg.training.get("learning_rate", 0)
    lr_str = f"{lr_float:.0e}".replace("e-0", "e")
    # Prefer wandb.experiment_tag; fall back to top-level or data tag
    tag = (
        cfg.wandb.get("experiment_tag")
        or getattr(cfg, "experiment_tag", None)
        or getattr(cfg.data, "experiment_tag", None)
    )
    if tag:
        tag = str(tag).strip().replace(" ", "-").replace("/", "-")[:40]
    model_details_parts = [model_name, f"b{batch_size}", f"lr{lr_str}"]
    model_details = "-".join(filter(None, model_details_parts))
    if tag:
        core = f"{user_prefix}_{tag}_{model_details}"
    else:
        core = f"{user_prefix}_{model_details}"
    return f"{core}_SCORE_PLACEHOLDER"


def finalize_run(final_loss: float):
    """Updates the run name with the final score and saves summary metrics."""
    if not wandb.run:
        print("W&B run not initialized. Skipping finalization.")
        return

    # 1. Update the run name with the final loss
    current_name = wandb.run.name or "run_SCORE_PLACEHOLDER"
    final_name = current_name.replace("_SCORE_PLACEHOLDER", f"_loss{final_loss:.4f}")
    wandb.run.name = final_name
    print(f"Finalized run name: {final_name}")

    # 2. Update the summary to pin key metrics to the overview
    wandb.summary["final_mean_loss"] = final_loss
    wandb.summary["final_run_name"] = final_name

    wandb.finish()


def log_validation_images(
    images, gt_bboxes, pred_bboxes, epoch, limit=8, seed: int = 42, filenames=None
):
    """Logs images with ground truth (green) and predicted (red) boxes to W&B.

    Adds a compact legend overlay and samples up to `limit` images with a fixed seed
    for diversity across epochs.
    """
    if not wandb.run:
        return

    log_images = []
    drawn_images = []
    sizes = []
    captions = []
    # Rank images to prefer those with BOTH GT and PRED present; then fall back
    N = min(len(images), len(gt_bboxes), len(pred_bboxes))
    pairs = []
    for i in range(N):
        g = len(gt_bboxes[i]) if gt_bboxes[i] is not None else 0
        p = len(pred_bboxes[i]) if pred_bboxes[i] is not None else 0
        pairs.append((i, g, p))

    # Deterministic shuffle per epoch
    try:
        epoch_int = int(epoch)
        local_seed = seed + max(0, epoch_int)
    except Exception:
        local_seed = seed
    rng = np.random.RandomState(local_seed)
    rng.shuffle(pairs)

    both = [i for (i, g, p) in pairs if g > 0 and p > 0]
    only_gt = [i for (i, g, p) in pairs if g > 0 and p == 0]
    only_pred = [i for (i, g, p) in pairs if g == 0 and p > 0]

    ordered = both + only_gt + only_pred
    idxs = ordered[: min(limit, len(ordered))]

    # For optional integrity table
    table_rows = []
    for rank, i in enumerate(idxs):
        image, gt_boxes, pred_boxes = images[i], gt_bboxes[i], pred_bboxes[i]

        # Convert to a proper RGB uint8 image for drawing and logging
        if torch.is_tensor(image):
            # image expected as CHW normalized with mean=std=0.5
            arr = image.detach().cpu().float().numpy()  # C,H,W
            if arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))  # H,W,C
            # Un-normalize from (-1,1) back to (0,1)
            arr = arr * 0.5 + 0.5
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        elif isinstance(image, np.ndarray):
            arr = image.copy()
            if arr.dtype != np.uint8:
                # Try to scale float images in [0,1]
                maxv = float(arr.max()) if arr.size > 0 else 1.0
                if maxv <= 1.5:
                    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        else:
            # Fallback: best-effort conversion
            arr = np.array(image)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

        img_to_draw = np.ascontiguousarray(arr)

        # Prepare counts and safe iterables
        g = len(gt_boxes) if gt_boxes is not None else 0
        p = len(pred_boxes) if pred_boxes is not None else 0
        gt_iter = gt_boxes if gt_boxes is not None else []
        pred_iter = pred_boxes if pred_boxes is not None else []

        # Draw ground truth boxes (in green)
        for box in gt_iter:
            box = np.array(box).reshape(4, 2).astype(np.int32)
            cv2.polylines(
                img_to_draw, [box], isClosed=True, color=(0, 255, 0), thickness=2
            )

        # Draw predicted boxes (in red)
        for box in pred_iter:
            box = np.array(box).reshape(4, 2).astype(np.int32)
            cv2.polylines(
                img_to_draw, [box], isClosed=True, color=(255, 0, 0), thickness=2
            )

        # Add small legend (top-left)
        legend_h = 36
        legend_w = 160
        overlay = img_to_draw.copy()
        cv2.rectangle(overlay, (0, 0), (legend_w, legend_h), (0, 0, 0), thickness=-1)
        alpha = 0.35
        cv2.addWeighted(overlay, alpha, img_to_draw, 1 - alpha, 0, dst=img_to_draw)
        # Green = GT, Red = Pred
        cv2.line(img_to_draw, (8, 12), (32, 12), (0, 255, 0), 3)
        cv2.putText(
            img_to_draw,
            "GT",
            (38, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.line(img_to_draw, (8, 26), (32, 26), (255, 0, 0), 3)
        cv2.putText(
            img_to_draw,
            "Pred",
            (38, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Ensure image is uint8 and in [0,255] range
        img_uint8 = np.clip(img_to_draw, 0, 255).astype(np.uint8)
        drawn_images.append(img_uint8)
        sizes.append(img_uint8.shape[:2])  # (H, W)
        # Filename & original dims if provided.
        # NOTE: Use original sampled index `i` (not the display ordering `rank`).
        # Previous implementation used `rank`, which reorders images by (GT/PRED presence)
        # causing filename/image mismatches in W&B captions.
        fname = "(unknown)"
        orig_w = -1
        orig_h = -1
        if filenames and i < len(filenames):
            fname, orig_w, orig_h = filenames[i]
            meta_prefix = f"{fname} ({orig_w}x{orig_h})"
        else:
            meta_prefix = "(unknown)"
        caption = f"{meta_prefix} | Ep {epoch} | GT={g} Pred={p}"
        captions.append(caption)
        if os.environ.get("LOG_VAL_IMAGE_TABLE", "0") == "1":
            # Lightweight perceptual fingerprint: SHA1 of raw bytes
            img_hash = hashlib.sha1(img_uint8.tobytes()).hexdigest()[:12]
            table_rows.append(
                [
                    rank,  # display order
                    i,  # original sample index
                    fname if (filenames and i < len(filenames)) else "(unknown)",
                    orig_w if (filenames and i < len(filenames)) else -1,
                    orig_h if (filenames and i < len(filenames)) else -1,
                    g,
                    p,
                    img_uint8.shape[1],
                    img_uint8.shape[0],  # logged W,H
                    img_hash,
                    caption,
                ]
            )

    # Pad images to the same size to avoid W&B UI warnings
    if drawn_images:
        max_h = max(h for h, _ in sizes)
        max_w = max(w for _, w in sizes)
        for idx, img in enumerate(drawn_images):
            h, w = img.shape[:2]
            pad_bottom = max_h - h
            pad_right = max_w - w
            if pad_bottom > 0 or pad_right > 0:
                img_padded = cv2.copyMakeBorder(
                    img,
                    0,
                    pad_bottom,
                    0,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            else:
                img_padded = img
            log_images.append(wandb.Image(img_padded, caption=captions[idx]))

    if log_images:
        payload = {"validation_images": log_images}
        if table_rows:
            table = wandb.Table(
                columns=[  # type: ignore[arg-type]
                    "display_rank",
                    "orig_index",
                    "filename",
                    "orig_w",
                    "orig_h",
                    "gt_count",
                    "pred_count",
                    "logged_w",
                    "logged_h",
                    "sha1_12",
                    "caption",
                ],
                data=table_rows,
            )
            payload["validation_image_table"] = table  # type: ignore[assignment]
        wandb.log(payload)
