# ocr/utils/config_validation.py
# needs to be repurposed for text detection

"""Runtime configuration & data sanity checks.

Lightweight warnings to catch common misconfigurations early without aborting training.
Extendable: add new check_* functions that append to the warnings list.
"""
from __future__ import annotations
from typing import List
import os


def validate_runtime(cfg) -> List[str]:
    warns: List[str] = []
    # Data directories
    train_dir = getattr(cfg.data, 'train_dir', None)
    val_dir = getattr(cfg.validation, 'data_dir', None)
    if train_dir and not os.path.isdir(train_dir):
        warns.append(f"Train data_dir missing: {train_dir}")
    if val_dir and not os.path.isdir(val_dir):
        warns.append(f"Validation data_dir missing: {val_dir}")
    # Batch size heuristics
    bs = int(getattr(cfg.data, 'batch_size', 0) or 0)
    if bs <= 0:
        warns.append("Batch size <=0 (did you forget to set data.batch_size?)")
    elif bs % 2 == 1:
        warns.append(f"Odd batch size {bs} may reduce performance on some GPUs.")
    # Model/loss alignment
    model_name = str(getattr(cfg.model, 'name', '')).lower()
    if 'anchorfree' in model_name or 'anchor_free' in model_name:
        # Accept either a 'name' containing anchor_free or a _target_ pointing to anchor_free_loss
        loss_ok = False
        loss_name = str(getattr(cfg.loss, 'name', '')).lower()
        if 'anchor_free' in loss_name or 'anchorfree' in loss_name:
            loss_ok = True
        target_path = getattr(cfg.loss, '_target_', '')
        if isinstance(target_path, str) and 'anchor_free_loss' in target_path:
            loss_ok = True
        if not loss_ok:
            warns.append("Anchor-free model with non anchor-free loss config.")
    # AMP expectations
    if getattr(cfg.training, 'device', 'cpu') == 'cuda':
        amp_flag = True  # AMP auto-enabled in trainer if CUDA available
    else:
        amp_flag = False
    if amp_flag and bool(getattr(cfg.training, 'disable_amp', False)):
        warns.append("AMP disabled explicitly while CUDA selected.")
    # Input size sanity
    inp = int(getattr(cfg.data, 'input_size', 0) or 0)
    if inp and (inp % 8 != 0):
        warns.append(f"Input size {inp} not divisible by 8; FPN strides may misalign.")
    return warns


__all__ = ["validate_runtime"]
import json
from glob import glob

def _read_images_count(data_dir: str) -> int:
    images_dir = os.path.join(data_dir, 'images')
    if not os.path.isdir(images_dir):
        return 0
    exts = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG')
    cnt = 0
    for pat in exts:
        cnt += len(glob(os.path.join(images_dir, pat)))
        if cnt > 0:
            break
    return cnt

def _read_annotation_count(data_dir: str, json_name: str) -> int:
    path = os.path.join(data_dir, 'ufo', json_name)
    if not os.path.exists(path):
        return -1
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return len(data.get('images', {}))
    except Exception:
        return -2

def validate_config_paths(cfg, mode: str):
    """Validate dataset-related paths; raise RuntimeError on hard errors.

    mode: 'train' | 'predict' | 'evaluate'
    """
    if os.environ.get('SKIP_CFG_VALIDATION'):
        return
    errors = []
    warnings = []

    data_dir = getattr(cfg.data, 'data_dir', None)
    if not data_dir:
        errors.append('cfg.data.data_dir missing')
    else:
        if not os.path.isdir(data_dir):
            errors.append(f'data_dir does not exist: {data_dir}')
        img_cnt = _read_images_count(data_dir)
        if img_cnt == 0:
            warnings.append(f'No images found under {data_dir}/images (count=0)')
        if mode == 'train':
            train_json = os.path.join(data_dir, 'ufo', 'train_split.json')
            if not os.path.exists(train_json):
                warnings.append(f'Missing train_split.json at {train_json} (will still proceed if custom loader overrides)')

    if mode in ('train', 'evaluate') and hasattr(cfg, 'validation'):
        val_dir = getattr(cfg.validation, 'data_dir', None)
        gt_json = getattr(cfg.validation, 'gt_json_name', None)
        if not val_dir:
            errors.append('cfg.validation.data_dir missing')
        else:
            if not os.path.isdir(val_dir):
                errors.append(f'validation.data_dir does not exist: {val_dir}')
            if gt_json:
                ann_path = os.path.join(val_dir, 'ufo', gt_json)
                if not os.path.exists(ann_path):
                    errors.append(f'Validation GT json not found: {ann_path}')
        if data_dir and val_dir:
            if 'tiny' in data_dir and 'tiny' not in val_dir:
                errors.append('Tiny training dataset paired with non-tiny validation directory. Set validation=tiny or update validation.data_dir.')
        if data_dir and val_dir and os.path.isdir(os.path.join(data_dir, 'ufo')) and os.path.isdir(os.path.join(val_dir, 'ufo')):
            train_n = _read_annotation_count(data_dir, 'train_split.json')
            val_n = _read_annotation_count(val_dir, gt_json) if gt_json else -1
            if train_n > 0 and val_n > 0 and val_n > 10 * train_n:
                warnings.append(f'Validation set ({val_n}) is >10x larger than train subset ({train_n}); check configs.')

    if mode == 'predict':
        pred_dir = getattr(cfg.predict, 'data_dir', None)
        if pred_dir and not os.path.isdir(pred_dir):
            errors.append(f'predict.data_dir does not exist: {pred_dir}')
        if pred_dir and data_dir and 'tiny' in pred_dir and 'tiny' not in data_dir:
            warnings.append('Predicting on tiny subset while base data config is non-tiny (ok, just noting).')

    if errors:
        msg = 'CONFIG VALIDATION FAILED:\n' + '\n'.join('  - ' + e for e in errors)
        if warnings:
            msg += '\nWarnings:\n' + '\n'.join('  - ' + w for w in warnings)
        raise RuntimeError(msg)
    if warnings:
        print('CONFIG VALIDATION WARNINGS:')
        for w in warnings:
            print('  - ' + w)