"""Unit tests for the CRAFT architecture components."""

from __future__ import annotations

import numpy as np
import torch

from ocr.models.decoder.craft_decoder import CraftDecoder
from ocr.models.encoder.craft_vgg import CraftVGGEncoder
from ocr.models.head.craft_head import CraftHead
from ocr.models.head.craft_postprocess import CraftPostProcessor
from ocr.models.loss.craft_loss import CraftLoss


def test_craft_encoder_output_shapes():
    encoder = CraftVGGEncoder(pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)

    features = encoder(dummy)

    assert len(features) == 4
    expected_channels = encoder.out_channels
    for feat, channels in zip(features, expected_channels, strict=False):
        assert feat.shape[1] == channels


def test_craft_decoder_shapes():
    decoder = CraftDecoder(in_channels=[64, 128, 256, 512], inner_channels=128, out_channels=192)

    features = [
        torch.randn(2, 64, 112, 112),
        torch.randn(2, 128, 56, 56),
        torch.randn(2, 256, 28, 28),
        torch.randn(2, 512, 14, 14),
    ]

    decoded = decoder(features)
    assert decoded.shape == (2, 192, 112, 112)


def test_craft_head_outputs():
    head = CraftHead(in_channels=192)
    feature_map = torch.randn(2, 192, 112, 112)

    prediction = head(feature_map)

    for key in ("region_logits", "affinity_logits", "region_score", "affinity_score"):
        assert key in prediction
    assert prediction["region_score"].shape == (2, 1, 112, 112)
    assert prediction["affinity_score"].shape == (2, 1, 112, 112)


def test_craft_loss_forward():
    loss_fn = CraftLoss()
    preds = {
        "region_score": torch.rand(2, 1, 64, 64),
        "affinity_score": torch.rand(2, 1, 64, 64),
    }
    gt_region = torch.rand(2, 1, 64, 64)
    gt_affinity = torch.rand(2, 1, 64, 64)

    loss, components = loss_fn(preds, gt_region, gt_affinity)

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert "loss_region" in components
    assert "loss_affinity" in components


def test_craft_postprocessor_simple_box():
    postprocess = CraftPostProcessor(text_threshold=0.2, link_threshold=0.1, low_text=0.1, min_area=4)
    region = torch.zeros(1, 1, 32, 32)
    affinity = torch.zeros(1, 1, 32, 32)
    region[:, :, 8:24, 8:24] = 1.0
    affinity[:, :, 8:24, 8:24] = 1.0

    batch = {"inverse_matrix": [np.eye(3, dtype=np.float32)]}
    boxes, scores = postprocess.represent(batch, {"region_score": region, "affinity_score": affinity})

    assert len(boxes[0]) == 1
    assert len(scores[0]) == 1
    assert scores[0][0] > 0
