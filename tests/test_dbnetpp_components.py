"""Unit tests for DBNet++ decoder integration."""

from __future__ import annotations

import torch

from ocr.models.decoder.dbpp_decoder import DBPPDecoder


def test_dbpp_decoder_output_shape():
    decoder = DBPPDecoder(in_channels=[64, 128, 256, 512], inner_channels=128, out_channels=128)

    features = [
        torch.randn(2, 64, 128, 128),
        torch.randn(2, 128, 64, 64),
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 512, 16, 16),
    ]

    result = decoder(features)

    assert result.shape == (2, 128, 128, 128)
