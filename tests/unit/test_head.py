import torch

from ocr.models.head.db_head import DBHead


def test_db_head_outputs_prob_maps_alias():
    head = DBHead(in_channels=256)
    inputs = torch.randn(2, 256, 32, 32)

    outputs = head(inputs)

    assert "prob_maps" in outputs
    assert "binary_map" in outputs
    assert "binary_logits" in outputs
    # Alias should reference the same tensor object for downstream consistency
    assert outputs["prob_maps"] is outputs["binary_map"]
    assert torch.allclose(torch.sigmoid(outputs["binary_logits"]), outputs["binary_map"], atol=1e-6)
