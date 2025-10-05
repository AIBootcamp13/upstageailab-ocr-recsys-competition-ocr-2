from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
import torch

from ocr.metrics.cleval_metric import CLEvalMetric


def test_calculate_rph_handles_zero_totals() -> None:
    metric = CLEvalMetric()
    calculate_rph = cast(
        Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        metric._CLEvalMetric__calculate_rph,
    )

    recall, precision, hmean = calculate_rph(
        torch.tensor(0, dtype=torch.int32),
        torch.tensor(0, dtype=torch.int32),
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(0, dtype=torch.int32),
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(0, dtype=torch.int32),
    )

    assert recall.item() == pytest.approx(0.0)
    assert precision.item() == pytest.approx(0.0)
    assert hmean.item() == pytest.approx(0.0)


def test_calculate_rph_produces_expected_scores() -> None:
    metric = CLEvalMetric()
    calculate_rph = cast(
        Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        metric._CLEvalMetric__calculate_rph,
    )

    recall, precision, hmean = calculate_rph(
        torch.tensor(10, dtype=torch.int32),
        torch.tensor(8, dtype=torch.int32),
        torch.tensor(1.5, dtype=torch.float32),
        torch.tensor(8, dtype=torch.int32),
        torch.tensor(0.5, dtype=torch.float32),
        torch.tensor(7, dtype=torch.int32),
    )

    assert recall.item() == pytest.approx(0.65)
    assert precision.item() == pytest.approx(0.8125)
    expected_hmean = (2 * 0.65 * 0.8125) / (0.65 + 0.8125)
    assert hmean.item() == pytest.approx(expected_hmean)
