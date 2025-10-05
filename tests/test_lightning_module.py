from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from ocr.lightning_modules.ocr_pl import OCRPLModule
from ocr.utils.orientation import remap_polygons


class DummyModel:
    def get_optimizers(self):
        return None


class DummyDataset:
    def __init__(self, anns, image_path=None):
        self.anns = anns
        self.image_path = image_path


def test_on_test_epoch_end_handles_missing_predictions(monkeypatch):
    dataset = {
        "test": DummyDataset(OrderedDict({"sample.jpg": [np.zeros((1, 8))]})),
    }
    module = OCRPLModule(model=DummyModel(), dataset=dataset, config=SimpleNamespace())
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)

    module.test_step_outputs.clear()

    module.on_test_epoch_end()

    assert module.test_step_outputs == {}


def test_on_validation_epoch_end_produces_scores_with_orientation(monkeypatch, tmp_path):
    filename = "sample.jpg"

    raw_width, raw_height = 100, 200
    raw_polygon = np.array(
        [
            [10.0, 20.0],
            [40.0, 20.0],
            [40.0, 50.0],
            [10.0, 50.0],
        ],
        dtype=np.float32,
    ).reshape(1, -1, 2)

    gt_polygons = [raw_polygon]

    orientation = 6
    canonical_polygon = remap_polygons([raw_polygon], raw_width, raw_height, orientation)[0].reshape(-1, 2)

    dataset = {
        "val": DummyDataset(OrderedDict({filename: gt_polygons}), image_path=tmp_path),
    }

    module = OCRPLModule(model=DummyModel(), dataset=dataset, config=SimpleNamespace())
    logged: dict[str, float] = {}

    def fake_log(name, value, **kwargs):
        logged[name] = float(value)

    monkeypatch.setattr(module, "log", fake_log)

    module.validation_step_outputs[filename] = {
        "boxes": [canonical_polygon],
        "orientation": orientation,
        "raw_size": (raw_width, raw_height),
    }

    module.on_validation_epoch_end()

    assert module.validation_step_outputs == {}
    assert logged.get("val/precision", 0.0) > 0.99
    assert logged.get("val/recall", 0.0) > 0.99
    assert logged.get("val/hmean", 0.0) > 0.99
