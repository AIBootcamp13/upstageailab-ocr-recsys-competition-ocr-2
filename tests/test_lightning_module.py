from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from ocr.lightning_modules.ocr_pl import OCRPLModule


class DummyModel:
    def get_optimizers(self):
        return None


class DummyDataset:
    def __init__(self, anns):
        self.anns = anns


def test_on_test_epoch_end_handles_missing_predictions(monkeypatch):
    dataset = {
        "test": DummyDataset(OrderedDict({"sample.jpg": [np.zeros((1, 8))]})),
    }
    module = OCRPLModule(model=DummyModel(), dataset=dataset, config=SimpleNamespace())
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)

    module.test_step_outputs.clear()

    module.on_test_epoch_end()

    assert module.test_step_outputs == {}
