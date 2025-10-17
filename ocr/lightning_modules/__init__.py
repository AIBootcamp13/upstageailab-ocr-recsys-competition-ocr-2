from ocr.datasets import get_datasets_by_cfg
from ocr.models import get_model_by_cfg

from .ocr_pl import OCRDataPLModule, OCRPLModule


def get_pl_modules_by_cfg(config):
    model = get_model_by_cfg(config.model)
    data_config = getattr(config, "data", None)
    dataset = get_datasets_by_cfg(config.datasets, data_config, config)
    metric_cfg = None
    if "metrics" in config and "eval" in config.metrics:
        metric_cfg = config.metrics.eval

    modules = OCRPLModule(model=model, dataset=dataset, config=config, metric_cfg=metric_cfg)
    data_modules = OCRDataPLModule(dataset=dataset, config=config)
    return modules, data_modules
