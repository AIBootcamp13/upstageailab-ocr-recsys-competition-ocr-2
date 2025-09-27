from . import architectures as _architectures  # noqa: F401
from .architecture import OCRModel


def get_model_by_cfg(config):
    return OCRModel(config)
