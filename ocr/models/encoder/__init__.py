from hydra.utils import instantiate

from .timm_backbone import TimmBackbone  # noqa: F401


def get_encoder_by_cfg(config):
    encoder = instantiate(config)
    return encoder
