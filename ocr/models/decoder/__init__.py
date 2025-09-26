from hydra.utils import instantiate

from .unet import UNet  # noqa: F401


def get_decoder_by_cfg(config):
    decoder = instantiate(config)
    return decoder
