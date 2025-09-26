from hydra.utils import instantiate

from .unet import UNetDecoder  # noqa: F401

# Backward compatibility
UNet = UNetDecoder


def get_decoder_by_cfg(config):
    decoder = instantiate(config)
    return decoder
