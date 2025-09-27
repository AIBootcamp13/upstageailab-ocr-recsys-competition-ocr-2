from hydra.utils import instantiate

from .craft_decoder import CraftDecoder  # noqa: F401
from .dbpp_decoder import DBPPDecoder  # noqa: F401
from .fpn_decoder import FPNDecoder  # noqa: F401
from .pan_decoder import PANDecoder  # noqa: F401
from .unet import UNetDecoder  # noqa: F401

# Backward compatibility
UNet = UNetDecoder


def get_decoder_by_cfg(config):
    return instantiate(config)
