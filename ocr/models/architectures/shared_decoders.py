"""Registration of decoder components shared across architectures."""

from ocr.models.core import registry
from ocr.models.decoder.fpn_decoder import FPNDecoder
from ocr.models.decoder.pan_decoder import PANDecoder


def register_shared_decoders() -> None:
    """Register decoder implementations that can be used across architectures."""

    registry.register_decoder("fpn_decoder", FPNDecoder)
    registry.register_decoder("pan_decoder", PANDecoder)


register_shared_decoders()
