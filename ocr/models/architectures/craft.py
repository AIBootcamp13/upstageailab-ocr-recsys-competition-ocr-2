"""Registration of CRAFT architecture components."""

from __future__ import annotations

from ocr.models.core import registry
from ocr.models.decoder.craft_decoder import CraftDecoder
from ocr.models.encoder.craft_vgg import CraftVGGEncoder
from ocr.models.head.craft_head import CraftHead
from ocr.models.loss.craft_loss import CraftLoss


def register_craft_components() -> None:
    registry.register_encoder("craft_vgg", CraftVGGEncoder)
    registry.register_decoder("craft_decoder", CraftDecoder)
    registry.register_head("craft_head", CraftHead)
    registry.register_loss("craft_loss", CraftLoss)

    registry.register_architecture(
        name="craft",
        encoder="craft_vgg",
        decoder="craft_decoder",
        head="craft_head",
        loss="craft_loss",
    )


register_craft_components()
