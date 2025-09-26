"""Registration of DBNet++ architecture components."""

from __future__ import annotations

from ocr.models.core import registry
from ocr.models.decoder.dbpp_decoder import DBPPDecoder
from ocr.models.encoder.timm_backbone import TimmBackbone
from ocr.models.head.db_head import DBHead
from ocr.models.loss.db_loss import DBLoss


def register_dbnetpp_components() -> None:
    registry.register_encoder("dbnetpp_backbone", TimmBackbone)
    registry.register_decoder("dbnetpp_decoder", DBPPDecoder)
    registry.register_head("dbnetpp_head", DBHead)
    registry.register_loss("dbnetpp_loss", DBLoss)

    registry.register_architecture(
        name="dbnetpp",
        encoder="dbnetpp_backbone",
        decoder="dbnetpp_decoder",
        head="dbnetpp_head",
        loss="dbnetpp_loss",
    )


register_dbnetpp_components()
