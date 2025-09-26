"""Registration of DBNet architecture components."""

from ocr.models.core import registry
from ocr.models.decoder.unet import UNetDecoder
from ocr.models.encoder.timm_backbone import TimmBackbone
from ocr.models.head.db_head import DBHead
from ocr.models.loss.db_loss import DBLoss


def register_dbnet_components():
    """Register all DBNet architecture components."""

    # Register encoder
    registry.register_encoder("timm_backbone", TimmBackbone)

    # Register decoder
    registry.register_decoder("unet", UNetDecoder)

    # Register head
    registry.register_head("db_head", DBHead)

    # Register loss
    registry.register_loss("db_loss", DBLoss)

    # Register complete DBNet architecture
    registry.register_architecture(
        name="dbnet",
        encoder="timm_backbone",
        decoder="unet",
        head="db_head",
        loss="db_loss",
    )


# Register components when module is imported
register_dbnet_components()
