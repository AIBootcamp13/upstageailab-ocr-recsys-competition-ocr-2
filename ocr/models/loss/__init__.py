from hydra.utils import instantiate

from .bce_loss import BCELoss  # noqa: F401
from .db_loss import DBLoss  # noqa: F401
from .dice_loss import DiceLoss  # noqa: F401
from .l1_loss import MaskL1Loss  # noqa: F401


def get_loss_by_cfg(config):
    loss = instantiate(config)
    return loss
