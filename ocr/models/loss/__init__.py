from hydra.utils import instantiate

from .bce_loss import BCELoss
from .db_loss import DBLoss
from .dice_loss import DiceLoss
from .l1_loss import MaskL1Loss


def get_loss_by_cfg(config):
    loss = instantiate(config)
    return loss
