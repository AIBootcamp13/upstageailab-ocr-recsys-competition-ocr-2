from hydra.utils import instantiate

from .db_head import DBHead
from .db_postprocess import DBPostProcessor


def get_head_by_cfg(config):
    head = instantiate(config)
    return head
