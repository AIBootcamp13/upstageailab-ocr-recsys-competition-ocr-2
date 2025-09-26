from hydra.utils import instantiate

from .db_head import DBHead  # noqa: F401
from .db_postprocess import DBPostProcessor  # noqa: F401


def get_head_by_cfg(config):
    head = instantiate(config)
    return head
