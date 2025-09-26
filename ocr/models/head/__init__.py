from hydra.utils import instantiate

from .craft_head import CraftHead  # noqa: F401
from .craft_postprocess import CraftPostProcessor  # noqa: F401
from .db_head import DBHead  # noqa: F401
from .db_postprocess import DBPostProcessor  # noqa: F401


def get_head_by_cfg(config):
    return instantiate(config)
