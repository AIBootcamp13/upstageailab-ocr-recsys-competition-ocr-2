from hydra.utils import instantiate

from .base import OCRDataset  # noqa: F401
from .craft_collate_fn import CraftCollateFN  # noqa: F401
from .db_collate_fn import DBCollateFN  # noqa: F401
from .preprocessing import DocumentPreprocessor, LensStylePreprocessorAlbumentations  # noqa: F401
from .transforms import DBTransforms  # noqa: F401


def get_datasets_by_cfg(config):
    train_dataset = instantiate(config.train_dataset)
    val_dataset = instantiate(config.val_dataset)
    test_dataset = instantiate(config.test_dataset)
    predict_dataset = instantiate(config.predict_dataset)
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "predict": predict_dataset,
    }
