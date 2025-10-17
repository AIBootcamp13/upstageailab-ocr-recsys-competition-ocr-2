from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Subset

from .base import ValidatedOCRDataset  # noqa: F401
from .craft_collate_fn import CraftCollateFN  # noqa: F401
from .db_collate_fn import DBCollateFN  # noqa: F401
from .preprocessing import DocumentPreprocessor, LensStylePreprocessorAlbumentations  # noqa: F401
from .transforms import DBTransforms  # noqa: F401


def get_datasets_by_cfg(datasets_config, data_config=None, full_config=None):
    train_dataset = instantiate(datasets_config.train_dataset)
    val_dataset = instantiate(datasets_config.val_dataset)
    test_dataset = instantiate(datasets_config.test_dataset)

    # Handle custom image directory for prediction
    if full_config is not None and hasattr(full_config, "image_dir") and full_config.image_dir is not None:
        # Override predict_dataset to use custom image directory
        predict_config = OmegaConf.create(OmegaConf.to_container(datasets_config.predict_dataset, resolve=True))
        predict_config.config.image_path = full_config.image_dir
        predict_dataset = instantiate(predict_config)
    else:
        predict_dataset = instantiate(datasets_config.predict_dataset)

    # Apply dataset limiting if configured
    if data_config is not None:
        try:
            if getattr(data_config, "train_num_samples", None) is not None:
                train_limit = min(data_config.train_num_samples, len(train_dataset))
                train_dataset = Subset(train_dataset, range(train_limit))

            if getattr(data_config, "val_num_samples", None) is not None:
                val_limit = min(data_config.val_num_samples, len(val_dataset))
                val_dataset = Subset(val_dataset, range(val_limit))

            if getattr(data_config, "test_num_samples", None) is not None:
                test_limit = min(data_config.test_num_samples, len(test_dataset))
                test_dataset = Subset(test_dataset, range(test_limit))
        except (AttributeError, KeyError):
            # If config access fails, use full datasets
            pass

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "predict": predict_dataset,
    }
