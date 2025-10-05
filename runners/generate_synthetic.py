from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ocr.utils.path_utils import get_path_resolver, setup_paths

setup_paths()

from ocr.synthetic_data.dataset import SyntheticDatasetGenerator  # noqa: E402


def _create_preview_grid(entries: list[dict], output_dir: str | Path) -> None:
    """Create a preview grid of generated images."""
    from PIL import Image

    output_dir = Path(output_dir)
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(exist_ok=True)

    # Take first 9 images for grid
    sample_entries = entries[:9]
    if not sample_entries:
        return

    # Load images
    images = []
    for entry in sample_entries:
        img_path = Path(entry["image_path"])
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            # Resize to thumbnail
            img.thumbnail((128, 128))
            images.append(img)

    if not images:
        return

    # Create grid
    grid_size = min(3, len(images))  # 3x3 max
    grid_width = grid_size * 128
    grid_height = ((len(images) - 1) // grid_size + 1) * 128

    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    for i, img in enumerate(images):
        x = (i % grid_size) * 128
        y = (i // grid_size) * 128
        grid.paste(img, (x, y))

    grid_path = preview_dir / "preview_grid.png"
    grid.save(grid_path)
    print(f"Preview grid saved to {grid_path}")


def _export_manifest(entries: list[dict], output_dir: str | Path, config: DictConfig) -> None:
    """Export dataset manifest with metadata."""
    import json
    from datetime import datetime

    from omegaconf import DictConfig, ListConfig

    class OmegaConfEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, DictConfig | ListConfig):
                return OmegaConf.to_container(obj)
            return super().default(obj)

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "config": OmegaConf.to_container(config),
        "entries": entries,  # entries are already dicts
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, cls=OmegaConfEncoder, ensure_ascii=False, indent=2)

    print(f"Manifest exported to {manifest_path}")


@hydra.main(config_path=str(get_path_resolver().config.config_dir), config_name="synthetic", version_base="1.2")
def generate_synthetic(config: DictConfig) -> None:
    """
    Generate synthetic OCR datasets using configurable backends.

    Args:
        config: Hydra configuration for synthetic generation
    """
    print(f"Generating synthetic dataset with backend: {config.backend}")
    print(f"Output directory: {config.output.root_dir}")

    # Instantiate generator with config
    generator = SyntheticDatasetGenerator(config=config)

    # Generate dataset
    entries = generator.generate_dataset(
        num_images=config.generation.num_images,
        output_dir=config.output.root_dir,
        image_size=config.generation.image_size,
        image_prefix=config.output.image_prefix,
        image_format=config.output.image_format,
        annotation_format=config.output.annotation_format,
        num_text_regions=config.generation.num_text_regions,
        background_type=config.generation.background_type,
    )

    print(f"Generated {len(entries)} synthetic images")

    # Create preview grid
    _create_preview_grid(entries, config.output.root_dir)

    # Export manifest
    _export_manifest(entries, config.output.root_dir, config)

    print("Synthetic dataset generation complete!")


def _create_preview_grid(entries: list[dict], output_dir: str | Path) -> None:
    """Create a preview grid of generated images."""
    from PIL import Image

    output_dir = Path(output_dir)
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(exist_ok=True)

    # Take first 9 images for grid
    sample_entries = entries[:9]
    if not sample_entries:
        return

    # Load images
    images = []
    for entry in sample_entries:
        img_path = Path(entry["image_path"])
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            # Resize to thumbnail
            img.thumbnail((128, 128))
            images.append(img)

    if not images:
        return

    # Create grid
    grid_size = min(3, len(images))  # 3x3 max
    grid_width = grid_size * 128
    grid_height = ((len(images) - 1) // grid_size + 1) * 128

    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    for i, img in enumerate(images):
        x = (i % grid_size) * 128
        y = (i // grid_size) * 128
        grid.paste(img, (x, y))

    grid_path = preview_dir / "preview_grid.png"
    grid.save(grid_path)
    print(f"Preview grid saved to {grid_path}")


if __name__ == "__main__":
    generate_synthetic()
