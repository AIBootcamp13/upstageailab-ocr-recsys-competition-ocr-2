"""Optimized image loading utilities     # Try TurboJPEG for JPEG files if available and enabled
if use_turbojpeg and TURBOJPEG_AVAILABLE and image_path.suffix.lower() in (".jpg", ".jpeg"):
    try:
        jpeg = TurboJPEG()
        with open(image_path, "rb") as f:
            jpeg_data = f.read()

        # Decode JPEG directly to numpy array, then convert to PIL
        img_array = jpeg.decode(jpeg_data)
        # Convert BGR to RGB (TurboJPEG outputs BGR)
        img_array = img_array[:, :, ::-1]
        return Image.fromarray(img_array)
    except Exception as e:
        if not turbojpeg_fallback:
            raise RuntimeError(f"TurboJPEG failed for {image_path} and fallback disabled: {e}")
        logger.debug(f"TurboJPEG failed for {image_path}, falling back to PIL: {e}")
elif not use_turbojpeg and image_path.suffix.lower() in (".jpg", ".jpeg"):
    logger.debug(f"TurboJPEG disabled for {image_path}, using PIL") support."""

import logging
from pathlib import Path
from typing import Any

try:
    from turbojpeg import TurboJPEG

    TURBOJPEG_AVAILABLE = True
except ImportError:
    TURBOJPEG_AVAILABLE = False

from PIL import Image

logger = logging.getLogger(__name__)


def load_image_optimized(image_path: str | Path, use_turbojpeg: bool = True, turbojpeg_fallback: bool = True) -> Image.Image:
    """Load image with TurboJPEG for JPEG files, fallback to PIL.

    Args:
        image_path: Path to the image file
        use_turbojpeg: Whether to try TurboJPEG for JPEG files
        turbojpeg_fallback: Whether to fallback to PIL if TurboJPEG fails

    Returns:
        PIL Image object

    Raises:
        RuntimeError: If image loading fails
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Try TurboJPEG for JPEG files if available
    if TURBOJPEG_AVAILABLE and image_path.suffix.lower() in (".jpg", ".jpeg"):
        try:
            jpeg = TurboJPEG()
            with open(image_path, "rb") as f:
                jpeg_data = f.read()

            # Decode JPEG directly to numpy array, then convert to PIL
            img_array = jpeg.decode(jpeg_data)
            # Convert BGR to RGB (TurboJPEG outputs BGR)
            img_array = img_array[:, :, ::-1]
            return Image.fromarray(img_array)
        except Exception as e:
            logger.debug(f"TurboJPEG failed for {image_path}, falling back to PIL: {e}")

    # Fallback to PIL
    try:
        return Image.open(image_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path}: {e}")


def get_image_loader_info() -> dict[str, Any]:
    """Get information about available image loading backends."""
    return {
        "turbojpeg_available": TURBOJPEG_AVAILABLE,
        "pil_available": True,  # PIL is always available in this environment
    }
