"""
Plain-Python + Pillow image validation for fridge photos.

No AI, no segmentation, no YOLO here -- this module only answers the
question "is this a real, openable, reasonably-sized JPEG/PNG/WEBP file?"
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.config import MAX_IMAGE_SIZE_MB

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
MIN_DIMENSION_PX = 50
MAX_DIMENSION_PX = 8000


class ImageValidationError(Exception):
    """Base class for all fridge-image validation failures."""


class ImageNotFoundError(ImageValidationError):
    pass


class UnsupportedImageFormatError(ImageValidationError):
    pass


class CorruptedImageError(ImageValidationError):
    pass


class ImageTooLargeError(ImageValidationError):
    pass


class ImageDimensionsError(ImageValidationError):
    pass


def validate_image(image_path: str | Path) -> Path:
    """
    Validate a fridge image file and return its resolved Path if valid.

    Checks (in order): exists, file size limit, not corrupted/openable,
    supported format (by actual content, not file extension), and
    reasonable pixel dimensions.

    Raises a subclass of ImageValidationError describing the first
    problem found.
    """
    path = Path(image_path)

    if not path.exists() or not path.is_file():
        raise ImageNotFoundError(f"Image file not found: {path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb <= 0:
        raise CorruptedImageError(f"Image file is empty: {path}")
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise ImageTooLargeError(
            f"Image is {size_mb:.2f} MB, which exceeds the {MAX_IMAGE_SIZE_MB} MB limit "
            "(set MAX_IMAGE_SIZE_MB to change this)."
        )

    # Image.verify() checks the file isn't truncated/corrupted, but it
    # invalidates the file object afterwards, so we must reopen to read
    # format/size.
    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise CorruptedImageError(f"Image could not be verified (corrupted?): {path} ({exc})") from exc

    try:
        with Image.open(path) as img:
            image_format = (img.format or "").upper()
            width, height = img.size
    except (UnidentifiedImageError, OSError) as exc:
        raise CorruptedImageError(f"Image could not be reopened after verification: {path} ({exc})") from exc

    if image_format not in SUPPORTED_FORMATS:
        raise UnsupportedImageFormatError(
            f"Unsupported image format '{image_format}'. Supported formats: {sorted(SUPPORTED_FORMATS)}"
        )

    if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
        raise ImageDimensionsError(
            f"Image is too small ({width}x{height}px, minimum is {MIN_DIMENSION_PX}x{MIN_DIMENSION_PX}px)."
        )
    if width > MAX_DIMENSION_PX or height > MAX_DIMENSION_PX:
        raise ImageDimensionsError(
            f"Image is too large ({width}x{height}px, maximum is {MAX_DIMENSION_PX}x{MAX_DIMENSION_PX}px)."
        )

    return path
