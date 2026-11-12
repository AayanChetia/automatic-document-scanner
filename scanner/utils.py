"""
utils.py - small shared helpers used across the pipeline.

Nothing here is document-specific; these are generic image/IO/timing helpers.
Keeping them in one place avoids repeating boilerplate (and makes the other
modules read like the algorithm they implement).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Image loading / saving
# --------------------------------------------------------------------------- #
def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as a BGR uint8 array.

    We use cv2.imread because the whole pipeline works in OpenCV's native BGR
    colour order. Raising immediately on failure (instead of returning None,
    which is OpenCV's default) means callers get a clear error instead of a
    confusing crash three steps later.
    """
    path = str(path)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path!r}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save an image, creating the parent folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write image: {path}")


# --------------------------------------------------------------------------- #
# Resizing (keep the ratio so we can map results back to full resolution)
# --------------------------------------------------------------------------- #
def resize_with_ratio(
    image: np.ndarray, height: int = 500
) -> tuple[np.ndarray, float]:
    """Resize to a fixed processing *height* and return (resized, ratio).

    WHY: edge detection and contour finding are much faster and more robust on
    a small image (noise and tiny texture edges shrink away). But we want the
    final, high-quality warp to use the ORIGINAL full-resolution pixels. So we
    detect the document on the small image, then multiply the found corners by
    ``ratio = original_height / processing_height`` to place them back on the
    original. This is the standard trick behind sharp scanner output.

    Returns the resized image and the scale ratio (original / resized).
    """
    orig_h = image.shape[0]
    if orig_h <= height:
        # Already small enough; ratio of 1.0 keeps corners unchanged.
        return image.copy(), 1.0
    ratio = orig_h / float(height)
    new_w = int(round(image.shape[1] / ratio))
    resized = cv2.resize(image, (new_w, height), interpolation=cv2.INTER_AREA)
    return resized, ratio


def to_display(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or single-channel image to RGB for matplotlib/Streamlit.

    OpenCV is BGR; most display libraries expect RGB. Grayscale/binary images
    are promoted to 3 channels so everything can be stacked uniformly.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def stack_side_by_side(
    images: list[np.ndarray], height: int = 300
) -> np.ndarray:
    """Resize a list of images to a common height and concatenate horizontally.

    Handy for debugging and for the "every intermediate step" view. Grayscale
    images are promoted to 3 channels so they can sit next to colour ones.
    """
    row = []
    for img in images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        scale = height / img.shape[0]
        resized = cv2.resize(img, (int(img.shape[1] * scale), height))
        row.append(resized)
    return np.hstack(row)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
@contextmanager
def timer() -> Iterator[list[float]]:
    """Context manager that measures wall-clock time in seconds.

    Usage:
        with timer() as t:
            do_work()
        elapsed = t[0]

    Used by evaluate.py to report average processing time per image.
    """
    result: list[float] = [0.0]
    start = time.perf_counter()
    try:
        yield result
    finally:
        result[0] = time.perf_counter() - start
