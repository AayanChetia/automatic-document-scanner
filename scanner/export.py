"""
export.py - save results as images and combine pages into a PDF.

Pipeline stage 5 (output). Uses Pillow for PDF creation because it can write a
multi-page PDF from a list of images with no external tools - keeping the whole
project free of heavy dependencies.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .utils import save_image


def save_scan(path: str | Path, image: np.ndarray) -> Path:
    """Save a single scan as PNG/JPG (format inferred from the extension)."""
    path = Path(path)
    save_image(path, image)
    return path


def _to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV image (BGR or gray) to a PIL RGB image for PDF export."""
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def images_to_pdf(images: list[np.ndarray], path: str | Path) -> Path:
    """Combine one or more images into a single multi-page PDF.

    WHY Pillow: a real scanner app exports a multi-page PDF (one page per photo).
    Pillow writes this directly - the first image is saved with ``save_all`` and
    the rest are appended - so we need no LaTeX/reportlab/ghostscript. Each
    OpenCV BGR array is converted to RGB first (PIL expects RGB).
    """
    if not images:
        raise ValueError("No images supplied for PDF export.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pil_pages = [_to_pil(img) for img in images]
    first, rest = pil_pages[0], pil_pages[1:]
    first.save(str(path), "PDF", save_all=True, append_images=rest, resolution=200.0)
    return path
