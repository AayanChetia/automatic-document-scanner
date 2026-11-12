"""
enhance.py - make the flattened page look like a clean scan.

Pipeline stage 4. After warping we have a correctly-shaped but still photo-like
page (uneven lighting, greyish paper). These functions produce the output
"modes" the user can choose between. The B&W / threshold modes are what give
the crisp CamScanner-style look.

Modes:
    original  - the warped colour image, untouched
    gray      - simple grayscale
    magic     - "magic colour": CLAHE contrast boost (keeps colour, pops text)
    bw        - black & white via adaptive Gaussian threshold (default)
    otsu      - black & white via global Otsu threshold (for comparison)
"""

from __future__ import annotations

import cv2
import numpy as np

MODES = ("original", "gray", "magic", "bw", "otsu")


def to_gray(warped: np.ndarray) -> np.ndarray:
    """Grayscale version of the warped page."""
    if warped.ndim == 2:
        return warped
    return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


def magic_color(warped: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """'Magic colour': boost local contrast with CLAHE, keep the colours.

    WHY CLAHE and not plain histogram equalisation: a document photo often has
    uneven lighting (bright near a lamp, dim in a corner). Global equalisation
    would over-amplify one region. CLAHE (Contrast Limited Adaptive Histogram
    Equalisation) equalises *small tiles* independently and clips the histogram
    to limit noise amplification, then interpolates between tiles. We apply it
    to the L (lightness) channel in LAB space so we brighten/contrast the page
    without shifting its hues - text gets darker and paper gets cleaner while
    colours (e.g. a signature, a logo) stay natural.
    """
    if warped.ndim == 2:
        warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def black_and_white(warped: np.ndarray, block_size: int = 15, c: int = 10) -> np.ndarray:
    """Adaptive Gaussian threshold -> crisp bilevel 'scan'.

    WHY adaptive (not a single global threshold): lighting across a page is
    rarely uniform, so one global cut-off either loses faint text in dim areas
    or fills bright areas with black. Adaptive thresholding computes a *local*
    threshold for every pixel from a Gaussian-weighted neighbourhood
    (``block_size`` x ``block_size``) and subtracts a constant ``c``. This keeps
    text readable even when one side of the page is darker - exactly the
    uneven-lighting case a phone scanner faces.

    ``block_size`` must be odd and > 1.
    """
    gray = to_gray(warped)
    # Light blur first so paper texture doesn't become speckle in the output.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c,
    )


def otsu(warped: np.ndarray) -> np.ndarray:
    """Global Otsu threshold -> bilevel image (baseline for comparison).

    Otsu automatically picks the single global threshold that minimises the
    within-class variance of the two pixel groups (foreground/background). It is
    the classic global method and works well under *even* lighting, but tends to
    smear when lighting is uneven - which is exactly what the evaluation
    compares against adaptive thresholding.
    """
    gray = to_gray(warped)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def enhance(warped: np.ndarray, mode: str = "bw") -> np.ndarray:
    """Dispatch to the chosen enhancement mode.

    ``mode`` is one of MODES. Defaults to 'bw' (adaptive threshold) because that
    gives the cleanest document-scan appearance.
    """
    mode = mode.lower()
    if mode == "original":
        return warped
    if mode == "gray":
        return to_gray(warped)
    if mode == "magic":
        return magic_color(warped)
    if mode == "bw":
        return black_and_white(warped)
    if mode == "otsu":
        return otsu(warped)
    raise ValueError(f"Unknown mode {mode!r}. Choose from {MODES}.")
