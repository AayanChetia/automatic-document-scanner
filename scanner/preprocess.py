"""
preprocess.py - turn a colour photo into a clean edge map.

Pipeline stage 1. The goal is NOT to understand the document yet, only to
produce a binary edge image in which the page boundary stands out as a closed
contour. The order of operations matters and each one has a reason:

    grayscale -> Gaussian blur -> Canny (auto thresholds) -> morphological close

Every function returns an intermediate so the UI can show the whole chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    """Bundle of intermediate images so the UI/report can show each stage."""

    gray: np.ndarray
    blurred: np.ndarray
    edges: np.ndarray
    closed: np.ndarray


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR -> single-channel grayscale.

    WHY: edges are defined by changes in *intensity*, not colour. Dropping to
    one channel removes colour information we don't need and makes every later
    step ~3x cheaper. Canny itself only works on single-channel images.
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blur(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply a Gaussian blur to suppress high-frequency noise.

    WHY (classic viva point): Canny computes image gradients, and gradients
    amplify noise. A single noisy pixel can create a spurious strong edge. A
    small Gaussian low-pass filter smooths out sensor noise and fine paper
    texture so Canny responds to the *real* page boundary instead of speckle.
    The kernel must be odd; 5x5 is a good balance between denoising and keeping
    the true edges crisp.
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def auto_canny(blurred: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny edge detection with thresholds derived from the image median.

    WHY: Canny needs a low and high threshold for its hysteresis step. Hard-
    coding them (e.g. 50/150) fails whenever lighting changes. Instead we set
    them relative to the median pixel intensity of this particular image:

        lower = max(0,   (1 - sigma) * median)
        upper = min(255, (1 + sigma) * median)

    This adapts automatically to bright and dark photos, which is exactly the
    kind of variation a phone-camera document scanner must handle.
    """
    median = float(np.median(blurred))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(blurred, lower, upper)


def close_edges(edges: np.ndarray, ksize: int = 5, iterations: int = 2) -> np.ndarray:
    """Dilate + morphological close to join broken edge segments.

    WHY: Canny often returns a page outline with small gaps (a shadow, a low-
    contrast corner). ``findContours`` can only trace *closed* curves, so those
    gaps would split the boundary into fragments. Dilation thickens the edges
    and morphological closing (dilate-then-erode) bridges the small breaks,
    turning the four page sides into one continuous contour we can detect.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dilated = cv2.dilate(edges, kernel, iterations=iterations)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    return closed


def preprocess(image: np.ndarray, blur_ksize: int = 5, sigma: float = 0.33) -> PreprocessResult:
    """Run the full preprocessing chain and return every intermediate."""
    gray = to_grayscale(image)
    blurred = blur(gray, blur_ksize)
    edges = auto_canny(blurred, sigma)
    closed = close_edges(edges)
    return PreprocessResult(gray=gray, blurred=blurred, edges=edges, closed=closed)
