"""
pipeline.py - the end-to-end scan, wiring every stage together.

This is the single entry point the CLI, the Streamlit app and the evaluation
all call. It keeps the "detect small, warp big" trick in one place and returns
a rich result object so the UI can display every intermediate image and the
evaluator can read the detected corners and timing.

    load -> resize(ratio) -> preprocess -> detect -> scale corners up
         -> warp original -> enhance -> result
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import detect as _detect
from . import enhance as _enhance
from . import preprocess as _preprocess
from .transform import four_point_transform, order_points
from .utils import resize_with_ratio


@dataclass
class PipelineResult:
    """Everything produced by one scan - images + metadata.

    Carrying the intermediates (gray/blurred/edges/contour/warped) is what lets
    the Streamlit UI show the whole chain and the report explain each step.
    """

    original: np.ndarray
    gray: np.ndarray
    blurred: np.ndarray
    edges: np.ndarray
    closed: np.ndarray
    contour_overlay: np.ndarray    # original with the detected quad drawn on it
    warped: np.ndarray
    enhanced: np.ndarray
    corners_full: np.ndarray       # (4,2) corners on the FULL-res image, ordered
    method: str                    # how the document was detected
    found: bool                    # whether a VALIDATED quad was detected
    confidence: float              # 0..1 detection confidence (0 for a fallback)
    clipped: bool                  # True if the page extends beyond the frame
    mode: str                      # enhancement mode applied


def _draw_quad(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Return a copy of ``image`` with the detected quad outlined in green."""
    overlay = image.copy()
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    for (x, y) in corners.astype(int):
        cv2.circle(overlay, (int(x), int(y)), 8, (0, 0, 255), -1)
    return overlay


def scan_image(
    image: np.ndarray,
    mode: str = "bw",
    process_height: int = 450,
    corners_override: np.ndarray | None = None,
) -> PipelineResult:
    """Run the full document-scan pipeline on one BGR image.

    Parameters
    ----------
    image : np.ndarray
        Input photo in OpenCV BGR format (full resolution).
    mode : str
        Enhancement mode (see ``scanner.enhance.MODES``).
    process_height : int
        Height the image is scaled to for detection. Detection deliberately runs
        at a modest resolution so fine text edges fade out (via the blur) while
        the long page edges survive; the warp always uses the original pixels
        (via the scale ratio), so output quality is unaffected.
    corners_override : np.ndarray | None
        If given (shape (4,2), in FULL-resolution coordinates), skip automatic
        detection and use these corners. This is how the Streamlit app applies
        manual/slider corner adjustments.

    Returns
    -------
    PipelineResult with every intermediate image and metadata.
    """
    # 1. Resize for fast, robust detection but remember the scale ratio.
    small, ratio = resize_with_ratio(image, height=process_height)

    # 2. Preprocess the small image: gray -> blur -> edges -> close.
    pre = _preprocess.preprocess(small)

    if corners_override is not None:
        corners_full = order_points(corners_override)
        method, found, confidence, clipped = "manual", True, 1.0, False
    else:
        # 3. Detect the document on the small COLOUR image (multi-cue + GrabCut).
        det = _detect.detect_document(small)
        method, found, confidence, clipped = (det.method, det.found,
                                              det.confidence, det.clipped)
        # 4. Scale the detected corners back up to the ORIGINAL resolution,
        #    so the warp samples full-quality pixels.
        corners_full = order_points(det.corners * ratio)

    # 5. Warp the ORIGINAL (full-res) image through the perspective transform.
    warped = four_point_transform(image, corners_full)

    # 6. Enhance the flattened page into the chosen output style.
    enhanced = _enhance.enhance(warped, mode=mode)

    # Build an overlay on the full-res original for display.
    contour_overlay = _draw_quad(image, corners_full)

    return PipelineResult(
        original=image,
        gray=pre.gray,
        blurred=pre.blurred,
        edges=pre.edges,
        closed=pre.closed,
        contour_overlay=contour_overlay,
        warped=warped,
        enhanced=enhanced,
        corners_full=corners_full,
        method=method,
        found=found,
        confidence=confidence,
        clipped=clipped,
        mode=mode,
    )
