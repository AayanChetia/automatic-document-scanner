"""Hard-case detection tests: white-on-white and a page clipped by the border.

These exercise the multi-cue edge detection, GrabCut fallback and border-aware
quad fitting added for difficult real-world photos.
"""

import cv2
import numpy as np

from scanner import scan_image
from scanner.detect import detect_document
from scanner.transform import order_points


def _quad_iou(a, b, shape):
    """IoU of two quads via rasterised masks (same method as evaluate.py)."""
    h, w = shape[:2]
    ma, mb = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    cv2.fillPoly(ma, [np.asarray(a, np.int32)], 255)
    cv2.fillPoly(mb, [np.asarray(b, np.int32)], 255)
    inter = np.count_nonzero((ma > 0) & (mb > 0))
    union = np.count_nonzero((ma > 0) | (mb > 0))
    return inter / union if union else 0.0


def _white_on_white():
    """A faintly-tinted page on a near-white background (low brightness contrast)."""
    h, w = 520, 420
    img = np.full((h, w, 3), 250, np.uint8)                 # near-white desk
    quad = np.array([[70, 60], [350, 72], [338, 460], [58, 448]], np.float32)
    # Page differs mostly in colour/saturation, only slightly in brightness.
    cv2.fillConvexPoly(img, quad.astype(np.int32), (232, 240, 246))
    cv2.putText(img, "NOTEBOOK", (110, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (120, 120, 120), 2, cv2.LINE_AA)
    return img, order_points(quad)


def _clipped_page():
    """A bright page that runs off the right edge of the frame."""
    h, w = 400, 600
    img = np.full((h, w, 3), 70, np.uint8)                  # mid-grey background
    # Top-right and bottom-right corners are beyond the right border (x > w).
    quad = np.array([[60, 50], [690, 60], [680, 350], [50, 340]], np.float32)
    cv2.fillConvexPoly(img, quad.astype(np.int32), (236, 238, 240))
    cv2.putText(img, "CLIPPED PAGE", (90, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (40, 40, 40), 2, cv2.LINE_AA)
    return img


def test_detects_white_page_on_white_background():
    """Multi-cue detection should find a faint white-on-white page."""
    img, truth = _white_on_white()
    result = scan_image(img, mode="bw")
    assert result.found is True, f"method={result.method}"
    iou = _quad_iou(result.corners_full, truth, img.shape)
    assert iou > 0.6, f"IoU too low: {iou:.2f} (method {result.method})"


def test_detects_page_clipped_by_border():
    """A page running off the frame is detected and flagged as clipped."""
    img = _clipped_page()
    result = scan_image(img, mode="bw")
    assert result.found is True, f"method={result.method}"
    assert result.clipped is True, "page off the frame should be flagged clipped"
    # The right-hand corners should sit on (clamped to) the image border.
    assert result.corners_full[:, 0].max() >= img.shape[1] - 5


def test_detect_document_reports_clipped_flag():
    """detect_document exposes the clipped flag directly on its result."""
    img = _clipped_page()
    det = detect_document(img)
    assert det.found is True
    assert det.clipped is True
