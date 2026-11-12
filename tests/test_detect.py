"""Tests for document detection and the full pipeline on a known quad."""

import cv2
import numpy as np

from scanner import scan_image
from scanner.transform import order_points


def _synthetic_doc(bg=40):
    """A white page as a tilted quad on a dark background, with known corners."""
    img = np.full((800, 600, 3), bg, np.uint8)
    quad = np.array([[150, 120], [480, 90], [520, 650], [110, 600]], np.float32)
    cv2.fillConvexPoly(img, quad.astype(np.int32), (235, 235, 235))
    return img, order_points(quad)


def test_detects_page_quad():
    """The pipeline should detect the page via an edge-based method."""
    img, _ = _synthetic_doc()
    result = scan_image(img, mode="bw")
    assert result.found is True
    assert result.method in ("contour", "multi-channel")


def test_detected_corners_near_truth():
    """Detected corners should land close to the true page corners."""
    img, truth = _synthetic_doc()
    result = scan_image(img, mode="bw")
    # Mean corner error should be a few pixels, not tens.
    err = np.linalg.norm(result.corners_full - truth, axis=1).mean()
    assert err < 15


def test_manual_override_is_used():
    """Passing corners_override bypasses detection and reports 'manual'."""
    img, truth = _synthetic_doc()
    result = scan_image(img, mode="bw", corners_override=truth)
    assert result.method == "manual"
    assert result.found is True


def test_pipeline_returns_all_stages():
    """Result must carry every intermediate image for the UI/report."""
    img, _ = _synthetic_doc()
    r = scan_image(img, mode="magic")
    for attr in ("gray", "blurred", "edges", "closed", "contour_overlay",
                 "warped", "enhanced"):
        assert getattr(r, attr) is not None
