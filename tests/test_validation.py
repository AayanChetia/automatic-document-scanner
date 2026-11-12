"""Regression tests for quad validation (the sliver-quad bug).

Background: on a newspaper photo that filled the frame, the Hough fallback once
returned a thin sliver quad along the text columns, the warp produced a long
thin strip, and the UI still reported success. Detection must now REJECT such
quads via ``validate_quad`` and fall back with found=False / confidence 0.
"""

import cv2
import numpy as np

from scanner.detect import (validate_quad, confidence_score, quad_metrics,
                            detect_document)

SHAPE = (400, 600)  # (h, w)


def _good_quad():
    """A near-rectangular page covering most of the frame."""
    return np.array([[60, 40], [540, 50], [535, 360], [55, 350]], np.float32)


def _sliver_quad():
    """A thin vertical strip along 'text columns' - must be rejected."""
    return np.array([[290, 20], [320, 20], [320, 380], [290, 380]], np.float32)


def test_sliver_quad_is_rejected():
    """The degenerate sliver quad must fail validation (regression)."""
    ok, metrics = validate_quad(_sliver_quad(), SHAPE)
    assert ok is False
    # It fails on both area and aspect.
    assert metrics["area_ratio"] < 0.20
    assert metrics["aspect"] > 4.0


def test_full_frame_quad_is_rejected():
    """The whole-image border must be rejected (it's the frame, not a page).

    Regression for pic7: a candidate covering the entire frame was wrongly
    reported as 'Auto-detected (multi-channel) · 79%'.
    """
    h, w = SHAPE
    frame = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    ok, _ = validate_quad(frame, SHAPE)
    assert ok is False


def test_near_full_frame_quad_is_rejected():
    """A quad covering > ~92% of the image (corners near every border) is rejected."""
    h, w = SHAPE
    m = 5
    q = np.array([[m, m], [w - 1 - m, m], [w - 1 - m, h - 1 - m], [m, h - 1 - m]],
                 np.float32)
    ok, metrics = validate_quad(q, SHAPE)
    assert ok is False
    assert metrics["area_ratio"] > 0.92


def test_good_quad_is_accepted():
    """A sensible page-like quad must pass validation."""
    ok, metrics = validate_quad(_good_quad(), SHAPE)
    assert ok is True
    assert metrics["convex"] is True


def test_tiny_quad_is_rejected():
    """A small quad (<20% of the frame) is rejected even if well-shaped."""
    small = np.array([[10, 10], [80, 10], [80, 80], [10, 80]], np.float32)
    ok, _ = validate_quad(small, SHAPE)
    assert ok is False


def test_confidence_low_for_sliver_high_for_good():
    """Confidence must separate a sliver from a real page."""
    c_bad = confidence_score(quad_metrics(_sliver_quad(), SHAPE))
    c_good = confidence_score(quad_metrics(_good_quad(), SHAPE))
    assert c_bad < 0.5 < c_good


def test_detect_document_rejects_sliver():
    """End-to-end: an image whose only structure is a sliver -> found=False.

    This is the core regression: previously the Hough path accepted the sliver
    along text columns and reported a (false) detection.
    """
    img = np.full((SHAPE[0], SHAPE[1], 3), 20, np.uint8)      # dark background
    cv2.rectangle(img, (290, 20), (320, 380), (235, 235, 235), -1)  # thin strip
    result = detect_document(img)
    assert result.found is False
    assert result.confidence == 0.0
    assert result.method == "fallback"
