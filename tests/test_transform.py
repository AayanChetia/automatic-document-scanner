"""Tests for corner ordering and the perspective transform."""

import numpy as np
import pytest

from scanner.transform import order_points, four_point_transform


def test_order_points_canonical():
    """Any input order must come out as TL, TR, BR, BL."""
    # Deliberately scrambled order.
    pts = np.array([[200, 10],   # TR
                    [10, 300],    # BL
                    [10, 10],     # TL
                    [200, 300]],  # BR
                   dtype=np.float32)
    ordered = order_points(pts)
    expected = np.array([[10, 10], [200, 10], [200, 300], [10, 300]], np.float32)
    assert np.allclose(ordered, expected)


def test_order_points_is_clockwise_from_tl():
    """TL has the smallest x+y; BR the largest."""
    pts = np.array([[50, 60], [400, 55], [410, 500], [40, 480]], np.float32)
    ordered = order_points(pts)
    sums = ordered.sum(axis=1)
    assert np.argmin(sums) == 0   # TL first
    assert np.argmax(sums) == 2   # BR third


def test_order_points_requires_four():
    """Non-4-point input should raise rather than silently misbehave."""
    with pytest.raises(Exception):
        order_points(np.array([[0, 0], [1, 1], [2, 2]], np.float32))


def test_four_point_transform_output_size():
    """Output dimensions come from the quad's side lengths."""
    img = np.zeros((400, 400, 3), np.uint8)
    quad = np.array([[50, 50], [250, 50], [250, 350], [50, 350]], np.float32)
    warped = four_point_transform(img, quad)
    # Width ~200, height ~300 (within rounding of 1px).
    assert abs(warped.shape[1] - 200) <= 1
    assert abs(warped.shape[0] - 300) <= 1


def test_four_point_transform_recovers_rectangle():
    """A known rectangle warped to a rectangle keeps its content.

    We paint a white rectangle exactly on the source quad, warp through those
    corners, and expect the output to be (almost) entirely white - i.e. the
    transform mapped the quad onto the full output canvas.
    """
    img = np.zeros((400, 400), np.uint8)
    quad = np.array([[60, 40], [330, 40], [330, 360], [60, 360]], np.float32)
    import cv2
    cv2.fillConvexPoly(img, quad.astype(np.int32), 255)
    warped = four_point_transform(img, quad)
    # The interior (avoiding 2px border sampling) should be white.
    inner = warped[3:-3, 3:-3]
    assert inner.mean() > 240
