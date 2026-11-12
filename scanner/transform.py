"""
transform.py - order the four corners and flatten the page (perspective warp).

Pipeline stage 3. Given four corner points of the document in the photo, we
compute a homography that maps them onto a perfect rectangle and warp the
original pixels through it. This is what removes the camera angle and gives the
flat "scanned from directly above" look.
"""

from __future__ import annotations

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return the 4 points consistently ordered as TL, TR, BR, BL.

    WHY: ``getPerspectiveTransform`` matches source point i to destination
    point i. If our four corners are in a random order (as contour detection
    returns them), the warp will twist or mirror the page. We must impose a
    canonical order first.

    The trick uses coordinate sums and differences:
      * top-left     has the smallest  x + y
      * bottom-right has the largest   x + y
      * top-right    has the smallest  y - x   (i.e. largest x - y)
      * bottom-left  has the largest   y - x   (i.e. smallest x - y)

    This is robust to rotation up to ~45 degrees, which covers normal handheld
    tilt. Returned as float32 because that is what OpenCV's transform expects.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left  -> smallest sum
    ordered[2] = pts[np.argmax(s)]  # bottom-right -> largest sum

    diff = np.diff(pts, axis=1).ravel()  # y - x
    ordered[1] = pts[np.argmin(diff)]  # top-right -> smallest (y - x)
    ordered[3] = pts[np.argmax(diff)]  # bottom-left -> largest (y - x)
    return ordered


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a perspective transform so the quad ``pts`` fills the output.

    Steps:
      1. Order the corners (TL, TR, BR, BL).
      2. Derive the output width/height from the *actual* side lengths of the
         quad, so we don't stretch the page. Width = max(top edge, bottom edge);
         height = max(left edge, right edge). Using max avoids cutting off pixels
         when the near edge is longer than the far edge (perspective fore-
         shortening).
      3. Build the destination rectangle (0,0)..(W-1,H-1).
      4. ``getPerspectiveTransform`` solves for the 3x3 homography H that sends
         each source corner to its destination corner, and ``warpPerspective``
         resamples the whole image through H.

    Returns the warped (deskewed, cropped) document image.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_top = _distance(tr, tl)
    width_bottom = _distance(br, bl)
    max_width = max(int(round(width_top)), int(round(width_bottom)))

    height_left = _distance(bl, tl)
    height_right = _distance(br, tr)
    max_height = max(int(round(height_left)), int(round(height_right)))

    # Guard against degenerate (collinear) quads.
    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped
