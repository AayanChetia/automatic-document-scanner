"""
detect.py - find the document's four corners in the edge image.

Pipeline stage 2. Strategy, in order of preference:

    1. Contour method: trace closed curves, keep the big ones, approximate each
       to a polygon, and accept the largest convex 4-sided shape.
    2. Hough fallback: if no clean quad is found, detect straight lines and
       intersect them to recover four corners.
    3. Full-image fallback: if everything fails, return the image's own corners
       so the rest of the pipeline still runs (and the UI can offer manual
       corner picking).

Each function returns corners as a float32 array of shape (4, 2) in (x, y).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class DetectionResult:
    """What the detector found, plus how it found it (for UI/evaluation)."""

    corners: np.ndarray            # (4, 2) float32, in processing-image coords
    method: str                    # "contour" | "hough" | "fallback"
    found: bool                    # True if a real quad was detected
    candidates: list = field(default_factory=list)  # contours considered


def _image_corners(shape: tuple[int, int]) -> np.ndarray:
    """Corners of the whole frame (used as the last-resort fallback quad)."""
    h, w = shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def find_contour_quad(
    closed_edges: np.ndarray, min_area_ratio: float = 0.1
) -> DetectionResult:
    """Find the document as the largest convex 4-point contour.

    HOW:
      * ``findContours`` traces every closed white curve in the edge image.
      * We sort by contour area (largest first) and only consider the biggest
        few - the page is normally the largest object in a document photo.
      * ``approxPolyDP`` (Ramer-Douglas-Peucker) simplifies each contour to a
        polygon with as few vertices as possible while staying within
        ``epsilon`` of the original. epsilon is set to ~2% of the contour's
        perimeter: big enough to collapse the many little edge wiggles into
        four straight sides, small enough not to lose a real corner.
      * We accept the first approximation that has exactly 4 points, is convex
        (a real page outline is convex) and is large enough to be the document
        rather than a stamp or logo.

    ``min_area_ratio`` rejects tiny quads: the candidate must cover at least
    this fraction of the frame.
    """
    # OpenCV 4/5 return (contours, hierarchy).
    contours, _ = cv2.findContours(
        closed_edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    frame_area = closed_edges.shape[0] * closed_edges.shape[1]

    # Largest contours first - the page is usually the biggest shape.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * frame_area:
            continue
        perimeter = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, closed=True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            corners = approx.reshape(4, 2).astype(np.float32)
            return DetectionResult(corners=corners, method="contour", found=True,
                                   candidates=contours)

    # Nothing suitable found.
    return DetectionResult(
        corners=_image_corners(closed_edges.shape),
        method="contour",
        found=False,
        candidates=contours,
    )


def _intersection(line1: np.ndarray, line2: np.ndarray) -> np.ndarray | None:
    """Intersection point of two lines given in Hough (rho, theta) form.

    Each Hough line satisfies x*cos(theta) + y*sin(theta) = rho. Two such
    equations form a 2x2 linear system A p = b; we solve it for p = (x, y).
    Returns None when the lines are (near) parallel and have no stable crossing.
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    x, y = np.linalg.solve(A, b)
    return np.array([x, y], dtype=np.float32)


def find_hough_quad(closed_edges: np.ndarray) -> DetectionResult:
    """Fallback detector: recover corners from dominant straight lines.

    WHY a fallback: the contour method fails when the page boundary is broken
    (strong shadow, low contrast against the background) so it never forms a
    single closed 4-gon. But a document still has four long straight edges.

    HOW: ``HoughLines`` votes for straight lines in the edge image. We split the
    detected lines into a "vertical-ish" group and a "horizontal-ish" group by
    their angle theta, take the two extreme lines from each group (the outer
    page borders), and intersect the two horizontals with the two verticals to
    get four corners. It is less precise than the contour method, which is why
    it is only the second choice.
    """
    lines = cv2.HoughLines(closed_edges, 1, np.pi / 180, threshold=120)
    if lines is None or len(lines) < 4:
        return DetectionResult(
            corners=_image_corners(closed_edges.shape), method="hough", found=False
        )

    horizontal, vertical = [], []
    for line in lines[:, 0, :]:
        rho, theta = line
        # theta near 0 or pi  -> vertical line; near pi/2 -> horizontal line.
        if theta < np.pi / 4 or theta > 3 * np.pi / 4:
            vertical.append((rho, theta))
        else:
            horizontal.append((rho, theta))

    if len(horizontal) < 2 or len(vertical) < 2:
        return DetectionResult(
            corners=_image_corners(closed_edges.shape), method="hough", found=False
        )

    # Outermost borders = smallest and largest rho in each orientation.
    horizontal.sort(key=lambda l: l[0])
    vertical.sort(key=lambda l: l[0])
    h_lines = [horizontal[0], horizontal[-1]]
    v_lines = [vertical[0], vertical[-1]]

    pts = []
    for h in h_lines:
        for v in v_lines:
            p = _intersection(np.array(h), np.array(v))
            if p is not None:
                pts.append(p)

    if len(pts) != 4:
        return DetectionResult(
            corners=_image_corners(closed_edges.shape), method="hough", found=False
        )

    return DetectionResult(
        corners=np.array(pts, dtype=np.float32), method="hough", found=True
    )


def detect_document(closed_edges: np.ndarray, min_area_ratio: float = 0.1) -> DetectionResult:
    """Full detection strategy with graceful fallbacks.

    Tries the contour method first, then Hough, then the whole frame. The
    caller can inspect ``result.found`` and ``result.method`` to decide whether
    to prompt the user for manual corners (as the Streamlit app does).
    """
    result = find_contour_quad(closed_edges, min_area_ratio)
    if result.found:
        return result

    hough = find_hough_quad(closed_edges)
    if hough.found:
        return hough

    # Last resort: use the whole image so downstream steps still run.
    return DetectionResult(
        corners=_image_corners(closed_edges.shape), method="fallback", found=False
    )
