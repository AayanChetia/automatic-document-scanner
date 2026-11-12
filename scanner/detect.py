"""
detect.py - find the document's four corners in the edge image.

Pipeline stage 2. Strategy, in order of preference:

    1. Contour method: trace closed curves, keep the big ones, approximate each
       to a polygon, and accept the largest convex 4-sided shape THAT PASSES
       VALIDATION.
    2. Hough fallback: if no clean quad is found, detect straight lines and
       intersect them to recover four corners - again, only if VALIDATED.
    3. Full-image fallback: if everything fails validation, return the image's
       own corners with found=False and confidence 0 so the caller knows this is
       a low-confidence guess (and the UI can ask the user to adjust corners).

Validation is shared by BOTH detectors so neither can return a nonsense quad
(e.g. a thin sliver along newspaper text columns). A quad is rejected when it is
too small, non-convex, has skewed interior angles, or warps to an extreme aspect
ratio. See ``validate_quad``.

Each function returns corners as a float32 array of shape (4, 2) in (x, y).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .transform import order_points

# --- Validation thresholds (used by both detectors) ----------------------- #
MIN_AREA_RATIO = 0.20     # quad must cover >= 20% of the frame
ANGLE_MIN = 45.0          # interior angles must stay within [45, 135] degrees
ANGLE_MAX = 135.0
MAX_ASPECT = 4.0          # warped page aspect ratio must be <= 4:1


@dataclass
class DetectionResult:
    """What the detector found, plus how/how-well it found it."""

    corners: np.ndarray            # (4, 2) float32, in processing-image coords
    method: str                    # "contour" | "hough" | "fallback"
    found: bool                    # True only if a VALIDATED quad was detected
    confidence: float = 0.0        # 0..1 quality score (0 for a fallback)
    metrics: dict | None = None    # area_ratio, angles, aspect, convex
    candidates: list = field(default_factory=list)


def _image_corners(shape: tuple[int, int]) -> np.ndarray:
    """Corners of the whole frame (used as the last-resort fallback quad)."""
    h, w = shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Quad geometry, validation and confidence
# --------------------------------------------------------------------------- #
def _polygon_area(rect: np.ndarray) -> float:
    """Shoelace area of an ordered quad."""
    x, y = rect[:, 0], rect[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _interior_angles(rect: np.ndarray) -> list[float]:
    """Interior angle (degrees) at each corner of an ordered quad."""
    angles = []
    for i in range(4):
        prev_, cur, next_ = rect[i - 1], rect[i], rect[(i + 1) % 4]
        v1, v2 = prev_ - cur, next_ - cur
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            angles.append(0.0)
            continue
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cosang))))
    return angles


def _aspect_ratio(rect: np.ndarray) -> float:
    """Aspect ratio of the warped page (always >= 1), from the side lengths."""
    (tl, tr, br, bl) = rect
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if min(width, height) < 1e-6:
        return 1e9
    return float(max(width / height, height / width))


def quad_metrics(corners: np.ndarray, shape: tuple[int, int]) -> dict:
    """Compute the geometric properties used for validation and confidence."""
    rect = order_points(corners)
    h, w = shape[:2]
    area = _polygon_area(rect)
    return {
        "area_ratio": area / float(h * w) if h * w else 0.0,
        "angles": _interior_angles(rect),
        "aspect": _aspect_ratio(rect),
        "convex": bool(cv2.isContourConvex(rect.astype(np.int32))),
    }


def validate_quad(corners: np.ndarray, shape: tuple[int, int]) -> tuple[bool, dict]:
    """Return (is_valid, metrics). Shared by the contour and Hough detectors.

    A quad is REJECTED if any of these hold:
      * area < MIN_AREA_RATIO of the image  (rejects tiny quads / text slivers)
      * it is not convex                     (a real page outline is convex)
      * any interior angle is outside [ANGLE_MIN, ANGLE_MAX]  (rejects skewed /
        collapsed shapes like a long thin strip along text columns)
      * the warped aspect ratio exceeds MAX_ASPECT  (rejects long thin strips)

    This is the fix for the failure where the Hough fallback returned a sliver
    quad along newspaper columns and the pipeline wrongly reported success.
    """
    m = quad_metrics(corners, shape)
    ok = bool(
        m["area_ratio"] >= MIN_AREA_RATIO
        and m["convex"]
        and all(ANGLE_MIN <= a <= ANGLE_MAX for a in m["angles"])
        and m["aspect"] <= MAX_ASPECT
    )
    return ok, m


def confidence_score(metrics: dict) -> float:
    """Map quad metrics to a 0..1 confidence (how page-like the quad is).

    Blends four sub-scores: how much of the frame it fills, how close its angles
    are to 90 degrees, how rectangular (non-stretched) it is, and convexity.
    """
    area_s = float(np.clip(metrics["area_ratio"] / 0.6, 0.0, 1.0))
    angle_s = float(np.mean([
        np.clip(1.0 - abs(a - 90.0) / 45.0, 0.0, 1.0) for a in metrics["angles"]
    ]))
    aspect_s = float(np.clip(1.0 - (metrics["aspect"] - 1.0) / (MAX_ASPECT - 1.0),
                             0.0, 1.0))
    convex_s = 1.0 if metrics["convex"] else 0.0
    return round(0.30 * area_s + 0.35 * angle_s + 0.25 * aspect_s + 0.10 * convex_s, 3)


# --------------------------------------------------------------------------- #
# Contour detector
# --------------------------------------------------------------------------- #
def find_contour_quad(closed_edges: np.ndarray) -> DetectionResult:
    """Find the document as the largest convex 4-point contour that validates.

    HOW: ``findContours`` traces closed white curves; we sort by area and
    approximate each with ``approxPolyDP`` (Ramer-Douglas-Peucker, epsilon ~2%
    of the perimeter). A candidate must be a convex quad AND pass
    ``validate_quad`` - so a thin sliver or a tiny logo is never accepted.
    """
    contours, _ = cv2.findContours(
        closed_edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, closed=True)
        if perimeter == 0:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, closed=True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        corners = approx.reshape(4, 2).astype(np.float32)
        ok, metrics = validate_quad(corners, closed_edges.shape)
        if ok:
            return DetectionResult(corners=corners, method="contour", found=True,
                                   confidence=confidence_score(metrics),
                                   metrics=metrics, candidates=contours)

    return DetectionResult(corners=_image_corners(closed_edges.shape),
                           method="contour", found=False, confidence=0.0,
                           candidates=contours)


# --------------------------------------------------------------------------- #
# Hough fallback detector
# --------------------------------------------------------------------------- #
def _intersection(line1: np.ndarray, line2: np.ndarray) -> np.ndarray | None:
    """Intersection point of two Hough lines given in (rho, theta) form."""
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

    Only returns found=True if the recovered quad passes ``validate_quad``. This
    is what stops the Hough path from accepting a sliver along text columns.
    """
    fail = DetectionResult(corners=_image_corners(closed_edges.shape),
                           method="hough", found=False, confidence=0.0)

    lines = cv2.HoughLines(closed_edges, 1, np.pi / 180, threshold=120)
    if lines is None or len(lines) < 4:
        return fail

    horizontal, vertical = [], []
    for line in lines[:, 0, :]:
        rho, theta = line
        if theta < np.pi / 4 or theta > 3 * np.pi / 4:
            vertical.append((rho, theta))
        else:
            horizontal.append((rho, theta))
    if len(horizontal) < 2 or len(vertical) < 2:
        return fail

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
        return fail

    corners = np.array(pts, dtype=np.float32)
    ok, metrics = validate_quad(corners, closed_edges.shape)
    if not ok:
        return fail
    return DetectionResult(corners=corners, method="hough", found=True,
                           confidence=confidence_score(metrics), metrics=metrics)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def detect_document(closed_edges: np.ndarray) -> DetectionResult:
    """Full detection strategy with validation and graceful fallbacks.

    Tries the contour method, then Hough; both must pass validation. If neither
    does, returns the full frame with found=False and confidence 0 so the caller
    treats it as a low-confidence guess (never a success).
    """
    result = find_contour_quad(closed_edges)
    if result.found:
        return result

    hough = find_hough_quad(closed_edges)
    if hough.found:
        return hough

    return DetectionResult(corners=_image_corners(closed_edges.shape),
                           method="fallback", found=False, confidence=0.0)
