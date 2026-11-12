"""
detect.py - find the document's four corners (multi-cue, score-based).

Pipeline stage 2. Hard cases such as a white page on a white table, or a page
that runs off the image border, defeat a single grayscale-Canny contour search.
So detection now:

  1. Builds a MULTI-CUE edge map: Canny over grayscale AND the LAB a/b colour
     channels AND the HSV saturation channel, plus a morphological gradient,
     OR-combined. A white page on a white desk usually still differs in colour
     or saturation even when brightness is identical.
  2. Generates CANDIDATE quads from several methods:
       - contour on the grayscale edges         (method "contour")
       - contour on the multi-cue edges         (method "multi-channel")
       - Hough line intersections               (method "hough")
       - paper-colour mask (bright + low sat)   (method "paper")
       - GrabCut foreground segmentation        (method "grabcut")
     The paper mask finds a white page on a coloured/dark background; GrabCut is
     the heavy fallback, run only when the cheap methods are weak. A candidate
     that is the whole image border is rejected as the frame, not a page.
  3. VALIDATES every candidate (area/convexity/angles/aspect) and SCORES the
     survivors by area + rectangularity + edge support along the quad sides +
     inside-vs-outside contrast. The best-scoring candidate wins, and its method
     is reported.
  4. Border-aware: a GrabCut quad whose corners hit the frame edge is accepted
     but flagged ``clipped`` ("some corners were estimated").
  5. If nothing validates, falls back to the full image (found=False), and the
     UI asks the user to set corners manually.

Corners are returned as a float32 array of shape (4, 2) in (x, y).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .preprocess import auto_canny, close_edges
from .transform import order_points

# --- Validation thresholds (used by every candidate) ---------------------- #
MIN_AREA_RATIO = 0.20     # quad must cover >= 20% of the frame
MAX_AREA_RATIO = 0.92     # above this it's the whole frame, not a page
ANGLE_MIN = 45.0          # interior angles must stay within [45, 135] degrees
ANGLE_MAX = 135.0
MAX_ASPECT = 4.0          # warped page aspect ratio must be <= 4:1
GOOD_SCORE = 0.78         # above this from a cheap method, skip GrabCut
BORDER_PX = 3             # a corner this close to the edge = touching the border

# Small score bonus so an edge/colour-based quad is preferred over GrabCut when
# both validate (contour/paper-mask are more precise than a segmentation blob).
METHOD_BONUS = {"contour": 0.05, "multi-channel": 0.05, "paper": 0.05,
                "hough": 0.02, "grabcut": 0.0}


@dataclass
class DetectionResult:
    """What the detector found, plus how/how-well it found it."""

    corners: np.ndarray            # (4, 2) float32, in processing-image coords
    method: str                    # contour | multi-channel | hough | grabcut | fallback
    found: bool                    # True only if a VALIDATED quad was chosen
    confidence: float = 0.0        # 0..1 candidate score (0 for a fallback)
    metrics: dict | None = None    # area_ratio, angles, aspect, convex
    clipped: bool = False          # True if the page extends beyond the frame
    candidates: list = field(default_factory=list)


def _image_corners(shape: tuple[int, int]) -> np.ndarray:
    """Corners of the whole frame (used as the last-resort fallback quad)."""
    h, w = shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Quad geometry, validation and (legacy) confidence
# --------------------------------------------------------------------------- #
def _polygon_area(rect: np.ndarray) -> float:
    x, y = rect[:, 0], rect[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _interior_angles(rect: np.ndarray) -> list[float]:
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
    (tl, tr, br, bl) = rect
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if min(width, height) < 1e-6:
        return 1e9
    return float(max(width / height, height / width))


def quad_metrics(corners: np.ndarray, shape: tuple[int, int]) -> dict:
    """Geometric properties used for validation and scoring."""
    rect = order_points(corners)
    h, w = shape[:2]
    area = _polygon_area(rect)
    return {
        "area_ratio": area / float(h * w) if h * w else 0.0,
        "angles": _interior_angles(rect),
        "aspect": _aspect_ratio(rect),
        "convex": bool(cv2.isContourConvex(rect.astype(np.int32))),
    }


def _is_frame(rect: np.ndarray, shape: tuple[int, int]) -> bool:
    """True if the quad is essentially the whole image border (not a page).

    A real document is surrounded by some background, so its quad does NOT hug
    all four image edges. The frame does. We flag a quad as the frame when its
    corners sit within a few pixels of the border on ALL four sides. (A page
    clipped on one side touches only that side, so it is not flagged.)
    """
    h, w = shape[:2]
    b = max(BORDER_PX, int(0.015 * min(h, w)))
    xs, ys = rect[:, 0], rect[:, 1]
    return bool(xs.min() <= b and ys.min() <= b
                and xs.max() >= w - 1 - b and ys.max() >= h - 1 - b)


def validate_quad(corners: np.ndarray, shape: tuple[int, int]) -> tuple[bool, dict]:
    """Return (is_valid, metrics). Rejects tiny / skewed / sliver / frame quads.

    In addition to the geometry checks, a candidate is rejected if it covers
    more than ``MAX_AREA_RATIO`` of the image, or hugs the border on all four
    sides — both mean the detector locked onto the whole frame, not a page.
    """
    m = quad_metrics(corners, shape)
    rect = order_points(corners)
    ok = bool(
        MIN_AREA_RATIO <= m["area_ratio"] <= MAX_AREA_RATIO
        and not _is_frame(rect, shape)
        and m["convex"]
        and all(ANGLE_MIN <= a <= ANGLE_MAX for a in m["angles"])
        and m["aspect"] <= MAX_ASPECT
    )
    return ok, m


def confidence_score(metrics: dict) -> float:
    """Legacy geometry-only confidence (kept for reference/tests)."""
    area_s = float(np.clip(metrics["area_ratio"] / 0.6, 0.0, 1.0))
    angle_s = float(np.mean([
        np.clip(1.0 - abs(a - 90.0) / 45.0, 0.0, 1.0) for a in metrics["angles"]
    ]))
    aspect_s = float(np.clip(1.0 - (metrics["aspect"] - 1.0) / (MAX_ASPECT - 1.0),
                             0.0, 1.0))
    convex_s = 1.0 if metrics["convex"] else 0.0
    return round(0.30 * area_s + 0.35 * angle_s + 0.25 * aspect_s + 0.10 * convex_s, 3)


# --------------------------------------------------------------------------- #
# Multi-cue edge map
# --------------------------------------------------------------------------- #
def multi_channel_edges(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Combine edges from grayscale, LAB a/b, HSV saturation and morph-gradient.

    WHY: a white page on a white desk can have (near) identical brightness, so
    grayscale Canny finds no boundary. But paper and desk usually differ in hue
    or saturation; the LAB a/b and HSV-S channels expose that difference. The
    morphological gradient adds a texture/edge cue. OR-ing all cues yields a
    boundary even when any single channel is blind to it. Returns (combined,
    closed) where ``closed`` has gaps bridged for contour tracing.
    """
    def canny_of(ch: np.ndarray) -> np.ndarray:
        return auto_canny(cv2.GaussianBlur(ch, (5, 5), 0))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cues = [canny_of(cv2.medianBlur(gray, 5))]

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    _, a, b = cv2.split(lab)
    cues.append(canny_of(a))
    cues.append(canny_of(b))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    cues.append(canny_of(hsv[:, :, 1]))        # saturation

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    cues.append(canny_of(grad))

    combined = cues[0]
    for c in cues[1:]:
        combined = cv2.bitwise_or(combined, c)
    return combined, close_edges(combined)


def _gray_closed(bgr: np.ndarray) -> np.ndarray:
    """Grayscale-only closed edge map (the cheap, classic cue)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(cv2.GaussianBlur(gray, (5, 5), 0), 5)
    return close_edges(auto_canny(g))


# --------------------------------------------------------------------------- #
# Candidate generators
# --------------------------------------------------------------------------- #
def _quad_from_contour(cnt: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, bool]:
    """Fit a quad to a contour: approxPolyDP, else minAreaRect. Flag clipping."""
    h, w = shape[:2]
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) if peri else cnt
    if len(approx) == 4 and cv2.isContourConvex(approx):
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        quad = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
    clipped = bool(
        np.any(quad[:, 0] <= BORDER_PX) or np.any(quad[:, 0] >= w - 1 - BORDER_PX)
        or np.any(quad[:, 1] <= BORDER_PX) or np.any(quad[:, 1] >= h - 1 - BORDER_PX)
    )
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    return quad, clipped


def _contour_candidates(closed: np.ndarray, max_n: int = 6) -> list[np.ndarray]:
    """Convex 4-point approximations of the largest contours."""
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_n]
    frame = closed.shape[0] * closed.shape[1]
    quads = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(approx.reshape(4, 2).astype(np.float32))
        elif cv2.contourArea(cnt) > 0.25 * frame:
            # Large non-quad blob: a rotated bounding box is a decent guess.
            quads.append(cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32))
    return quads


def _hough_candidates(closed: np.ndarray) -> list[np.ndarray]:
    """Recover a quad from the outermost horizontal/vertical Hough lines."""
    lines = cv2.HoughLines(closed, 1, np.pi / 180, threshold=120)
    if lines is None or len(lines) < 4:
        return []
    horizontal, vertical = [], []
    for rho, theta in lines[:, 0, :]:
        (vertical if theta < np.pi / 4 or theta > 3 * np.pi / 4 else horizontal
         ).append((rho, theta))
    if len(horizontal) < 2 or len(vertical) < 2:
        return []
    horizontal.sort(key=lambda l: l[0])
    vertical.sort(key=lambda l: l[0])

    def inter(l1, l2):
        (r1, t1), (r2, t2) = l1, l2
        A = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        return np.linalg.solve(A, np.array([r1, r2])).astype(np.float32)

    pts = []
    for hh in (horizontal[0], horizontal[-1]):
        for vv in (vertical[0], vertical[-1]):
            p = inter(hh, vv)
            if p is not None:
                pts.append(p)
    return [np.array(pts, np.float32)] if len(pts) == 4 else []


def _fill_and_fit(mask: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, bool] | None:
    """Clean a binary mask, take the largest filled component, fit a quad."""
    h, w = shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.15 * h * w:
        return None
    return _quad_from_contour(cnt, (h, w))


def _skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Detect skin-coloured pixels (fingers) in YCrCb and HSV.

    Used to keep a hand holding the page out of the GrabCut foreground, so the
    detected document isn't stretched to include fingers.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    skin1 = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    skin2 = ((hh <= 25) | (hh >= 165)) & (ss >= 30) & (ss <= 190) & (vv >= 40)
    skin = ((skin1 | skin2) * 255).astype(np.uint8)
    return cv2.morphologyEx(skin, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def _paper_mask_candidates(bgr: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Detect a page by its colour: bright (high V) and low saturation (S).

    WHY: paper is close to white — high brightness and almost no colour — which
    stands out from a coloured or dark background (a pink cloth, a wooden desk)
    even when its *edges* are weak. We threshold bright + low-saturation pixels
    in HSV (V via Otsu so it adapts to exposure, S with a fixed low cap), clean
    the mask with morphology to bridge text gaps, take the largest filled
    component and fit a quad. This cheaply finds white pages the edge methods
    miss (e.g. pic7: a white notebook on a pink patterned cloth).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    _, bright = cv2.threshold(val, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_sat = (sat < 80).astype(np.uint8) * 255
    paper = cv2.bitwise_and(bright, low_sat)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    res = _fill_and_fit(paper, bgr.shape)
    return [res] if res else []


def _grabcut_candidates(bgr: np.ndarray, inset: float = 0.06) -> list[tuple[np.ndarray, bool]]:
    """Segment the page with GrabCut, then fit a quad to the foreground.

    WHY: when no edge-based quad survives (e.g. a white page on a white desk
    with faint borders), GrabCut separates foreground from background by colour
    statistics. Initialised with a rectangle inset from the image border, it
    labels the central object as foreground; we take the largest foreground
    component and fit a quad (approxPolyDP, else minAreaRect). If the component
    reaches the frame edge, the page is clipped and we flag it.
    """
    h, w = bgr.shape[:2]
    if min(h, w) < 20:
        return []
    rx, ry = int(w * inset), int(h * inset)
    rw, rh = int(w * (1 - 2 * inset)), int(h * (1 - 2 * inset))
    # Init mask: everything probable-background, the inset rectangle probable-
    # foreground, and any skin-coloured pixels (fingers) forced to definite
    # background so a hand holding the page is not merged into the document.
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    mask[ry:ry + rh, rx:rx + rw] = cv2.GC_PR_FGD
    mask[_skin_mask(bgr) > 0] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return []
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.15 * h * w:
        return []
    quad, _ = _quad_from_contour(cnt, (h, w))

    # Border-aware: if the foreground presses against the init-rectangle edge,
    # the page continues beyond it (clipped). Extend those corners to the image
    # border and flag the result, as required for pages that leave the frame.
    b = 3
    sub = fg > 0
    touch_left = bool(sub[ry:ry + rh, rx:rx + b].any())
    touch_right = bool(sub[ry:ry + rh, rx + rw - b:rx + rw].any())
    touch_top = bool(sub[ry:ry + b, rx:rx + rw].any())
    touch_bottom = bool(sub[ry + rh - b:ry + rh, rx:rx + rw].any())
    oq = order_points(quad)          # TL, TR, BR, BL
    if touch_left:
        oq[0, 0] = 0; oq[3, 0] = 0
    if touch_right:
        oq[1, 0] = w - 1; oq[2, 0] = w - 1
    if touch_top:
        oq[0, 1] = 0; oq[1, 1] = 0
    if touch_bottom:
        oq[3, 1] = h - 1; oq[2, 1] = h - 1
    clipped = touch_left or touch_right or touch_top or touch_bottom
    return [(oq.astype(np.float32), clipped)]


# --------------------------------------------------------------------------- #
# Candidate scoring
# --------------------------------------------------------------------------- #
def _edge_support(quad: np.ndarray, edge_map: np.ndarray) -> float:
    """Fraction of the quad's outline that lies on real edges (0..1)."""
    outline = np.zeros(edge_map.shape, np.uint8)
    cv2.polylines(outline, [quad.astype(np.int32)], True, 255, 2)
    total = int(np.count_nonzero(outline))
    if total == 0:
        return 0.0
    ed = cv2.dilate(edge_map, np.ones((3, 3), np.uint8))
    inter = int(np.count_nonzero((outline > 0) & (ed > 0)))
    return inter / total


def _contrast(quad: np.ndarray, bgr: np.ndarray) -> float:
    """Colour contrast between a band just inside vs just outside the quad."""
    h, w = bgr.shape[:2]
    fill = np.zeros((h, w), np.uint8)
    cv2.fillPoly(fill, [quad.astype(np.int32)], 255)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    inner = cv2.erode(fill, k, iterations=3)
    ring = cv2.subtract(cv2.dilate(fill, k, iterations=3), fill)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if np.count_nonzero(inner) < 10 or np.count_nonzero(ring) < 10:
        return 0.0
    mean_in = lab[inner > 0].mean(axis=0)
    mean_out = lab[ring > 0].mean(axis=0)
    return float(np.clip(np.linalg.norm(mean_in - mean_out) / 30.0, 0.0, 1.0))


def score_candidate(quad: np.ndarray, bgr: np.ndarray,
                    edge_map: np.ndarray, metrics: dict) -> float:
    """Blend area, rectangularity, edge support and contrast into a 0..1 score."""
    # Full area score once the page covers ~half the frame (typical for a photo).
    area_s = float(np.clip(metrics["area_ratio"] / 0.5, 0.0, 1.0))
    angle_s = float(np.mean([
        np.clip(1.0 - abs(a - 90.0) / 45.0, 0.0, 1.0) for a in metrics["angles"]
    ]))
    edge_s = _edge_support(quad, edge_map)
    contrast_s = _contrast(quad, bgr)
    return float(0.25 * area_s + 0.25 * angle_s + 0.30 * edge_s + 0.20 * contrast_s)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def detect_document(image: np.ndarray) -> DetectionResult:
    """Full multi-cue, score-based detection with graceful fallbacks.

    ``image`` is the (down-scaled) BGR photo. A single-channel image is accepted
    too (treated as already-gray) so edge maps can be passed in tests.
    """
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    shape = bgr.shape
    combined, combined_closed = multi_channel_edges(bgr)
    gray_closed = _gray_closed(bgr)

    # best holds (effective_score, raw_score, quad, method, clipped, metrics).
    # effective_score adds a small per-method bonus so a precise edge/colour quad
    # beats a GrabCut blob on a tie; raw_score is reported as the confidence.
    best = None

    def consider(quad, method, clipped):
        nonlocal best
        ok, m = validate_quad(quad, shape)
        if not ok:
            return
        s = score_candidate(quad, bgr, combined, m)
        eff = s + METHOD_BONUS.get(method, 0.0)
        if best is None or eff > best[0]:
            best = (eff, s, quad, method, clipped, m)

    # Cheap candidates: contour (grayscale + multi-cue), Hough, paper-colour mask.
    for q in _contour_candidates(gray_closed):
        consider(q, "contour", False)
    for q in _contour_candidates(combined_closed):
        consider(q, "multi-channel", False)
    for q in _hough_candidates(combined_closed):
        consider(q, "hough", False)
    for q, clipped in _paper_mask_candidates(bgr):
        consider(q, "paper", clipped)

    # Expensive GrabCut only when the cheap methods are weak or absent.
    if best is None or best[1] < GOOD_SCORE:
        for q, clipped in _grabcut_candidates(bgr):
            consider(q, "grabcut", clipped)

    if best is None:
        return DetectionResult(_image_corners(shape), "fallback", False, 0.0,
                               None, False)
    _, s, quad, method, clipped, m = best
    return DetectionResult(quad, method, True, round(s, 3), m, clipped)
