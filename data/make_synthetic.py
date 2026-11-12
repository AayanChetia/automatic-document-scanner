"""
make_synthetic.py - build a synthetic test set with known ground truth.

Why synthetic data: to evaluate the scanner we need images where we know the
TRUE page corners. Hand-labelling is slow, and before you add your own photos
there is nothing to measure. So we *generate* documents by taking a clean,
upright page and projecting it through a known perspective onto a background.
Because we choose the destination quad ourselves, those four points ARE the
ground-truth corners - no labelling required.

Four conditions mirror real-world difficulty (these become the chart buckets):
    plain     - clean page, plain background, mild perspective
    cluttered - busy/noisy background that competes with the page edges
    lowlight  - darkened, low-contrast capture
    tilted    - strong perspective / steep viewing angle

Outputs:
    data/test_images/<condition>_<i>.png   the synthetic "photos"
    data/pages/<condition>_<i>.png         the clean upright page (for the
                                           thresholding-quality comparison)
    data/ground_truth.json                 corners + condition + page path

Run:
    python data/make_synthetic.py            # default 5 images per condition
    python data/make_synthetic.py --per 8
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TEST_DIR = HERE / "test_images"
PAGE_DIR = HERE / "pages"
GT_PATH = HERE / "ground_truth.json"

CONDITIONS = ["plain", "cluttered", "lowlight", "tilted"]

PAGE_W, PAGE_H = 600, 800          # clean page size
CANVAS_W, CANVAS_H = 760, 980      # output "photo" size (page + background margin)
MARGIN = 70                        # page inset from the canvas edge

LOREM = [
    "AUTOMATIC DOCUMENT SCANNER",
    "Computer Vision Project Report",
    "",
    "This page is a synthetic test document used to",
    "evaluate the scanning pipeline. The detector must",
    "locate the four corners of this page, correct the",
    "perspective, and produce a clean, flat scan.",
    "",
    "1. Preprocess: grayscale, blur, Canny, close.",
    "2. Detect: contours, approxPolyDP, largest quad.",
    "3. Transform: order corners, warp perspective.",
    "4. Enhance: adaptive threshold for crisp text.",
    "",
    "Mean intensity, local contrast and lighting all",
    "vary between the test conditions so the results",
    "reflect real handheld document photography.",
]


def make_page() -> np.ndarray:
    """Render a clean, upright white page with black text."""
    page = np.full((PAGE_H, PAGE_W, 3), 252, np.uint8)
    cv2.rectangle(page, (4, 4), (PAGE_W - 5, PAGE_H - 5), (210, 210, 210), 2)
    y = 70
    for i, line in enumerate(LOREM):
        scale = 0.9 if i == 0 else (0.7 if i == 1 else 0.6)
        thick = 2 if i < 2 else 1
        if line:
            cv2.putText(page, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (20, 20, 20), thick, cv2.LINE_AA)
        y += 42
    return page


def _background(condition: str, rng: random.Random) -> np.ndarray:
    """Build a background canvas appropriate to the condition."""
    if condition == "cluttered":
        # Busy background with moderate contrast against the page, so the page
        # edge is visible but competes with many distracting lines/shapes.
        bg = np.random.randint(80, 165, (CANVAS_H, CANVAS_W, 3), np.uint8)
        for _ in range(55):
            p1 = (rng.randint(0, CANVAS_W), rng.randint(0, CANVAS_H))
            p2 = (rng.randint(0, CANVAS_W), rng.randint(0, CANVAS_H))
            col = tuple(rng.randint(0, 255) for _ in range(3))
            if rng.random() < 0.5:
                cv2.line(bg, p1, p2, col, rng.randint(1, 5))
            else:
                cv2.rectangle(bg, p1, p2, col, rng.randint(1, 3))
        bg = cv2.GaussianBlur(bg, (3, 3), 0)
    else:
        # Plain-ish desk surface with a soft gradient and a little texture.
        base = rng.randint(50, 90)
        bg = np.full((CANVAS_H, CANVAS_W, 3), base, np.uint8)
        grad = np.tile(np.linspace(-20, 20, CANVAS_W, dtype=np.float32),
                       (CANVAS_H, 1))
        bg = np.clip(bg.astype(np.float32) + grad[..., None], 0, 255).astype(np.uint8)
        noise = np.random.randint(-8, 8, bg.shape, np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bg


def _dest_quad(condition: str, rng: random.Random) -> np.ndarray:
    """Pick the destination corners (ground truth) for this condition."""
    m = MARGIN  # margin from canvas edge
    base = np.array([[m, m], [CANVAS_W - m, m],
                     [CANVAS_W - m, CANVAS_H - m], [m, CANVAS_H - m]],
                    dtype=np.float32)
    # Jitter magnitude grows with difficulty.
    jit = {"plain": 18, "cluttered": 22, "lowlight": 20, "tilted": 55}[condition]
    quad = base.copy()
    for i in range(4):
        quad[i, 0] += rng.uniform(-jit, jit)
        quad[i, 1] += rng.uniform(-jit, jit)
    if condition == "tilted":
        # Add a systematic keystone (top narrower than bottom) for a steep angle.
        quad[0, 0] += 40
        quad[1, 0] -= 40
    return quad


def _apply_lighting(image: np.ndarray, condition: str, rng: random.Random) -> np.ndarray:
    """Apply condition-specific lighting and capture artefacts."""
    out = image.astype(np.float32)
    if condition == "lowlight":
        out *= rng.uniform(0.20, 0.34)              # strong global darkening
        out += np.random.normal(0, 13, out.shape)   # sensor noise in the dark
    else:
        out += np.random.normal(0, 4, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    # A mild blur everywhere mimics a hand-held camera / imperfect focus.
    k = 3 if condition != "tilted" else 5
    out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def synth_one(condition: str, rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one synthetic photo; return (photo, clean_page, gt_corners)."""
    page = make_page()
    bg = _background(condition, rng)
    dst = _dest_quad(condition, rng)
    src = np.array([[0, 0], [PAGE_W, 0], [PAGE_W, PAGE_H], [0, PAGE_H]],
                   dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    warped_page = cv2.warpPerspective(page, H, (CANVAS_W, CANVAS_H))
    mask = cv2.warpPerspective(np.full((PAGE_H, PAGE_W), 255, np.uint8), H,
                               (CANVAS_W, CANVAS_H))

    # Composite the warped page onto the background using the mask.
    photo = bg.copy()
    photo[mask > 0] = warped_page[mask > 0]

    if condition == "cluttered":
        m = MARGIN
        # A few objects that OVERLAP the page border (pen, ruler, torn edge),
        # which can break the page outline into fragments - this is what makes
        # the contour step sometimes fail and the Hough fallback earn its keep.
        for _ in range(rng.randint(1, 2)):
            side = rng.choice(["top", "bottom", "left", "right"])
            col = tuple(rng.randint(0, 90) for _ in range(3))
            th = rng.randint(5, 11)
            if side in ("top", "bottom"):
                y = m if side == "top" else CANVAS_H - m
                x0 = rng.randint(m, CANVAS_W - 2 * m)
                cv2.line(photo, (x0, y - 40), (x0 + rng.randint(-30, 30), y + 40),
                         col, th)
            else:
                x = m if side == "left" else CANVAS_W - m
                y0 = rng.randint(m, CANVAS_H - 2 * m)
                cv2.line(photo, (x - 40, y0), (x + 40, y0 + rng.randint(-30, 30)),
                         col, th)

    photo = _apply_lighting(photo, condition, rng)
    return photo, page, dst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic test set.")
    parser.add_argument("--per", type=int, default=5,
                        help="Images per condition (default 5).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)

    ground_truth: dict[str, dict] = {}
    count = 0
    for condition in CONDITIONS:
        for i in range(args.per):
            name = f"{condition}_{i}.png"
            photo, page, dst = synth_one(condition, rng)
            cv2.imwrite(str(TEST_DIR / name), photo)
            cv2.imwrite(str(PAGE_DIR / name), page)
            ground_truth[name] = {
                "condition": condition,
                # Ordered TL, TR, BR, BL - matches the pipeline's corner order.
                "corners": dst.tolist(),
                "page": f"pages/{name}",
            }
            count += 1

    GT_PATH.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    print(f"Generated {count} images across {len(CONDITIONS)} conditions.")
    print(f"  photos      -> {TEST_DIR}")
    print(f"  clean pages -> {PAGE_DIR}")
    print(f"  ground truth-> {GT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
