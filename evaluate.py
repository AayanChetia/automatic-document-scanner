"""
evaluate.py - measure the scanner on a labelled test set and plot results.

Metrics (reported overall and per condition):
  * Detection success rate - fraction of images where the detected quad matches
    the ground-truth quad with IoU >= SUCCESS_IOU (default 0.85). This is a
    stricter, more honest success measure than "did it return 4 points".
  * Corner accuracy (mean IoU) - average intersection-over-union between the
    detected quad and the hand-/auto-labelled ground-truth quad.
  * Average processing time per image (seconds).
  * Thresholding comparison - adaptive Gaussian vs global Otsu, scored by how
    well each reconstructs the known clean page (pixel agreement). Only runs for
    images that ship a clean 'page' reference (the synthetic set does).

Charts saved to results/:
    results/success_by_condition.png
    results/threshold_comparison.png
Plus results/metrics.json and results/metrics.md for the report.

Run:
    python evaluate.py
    python evaluate.py --success-iou 0.9
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")          # headless backend: save PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np

from scanner import scan_image
from scanner.enhance import black_and_white, otsu
from scanner.transform import four_point_transform, order_points
from scanner.utils import load_image, timer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TEST_DIR = DATA / "test_images"
GT_PATH = DATA / "ground_truth.json"
RESULTS = HERE / "results"

SUCCESS_IOU = 0.85


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def quad_iou(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> float:
    """Intersection-over-union of two quadrilaterals, via rasterised masks.

    Exact polygon IoU is fiddly for arbitrary quads, so we paint both quads
    onto blank masks the size of the image and compare pixels. IoU = |A ∩ B| /
    |A ∪ B|. This is robust to corner ordering and to slightly non-convex quads.
    """
    h, w = shape[:2]
    mask_a = np.zeros((h, w), np.uint8)
    mask_b = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask_a, [a.astype(np.int32)], 255)
    cv2.fillPoly(mask_b, [b.astype(np.int32)], 255)
    inter = np.count_nonzero((mask_a > 0) & (mask_b > 0))
    union = np.count_nonzero((mask_a > 0) | (mask_b > 0))
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# Thresholding quality (adaptive vs Otsu)
# --------------------------------------------------------------------------- #
def _binary_text_mask(gray: np.ndarray) -> np.ndarray:
    """True where there is text (dark ink) on the clean reference page."""
    return gray < 128


def threshold_accuracy(photo: np.ndarray, gt_corners: np.ndarray,
                       page_path: Path) -> tuple[float, float] | None:
    """Score adaptive vs Otsu against the known clean page.

    We warp the photo through the GROUND-TRUTH corners (so detection error does
    not contaminate this comparison), threshold with each method, resize to the
    clean page, and measure pixel agreement with the clean page's text mask.
    Returns (adaptive_accuracy, otsu_accuracy), or None if no reference page.
    """
    if not page_path.exists():
        return None
    page = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    gt_text = _binary_text_mask(page)

    flat = four_point_transform(photo, order_points(gt_corners))

    def score(binary: np.ndarray) -> float:
        resized = cv2.resize(binary, (page.shape[1], page.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        pred_text = resized < 128     # method output: text is black (0)
        return float(np.mean(pred_text == gt_text))

    return score(black_and_white(flat)), score(otsu(flat))


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_success_by_condition(per_cond: dict, path: Path) -> None:
    """Grouped bar chart: detection success rate and mean IoU per condition."""
    conds = list(per_cond.keys())
    success = [per_cond[c]["success_rate"] * 100 for c in conds]
    iou = [per_cond[c]["mean_iou"] * 100 for c in conds]

    x = np.arange(len(conds))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, success, width, label="Success rate (IoU≥%.2f)" % SUCCESS_IOU,
           color="#2E86DE")
    ax.bar(x + width / 2, iou, width, label="Mean corner IoU", color="#27AE60")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in conds])
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Detection performance by condition")
    for i, (s, j) in enumerate(zip(success, iou)):
        ax.text(x[i] - width / 2, s + 1, f"{s:.0f}", ha="center", fontsize=9)
        ax.text(x[i] + width / 2, j + 1, f"{j:.0f}", ha="center", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_threshold_comparison(adaptive: float, otsu_: float, path: Path) -> None:
    """Bar chart comparing adaptive vs Otsu reconstruction accuracy."""
    fig, ax = plt.subplots(figsize=(6, 4.6))
    methods = ["Adaptive\nGaussian", "Global\nOtsu"]
    vals = [adaptive * 100, otsu_ * 100]
    bars = ax.bar(methods, vals, color=["#2E86DE", "#E67E22"], width=0.5)
    ax.set_ylabel("Pixel agreement with clean page (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Binarisation quality: adaptive vs Otsu")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main evaluation loop
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    global SUCCESS_IOU
    parser = argparse.ArgumentParser(description="Evaluate the document scanner.")
    parser.add_argument("--success-iou", type=float, default=SUCCESS_IOU,
                        help="IoU threshold that counts as a detection success.")
    args = parser.parse_args(argv)
    SUCCESS_IOU = args.success_iou

    if not GT_PATH.exists():
        print(f"No ground truth at {GT_PATH}. Run: python data/make_synthetic.py")
        return 1
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    RESULTS.mkdir(parents=True, exist_ok=True)

    per_image = []
    times, ious = [], []
    adaptive_scores, otsu_scores = [], []
    by_cond_iou = defaultdict(list)
    by_cond_success = defaultdict(list)

    for name, entry in sorted(gt.items()):
        path = TEST_DIR / name
        if not path.exists():
            print(f"  (missing image {name}, skipping)")
            continue
        image = load_image(path)
        gt_corners = np.array(entry["corners"], dtype=np.float32)
        condition = entry.get("condition", "custom")

        with timer() as t:
            result = scan_image(image, mode="bw")
        elapsed = t[0]

        iou = quad_iou(result.corners_full, gt_corners, image.shape)
        success = iou >= SUCCESS_IOU

        times.append(elapsed)
        ious.append(iou)
        by_cond_iou[condition].append(iou)
        by_cond_success[condition].append(success)
        per_image.append({"name": name, "condition": condition,
                          "iou": round(iou, 4), "success": bool(success),
                          "time_s": round(elapsed, 4), "method": result.method})

        if "page" in entry:
            scores = threshold_accuracy(image, gt_corners, DATA / entry["page"])
            if scores:
                adaptive_scores.append(scores[0])
                otsu_scores.append(scores[1])

    if not per_image:
        print("No evaluable images found.")
        return 1

    # Aggregate.
    per_cond = {}
    for cond in sorted(by_cond_iou):
        per_cond[cond] = {
            "n": len(by_cond_iou[cond]),
            "mean_iou": float(np.mean(by_cond_iou[cond])),
            "success_rate": float(np.mean(by_cond_success[cond])),
        }

    overall = {
        "n_images": len(per_image),
        "success_rate": float(np.mean([p["success"] for p in per_image])),
        "mean_iou": float(np.mean(ious)),
        "avg_time_s": float(np.mean(times)),
        "success_iou_threshold": SUCCESS_IOU,
        "adaptive_accuracy": float(np.mean(adaptive_scores)) if adaptive_scores else None,
        "otsu_accuracy": float(np.mean(otsu_scores)) if otsu_scores else None,
    }

    # Charts.
    plot_success_by_condition(per_cond, RESULTS / "success_by_condition.png")
    if adaptive_scores:
        plot_threshold_comparison(overall["adaptive_accuracy"],
                                  overall["otsu_accuracy"],
                                  RESULTS / "threshold_comparison.png")

    # Persist metrics (json for machines, md table for the report).
    (RESULTS / "metrics.json").write_text(
        json.dumps({"overall": overall, "by_condition": per_cond,
                    "per_image": per_image}, indent=2), encoding="utf-8")
    _write_markdown(overall, per_cond, RESULTS / "metrics.md")

    # Console summary.
    print("\n==== Evaluation summary ====")
    print(f"Images evaluated      : {overall['n_images']}")
    print(f"Detection success rate: {overall['success_rate']*100:.1f}% "
          f"(IoU >= {SUCCESS_IOU})")
    print(f"Mean corner IoU       : {overall['mean_iou']*100:.1f}%")
    print(f"Avg time per image    : {overall['avg_time_s']*1000:.1f} ms")
    if adaptive_scores:
        print(f"Binarisation accuracy : adaptive {overall['adaptive_accuracy']*100:.1f}%"
              f"  vs  Otsu {overall['otsu_accuracy']*100:.1f}%")
    print("\nPer condition:")
    for cond, m in per_cond.items():
        print(f"  {cond:10s} n={m['n']:2d}  success={m['success_rate']*100:5.1f}%  "
              f"mean IoU={m['mean_iou']*100:5.1f}%")
    print(f"\nCharts + metrics written to {RESULTS}")
    return 0


def _write_markdown(overall: dict, per_cond: dict, path: Path) -> None:
    """Write a Markdown metrics table for pasting into the report."""
    lines = ["# Evaluation results", "",
             f"- Images evaluated: **{overall['n_images']}**",
             f"- Detection success rate (IoU ≥ {overall['success_iou_threshold']}): "
             f"**{overall['success_rate']*100:.1f}%**",
             f"- Mean corner IoU: **{overall['mean_iou']*100:.1f}%**",
             f"- Average processing time: **{overall['avg_time_s']*1000:.1f} ms/image**"]
    if overall["adaptive_accuracy"] is not None:
        lines.append(f"- Binarisation accuracy: adaptive "
                     f"**{overall['adaptive_accuracy']*100:.1f}%** vs Otsu "
                     f"**{overall['otsu_accuracy']*100:.1f}%**")
    lines += ["", "## By condition", "",
              "| Condition | N | Success rate | Mean IoU |",
              "|-----------|---|--------------|----------|"]
    for cond, m in per_cond.items():
        lines.append(f"| {cond.capitalize()} | {m['n']} | "
                     f"{m['success_rate']*100:.1f}% | {m['mean_iou']*100:.1f}% |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
