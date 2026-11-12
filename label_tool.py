"""
label_tool.py - click four corners to build ground-truth labels.

Use this to label your OWN photos (drop them in data/test_images/) so the
evaluator can measure corner accuracy against a human-labelled quad. The
synthetic set already has ground truth from make_synthetic.py; this tool is for
real images you add later.

Controls (per image):
    Left click x4 : mark Top-left, Top-right, Bottom-right, Bottom-left
                    (click them IN THAT ORDER)
    u             : undo the last point
    r             : reset all points for this image
    s             : skip this image (don't label)
    Enter / n     : save the 4 points and go to the next image
    q / ESC       : quit (keeps everything saved so far)

Usage:
    python label_tool.py                         # label data/test_images
    python label_tool.py --dir path/to/images
    python label_tool.py --relabel               # also re-label existing ones
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE / "data" / "test_images"
GT_PATH = HERE / "data" / "ground_truth.json"

LABELS = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_gt() -> dict:
    if GT_PATH.exists():
        return json.loads(GT_PATH.read_text(encoding="utf-8"))
    return {}


def _save_gt(gt: dict) -> None:
    GT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GT_PATH.write_text(json.dumps(gt, indent=2), encoding="utf-8")


def label_image(path: Path) -> np.ndarray | None:
    """Open a window and collect four clicked corners. Returns (4,2) or None."""
    image = cv2.imread(str(path))
    if image is None:
        print(f"  (could not open {path.name}, skipping)")
        return None

    # Scale big images down to fit the screen while labelling; scale points back.
    disp_h = 800
    scale = disp_h / image.shape[0] if image.shape[0] > disp_h else 1.0
    disp = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    points: list[list[int]] = []
    window = f"Label: {path.name}  (click TL, TR, BR, BL)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = disp.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(canvas, LABELS[i], (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if len(points) >= 2:
            cv2.polylines(canvas, [np.array(points, np.int32)], len(points) == 4,
                          (0, 255, 0), 2)
        hint = f"{len(points)}/4 points | u=undo r=reset s=skip Enter=save q=quit"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(canvas, hint, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            raise KeyboardInterrupt
        if key == ord("u") and points:
            points.pop()
        if key == ord("r"):
            points.clear()
        if key == ord("s"):
            cv2.destroyWindow(window)
            return None
        if key in (13, ord("n")) and len(points) == 4:
            cv2.destroyWindow(window)
            return (np.array(points, dtype=np.float32) / scale)

    # unreachable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Click 4 corners to label images.")
    parser.add_argument("--dir", default=str(DEFAULT_DIR),
                        help="Folder of images to label.")
    parser.add_argument("--relabel", action="store_true",
                        help="Also re-label images already present in ground truth.")
    args = parser.parse_args(argv)

    folder = Path(args.dir)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"No images found in {folder}")
        return 1

    gt = _load_gt()
    print("Labelling. Click TL, TR, BR, BL in order. See window title for help.")
    try:
        for path in files:
            if path.name in gt and not args.relabel:
                continue
            corners = label_image(path)
            if corners is None:
                continue
            entry = gt.get(path.name, {})
            entry["corners"] = corners.tolist()
            entry.setdefault("condition", "custom")
            gt[path.name] = entry
            _save_gt(gt)
            print(f"  labelled {path.name}")
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cv2.destroyAllWindows()
        _save_gt(gt)
    print(f"Ground truth saved to {GT_PATH} ({len(gt)} entries).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
