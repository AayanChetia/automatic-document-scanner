"""
demo_check.py - run detection on your own photos and save the results to look at.

Put real photos in ``data/demo_images/`` and run:

    python demo_check.py

For each image it saves, into ``results/demo_check/``:
  * ``<name>_overlay.jpg`` - the photo with the detected page outlined and a
    caption showing the method, confidence, and clipped/fallback flags
  * ``<name>_scan.jpg``    - the final enhanced (magic-colour) scan

Both ``data/demo_images/`` and ``results/demo_check/`` are gitignored, because
your photos may contain personal information - nothing here is committed.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from scanner import scan_image
from scanner.utils import load_image

HERE = Path(__file__).resolve().parent
IN_DIR = HERE / "data" / "demo_images"
OUT_DIR = HERE / "results" / "demo_check"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> int:
    if not IN_DIR.exists():
        print(f"No folder {IN_DIR}. Create it and drop some photos in.")
        return 1
    images = sorted(p for p in IN_DIR.iterdir() if p.suffix.lower() in EXTS)
    if not images:
        print(f"No images in {IN_DIR}.")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running detection on {len(images)} image(s)...\n")
    print(f"{'image':32s} {'method':13s} {'found':5s} {'conf':5s} clipped")
    print("-" * 70)
    for path in images:
        image = load_image(path)
        r = scan_image(image, mode="magic")

        overlay = r.contour_overlay.copy()
        status = (f"{r.method}  conf={r.confidence:.2f}"
                  + ("  CLIPPED" if r.clipped else "")
                  + ("" if r.found else "  FALLBACK"))
        colour = (0, 255, 0) if (r.found and r.confidence >= 0.5) else (0, 165, 255)
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(overlay, status, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    colour, 2, cv2.LINE_AA)

        cv2.imwrite(str(OUT_DIR / f"{path.stem}_overlay.jpg"), overlay)
        cv2.imwrite(str(OUT_DIR / f"{path.stem}_scan.jpg"), r.enhanced)

        print(f"{path.name:32s} {r.method:13s} {str(r.found):5s} "
              f"{r.confidence:.2f}  {r.clipped}")

    print(f"\nOverlays + scans written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
