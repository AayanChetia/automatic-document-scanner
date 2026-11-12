"""
webcam.py - live document scanning from a webcam.

Shows the camera feed with the detected document outline drawn in real time.
Press keys to act:

    SPACE / C : capture the current frame, run the full scan, save it
    M         : cycle the enhancement mode (bw -> magic -> gray -> original)
    Q / ESC   : quit

Usage:
    python webcam.py                      # default camera, saves to results/
    python webcam.py --camera 1           # pick a camera index
    python webcam.py --output my_scans    # choose output folder

The live outline uses the same detection code as the rest of the project; only
the final, saved image runs the full-resolution warp + enhancement.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from scanner import scan_image
from scanner.detect import detect_document
from scanner.preprocess import preprocess
from scanner.transform import order_points
from scanner.enhance import MODES

LIVE_MODES = ["bw", "magic", "gray", "original"]


def _detect_live(frame: np.ndarray, process_height: int = 500):
    """Detect the document on a frame; return (corners_full, found)."""
    ratio = frame.shape[0] / float(process_height)
    small = cv2.resize(frame, (int(frame.shape[1] / ratio), process_height))
    pre = preprocess(small)
    det = detect_document(pre.closed)
    corners = order_points(det.corners * ratio)
    return corners, det.found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live webcam document scanner.")
    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="Camera index (default 0).")
    parser.add_argument("--output", "-o", default="results/webcam",
                        help="Folder to save captured scans into.")
    args = parser.parse_args(argv)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: could not open camera {args.camera}.")
        return 1

    mode_idx = 0
    saved = 0
    print("Live scanner running. SPACE/C = capture, M = mode, Q/ESC = quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: failed to read frame.")
            break

        display = frame.copy()
        try:
            corners, found = _detect_live(frame)
            colour = (0, 255, 0) if found else (0, 165, 255)
            pts = corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(display, [pts], True, colour, 3)
            status = "DOCUMENT DETECTED" if found else "searching..."
        except Exception:
            status = "searching..."

        banner = f"[{status}]  mode={LIVE_MODES[mode_idx]}  " \
                 f"SPACE=capture  M=mode  Q=quit  saved={saved}"
        cv2.rectangle(display, (0, 0), (display.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(display, banner, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Automatic Document Scanner - live", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        if key == ord("m"):
            mode_idx = (mode_idx + 1) % len(LIVE_MODES)
        if key in (ord(" "), ord("c")):
            result = scan_image(frame, mode=LIVE_MODES[mode_idx])
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"scan_{stamp}.png"
            cv2.imwrite(str(path), result.enhanced)
            saved += 1
            print(f"Saved {path} (detected={result.found})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {saved} scan(s) saved to {out_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
