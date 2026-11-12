"""Automatic document scanner.

Usage:
    python scanner.py photo.jpg                 # scan an image file -> photo_scan.pdf
    python scanner.py photo.jpg -o out.png      # save as image
    python scanner.py --camera                  # use webcam (SPACE = scan, Q = quit)
    python scanner.py a.jpg b.jpg -o all.pdf    # several pages -> one PDF
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def find_document_contour(image):
    """Return the 4 corner points of the largest document-like quadrilateral, or None."""
    ratio = 800.0 / max(image.shape[:2])
    small = cv2.resize(image, None, fx=ratio, fy=ratio) if ratio < 1 else image.copy()
    ratio = small.shape[0] / image.shape[0]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    min_area = 0.1 * small.shape[0] * small.shape[1]

    for c in contours:
        if cv2.contourArea(c) < min_area:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            return approx.reshape(4, 2) / ratio
    return None


def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype="float32")
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (width, height))


def enhance(image):
    """Make the page look like a clean scan (white paper, dark text)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 21, 10)


def scan_image(image, mode="bw"):
    """Detect, flatten and enhance a document in a BGR image."""
    corners = find_document_contour(image)
    if corners is None:
        print("  ! No document outline found - using the whole image.")
        warped = image
    else:
        warped = four_point_transform(image, corners)
    if mode == "bw":
        return enhance(warped)
    if mode == "gray":
        return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return warped  # "color"


def save_pages(pages, output):
    output = Path(output)
    pil_pages = [Image.fromarray(p if p.ndim == 2 else cv2.cvtColor(p, cv2.COLOR_BGR2RGB))
                 .convert("RGB") for p in pages]
    if output.suffix.lower() == ".pdf":
        pil_pages[0].save(output, save_all=True, append_images=pil_pages[1:])
    elif len(pil_pages) == 1:
        pil_pages[0].save(output)
    else:
        for i, p in enumerate(pil_pages, 1):
            p.save(output.with_name(f"{output.stem}_{i}{output.suffix}"))
    print(f"Saved {len(pil_pages)} page(s) -> {output}")


def run_camera(mode, output):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open the webcam.")
    pages = []
    print("SPACE = capture page, ENTER = finish and save, Q = quit without saving")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        preview = frame.copy()
        corners = find_document_contour(frame)
        if corners is not None:
            cv2.polylines(preview, [corners.astype(int)], True, (0, 255, 0), 3)
        cv2.putText(preview, f"Pages: {len(pages)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Document scanner", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            pages.append(scan_image(frame, mode))
            print(f"Captured page {len(pages)}")
        elif key == 13 and pages:
            break
        elif key in (ord("q"), 27):
            pages = []
            break
    cap.release()
    cv2.destroyAllWindows()
    if pages:
        save_pages(pages, output)


def main():
    ap = argparse.ArgumentParser(description="Automatic document scanner")
    ap.add_argument("images", nargs="*", help="input image files")
    ap.add_argument("-o", "--output", help="output file (.pdf, .png, .jpg)")
    ap.add_argument("--camera", action="store_true", help="scan from the webcam")
    ap.add_argument("--mode", choices=["bw", "gray", "color"], default="bw",
                    help="bw = black & white (default), gray, or color")
    args = ap.parse_args()

    if args.camera:
        run_camera(args.mode, args.output or "scan.pdf")
        return
    if not args.images:
        ap.error("give at least one image file, or use --camera")

    pages = []
    for path in args.images:
        print(f"Scanning {path}")
        img = cv2.imread(path)
        if img is None:
            sys.exit(f"Could not read image: {path}")
        pages.append(scan_image(img, args.mode))
    output = args.output or f"{Path(args.images[0]).stem}_scan.pdf"
    save_pages(pages, output)


if __name__ == "__main__":
    main()
