"""
scan.py - command-line interface for the Automatic Document Scanner.

Examples
--------
Single image, default black & white, saved next to the pipeline default:
    python scan.py --input photo.jpg --output scan.png

Choose an enhancement mode:
    python scan.py --input photo.jpg --output scan.png --mode magic

Batch-process a folder of images into one output folder:
    python scan.py --input ./data/test_images --output ./results/scans

Batch into a single multi-page PDF:
    python scan.py --input ./photos --output ./results/doc.pdf

Also run OCR and print/save the recognised text:
    python scan.py --input photo.jpg --output scan.png --ocr

Modes: original | gray | magic | bw (default) | otsu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scanner import scan_image
from scanner.enhance import MODES
from scanner.export import images_to_pdf, save_scan
from scanner.ocr import extract_text, tesseract_available
from scanner.utils import load_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _gather_inputs(input_path: Path) -> list[Path]:
    """Return the list of image files to process from a file or folder path."""
    if input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not files:
            raise FileNotFoundError(f"No images found in folder: {input_path}")
        return files
    if input_path.is_file():
        return [input_path]
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _process_one(path: Path, mode: str, run_ocr: bool) -> tuple[object, str | None]:
    """Scan a single image; return (enhanced_image, ocr_text_or_None)."""
    image = load_image(path)
    result = scan_image(image, mode=mode)
    status = "detected" if result.found else f"NO quad (used {result.method})"
    print(f"  [{path.name}] {status}; output {result.enhanced.shape[1]}x"
          f"{result.enhanced.shape[0]}")
    text = extract_text(result.enhanced) if run_ocr else None
    return result.enhanced, text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Automatic Document Scanner (classical OpenCV pipeline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to an image, or a folder of images.")
    parser.add_argument("--output", "-o", required=True,
                        help="Output image path, a .pdf path, or an output folder.")
    parser.add_argument("--mode", "-m", default="bw", choices=MODES,
                        help="Enhancement mode (default: bw).")
    parser.add_argument("--ocr", action="store_true",
                        help="Also run OCR (needs Tesseract installed).")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        inputs = _gather_inputs(input_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Scanning {len(inputs)} image(s) in mode '{args.mode}'...")
    if args.ocr and not tesseract_available():
        print("  Note: --ocr requested but Tesseract is not available; "
              "text extraction will report that instead of failing.")

    enhanced_images: list = []
    ocr_texts: list[str] = []
    for path in inputs:
        enhanced, text = _process_one(path, args.mode, args.ocr)
        enhanced_images.append(enhanced)
        if text is not None:
            ocr_texts.append(f"===== {path.name} =====\n{text}")

    # Decide how to write output based on the output path / number of images.
    if output_path.suffix.lower() == ".pdf":
        images_to_pdf(enhanced_images, output_path)
        print(f"Saved PDF: {output_path}")
    elif len(enhanced_images) == 1 and output_path.suffix:
        save_scan(output_path, enhanced_images[0])
        print(f"Saved: {output_path}")
    else:
        # Multiple images (or a folder output): write one file per input.
        output_path.mkdir(parents=True, exist_ok=True)
        for src, img in zip(inputs, enhanced_images):
            dst = output_path / f"{src.stem}_scan.png"
            save_scan(dst, img)
        print(f"Saved {len(enhanced_images)} scans to: {output_path}")

    if ocr_texts:
        txt_path = (output_path.with_suffix(".txt")
                    if output_path.suffix else output_path / "ocr.txt")
        Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
        Path(txt_path).write_text("\n\n".join(ocr_texts), encoding="utf-8")
        print(f"Saved OCR text: {txt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
