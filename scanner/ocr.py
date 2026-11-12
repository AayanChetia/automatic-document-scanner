"""
ocr.py - optional text extraction with Tesseract (pytesseract).

This stage is deliberately optional. Tesseract is a separate native program; a
marker's laptop may not have it installed. So every public function here checks
availability first and returns a friendly message instead of crashing. OCR is a
"nice to have" on top of the core CV pipeline, never a hard dependency.
"""

from __future__ import annotations

import shutil

import numpy as np

try:
    import pytesseract  # pure-python wrapper; still needs the tesseract binary
    _HAS_PYTESSERACT = True
except Exception:  # pragma: no cover - import guard
    _HAS_PYTESSERACT = False


def tesseract_available() -> bool:
    """True only if both the python wrapper and the tesseract binary exist.

    We check ``shutil.which('tesseract')`` because pytesseract can be installed
    while the actual OCR engine is not - calling it then would raise. Detecting
    this up front lets the UI show "OCR unavailable" instead of erroring.
    """
    if not _HAS_PYTESSERACT:
        return False
    if shutil.which("tesseract") is not None:
        return True
    # pytesseract may have an explicit path configured even if not on PATH.
    try:
        cmd = pytesseract.pytesseract.tesseract_cmd
        return shutil.which(cmd) is not None or cmd not in ("", "tesseract")
    except Exception:
        return False


def extract_text(image: np.ndarray, lang: str = "eng") -> str:
    """Run OCR on an image and return the recognised text.

    Returns a clear explanatory string (never raises) when Tesseract is
    unavailable, so callers can display the result directly.
    """
    if not tesseract_available():
        return (
            "[OCR unavailable] Tesseract is not installed. Install it from "
            "https://github.com/UB-Mannheim/tesseract/wiki (Windows) and ensure "
            "'tesseract' is on your PATH to enable text extraction."
        )
    try:
        return pytesseract.image_to_string(image, lang=lang).strip()
    except Exception as exc:  # pragma: no cover - runtime guard
        return f"[OCR error] {exc}"
