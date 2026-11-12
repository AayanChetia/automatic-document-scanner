"""
app.py - Streamlit web UI for the Automatic Document Scanner.

Run with:
    streamlit run app.py

Features (all backed by the classical CV pipeline in scanner/):
  * Upload one or more photos.
  * See every intermediate stage side by side: grayscale, blur, Canny edges,
    morphological close, detected-contour overlay, warped page, final result.
  * If auto-detection is off, switch on "Adjust corners" and nudge the four
    corners with sliders - the warp updates live.
  * Pick an enhancement mode (Original / Gray / Magic colour / B&W / Otsu).
  * Download the current scan as PNG, or all pages combined into one PDF.
  * Optional OCR (if Tesseract is installed).

The UI is intentionally clean so it demos well on a projector.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from scanner import scan_image
from scanner.enhance import MODES
from scanner.export import images_to_pdf
from scanner.ocr import extract_text, tesseract_available
from scanner.utils import to_display

st.set_page_config(page_title="Automatic Document Scanner", page_icon="📄",
                   layout="wide")

MODE_LABELS = {
    "original": "Original colour",
    "gray": "Grayscale",
    "magic": "Magic colour (CLAHE)",
    "bw": "Black & White (adaptive)",
    "otsu": "Black & White (Otsu)",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def decode_upload(file) -> np.ndarray:
    """Decode an uploaded file (bytes) into an OpenCV BGR image."""
    data = np.frombuffer(file.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {file.name}")
    return img


def to_png_bytes(image: np.ndarray) -> bytes:
    """Encode an OpenCV image to PNG bytes for download."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise IOError("PNG encoding failed")
    return buf.tobytes()


def to_pdf_bytes(images: list[np.ndarray]) -> bytes:
    """Encode a list of images into a single multi-page PDF (in memory)."""
    pages = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 \
            else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pages.append(Image.fromarray(rgb))
    buf = io.BytesIO()
    pages[0].save(buf, "PDF", save_all=True, append_images=pages[1:], resolution=200.0)
    return buf.getvalue()


def apply_rotation(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate an image by a multiple of 90 degrees (for sideways photos)."""
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


# Below this confidence a detection is treated as low-confidence (yellow),
# never shown as a green success.
CONFIDENCE_OK = 0.5


# --------------------------------------------------------------------------- #
# Sidebar - global controls
# --------------------------------------------------------------------------- #
st.sidebar.title("📄 Scanner controls")
st.sidebar.markdown("Classical computer-vision document scanner (OpenCV).")

mode = st.sidebar.selectbox(
    "Enhancement mode",
    options=list(MODES),
    index=list(MODES).index("bw"),
    format_func=lambda m: MODE_LABELS[m],
)
process_height = st.sidebar.slider(
    "Detection resolution (px height)", 300, 900, 500, step=50,
    help="Detection runs on a down-scaled copy; the final warp always uses the "
         "full-resolution original.",
)
show_stages = st.sidebar.checkbox("Show intermediate stages", value=True)
run_ocr = st.sidebar.checkbox("Run OCR on result", value=False)

if run_ocr and not tesseract_available():
    st.sidebar.warning("Tesseract not detected - OCR will explain how to install it.")

st.sidebar.markdown("---")
st.sidebar.caption("Pipeline: preprocess → detect → warp → enhance")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("Automatic Document Scanner")
st.markdown(
    "Turn a photo of a document into a flat, cropped, enhanced scan — "
    "every step is classical computer vision you can explain."
)

uploads = st.file_uploader(
    "Upload one or more document photos",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
    accept_multiple_files=True,
)

if not uploads:
    st.info("⬆️ Upload at least one image to begin. "
            "Tip: shoot the page on a dark, plain surface for best edge detection.")
    st.stop()

# Let the user pick which uploaded image to inspect/adjust in detail.
names = [f.name for f in uploads]
selected_name = st.selectbox("Image to inspect", names, index=0)
selected_file = uploads[names.index(selected_name)]

try:
    image = decode_upload(selected_file)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# --- Rotation control (for sideways photos, e.g. a newspaper shot landscape) --
rot_key = f"rot::{selected_name}"
if rot_key not in st.session_state:
    st.session_state[rot_key] = 0
rcol1, rcol2 = st.columns([1, 4])
with rcol1:
    if st.button("🔄 Rotate 90°"):
        st.session_state[rot_key] = (st.session_state[rot_key] + 90) % 360
rotation = st.session_state[rot_key]
with rcol2:
    st.caption(f"Current rotation: **{rotation}°** — use this if the photo is "
               "sideways or upside-down.")
image = apply_rotation(image, rotation)

h, w = image.shape[:2]

# First pass: automatic detection, to seed the corner sliders.
auto = scan_image(image, mode=mode, process_height=process_height)

# ----- Optional manual corner adjustment --------------------------------- #
adjust = st.checkbox(
    "🛠️ Adjust corners manually",
    value=not (auto.found and auto.confidence >= CONFIDENCE_OK),
    help="Turn on if the detected outline is wrong. Sliders start from the "
         "auto-detected corners.",
)

corners_override = None
if adjust:
    st.caption("Drag the sliders to move each corner (pixels on the full image).")
    labels = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
    pts = auto.corners_full.astype(int)
    new_pts = []
    cols = st.columns(4)
    for i, (label, col) in enumerate(zip(labels, cols)):
        with col:
            st.markdown(f"**{label}**")
            x = st.slider(f"{label} x", 0, w - 1, int(pts[i][0]), key=f"x{i}")
            y = st.slider(f"{label} y", 0, h - 1, int(pts[i][1]), key=f"y{i}")
            new_pts.append([x, y])
    corners_override = np.array(new_pts, dtype=np.float32)

# Final result (with override if the user adjusted corners).
result = scan_image(image, mode=mode, process_height=process_height,
                    corners_override=corners_override)

# ----- Detection status + confidence ------------------------------------- #
conf_pct = f"{result.confidence * 100:.0f}%"
scol1, scol2 = st.columns([3, 1])
with scol2:
    st.metric("Detection confidence", conf_pct)
with scol1:
    if result.method == "manual":
        st.success("Using your manually adjusted corners.")
    elif result.found and result.confidence >= CONFIDENCE_OK:
        st.success(f"Document detected automatically "
                   f"(method: {result.method}, confidence {conf_pct}).")
    else:
        # Fallback OR low-confidence detection: never a green success.
        st.warning("⚠️ Low confidence – please adjust corners manually. "
                   f"(method: {result.method}, confidence {conf_pct}) "
                   "Turn on 'Adjust corners manually' below, and try the "
                   "'Rotate 90°' button if the page is sideways.")

# ----- Detected outline + final result side by side ---------------------- #
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Detected page")
    st.image(to_display(result.contour_overlay), use_container_width=True)
with col_b:
    st.subheader(f"Final scan — {MODE_LABELS[mode]}")
    st.image(to_display(result.enhanced), use_container_width=True)

# ----- Intermediate stages ----------------------------------------------- #
if show_stages:
    st.markdown("### Pipeline stages")
    stage_items = [
        ("1 · Grayscale", result.gray),
        ("2 · Gaussian blur", result.blurred),
        ("3 · Canny edges", result.edges),
        ("4 · Morph. close", result.closed),
        ("5 · Contour overlay", result.contour_overlay),
        ("6 · Warped (flattened)", result.warped),
    ]
    cols = st.columns(3)
    for i, (caption, img) in enumerate(stage_items):
        with cols[i % 3]:
            st.image(to_display(img), caption=caption, use_container_width=True)

# ----- OCR --------------------------------------------------------------- #
if run_ocr:
    st.markdown("### OCR text")
    with st.spinner("Running OCR..."):
        text = extract_text(result.enhanced)
    st.text_area("Recognised text", text, height=180)

# ----- Downloads --------------------------------------------------------- #
st.markdown("### Download")
dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "⬇️ This scan (PNG)",
        data=to_png_bytes(result.enhanced),
        file_name=f"{selected_name.rsplit('.', 1)[0]}_scan.png",
        mime="image/png",
    )
with dl2:
    # Build a combined PDF of ALL uploaded images in the selected mode.
    all_enhanced = []
    for f in uploads:
        try:
            img = decode_upload(f)
        except ValueError:
            continue
        # Respect any rotation the user applied to this image.
        img = apply_rotation(img, st.session_state.get(f"rot::{f.name}", 0))
        all_enhanced.append(scan_image(img, mode=mode,
                                       process_height=process_height).enhanced)
    st.download_button(
        f"⬇️ All {len(all_enhanced)} page(s) (PDF)",
        data=to_pdf_bytes(all_enhanced),
        file_name="scanned_document.pdf",
        mime="application/pdf",
    )

st.caption("Built with OpenCV + Streamlit · no deep learning, no paid APIs.")
