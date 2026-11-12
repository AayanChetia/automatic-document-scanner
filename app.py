"""
app.py - polished Streamlit web UI for the Automatic Document Scanner.

Run with:
    streamlit run app.py

This is a UI/UX layer over the classical CV pipeline in scanner/. The computer-
vision logic is unchanged; this module only handles layout, per-page state and
export. The interface is a 3-step workflow — Upload -> Review & adjust -> Export
— designed to look clean on a projector for a live demo.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from scanner import scan_image
from scanner.enhance import MODES
from scanner.ocr import extract_text, tesseract_available
from scanner.transform import order_points
from scanner.utils import to_display

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    _HAS_COORDS = True
except Exception:  # pragma: no cover
    _HAS_COORDS = False

# --------------------------------------------------------------------------- #
# Page config + styling
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Document Scanner", page_icon="📄", layout="wide")

ACCENT = "#2DD4BF"
MODE_LABELS = {
    "original": "Original",
    "gray": "Grayscale",
    "magic": "Magic colour",
    "bw": "Black & White",
    "otsu": "B&W (Otsu)",
}
# Modes offered in the review toolbar (Otsu stays available via the pipeline).
TOOLBAR_MODES = ["original", "gray", "magic", "bw"]
DEFAULT_MODE = "magic"
CONFIDENCE_OK = 0.5
PREVIEW_H = 430          # cap preview image height (px)
THUMB_H = 150

STAGE_CAPTIONS = [
    ("Grayscale", "Drop colour: edges come from intensity, not hue"),
    ("Blur (Gaussian + median)", "Smooth noise & erase fine text so real edges remain"),
    ("Canny edges", "Finds strong intensity changes = candidate edges"),
    ("Morphological close", "Bridge small gaps so the page outline is one closed loop"),
    ("Detected page", "Largest validated convex quad = the document corners"),
    ("Warped (flattened)", "Homography removes the camera angle -> top-down page"),
]

CSS = f"""
<style>
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }}
div[data-testid="stImage"] {{ display: flex; justify-content: center; }}
div[data-testid="stImage"] img {{ border-radius: 8px; }}
.app-header {{
    display: flex; align-items: center; gap: 14px; margin-bottom: 2px;
}}
.app-title {{ font-size: 2.0rem; font-weight: 800; letter-spacing: -.5px; }}
.app-badge {{
    background: rgba(45,212,191,.15); color: {ACCENT}; border: 1px solid {ACCENT};
    padding: 3px 10px; border-radius: 999px; font-size: .72rem; font-weight: 700;
    white-space: nowrap;
}}
.app-tag {{ color: #9aa4b2; margin: 0 0 6px 2px; }}
.step-head {{
    font-size: 1.15rem; font-weight: 700; margin: 6px 0 2px;
    border-left: 4px solid {ACCENT}; padding-left: 10px;
}}
.pill {{
    display:inline-block; padding: 5px 12px; border-radius: 999px;
    font-size: .82rem; font-weight: 700;
}}
.pill-green  {{ background: rgba(34,197,94,.16);  color:#34d399; border:1px solid #34d399; }}
.pill-yellow {{ background: rgba(234,179,8,.16);   color:#fbbf24; border:1px solid #fbbf24; }}
.pill-blue   {{ background: rgba(59,130,246,.16);  color:#60a5fa; border:1px solid #60a5fa; }}
.card-title {{ font-weight:700; color:#cbd5e1; margin-bottom:6px; }}
.thumb-name {{ font-size:.72rem; color:#9aa4b2; text-align:center; overflow:hidden;
    white-space:nowrap; text-overflow:ellipsis; }}
.stDownloadButton button, .stButton button {{ border-radius: 8px; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    st.session_state.setdefault("pages", [])         # list of page dicts
    st.session_state.setdefault("sel_id", None)      # selected page id
    st.session_state.setdefault("next_id", 1)
    st.session_state.setdefault("up_round", 0)       # uploader reset counter


def new_page(name: str, data: bytes) -> dict:
    pid = st.session_state.next_id
    st.session_state.next_id += 1
    return {"id": pid, "name": name, "bytes": data,
            "rotation": 0, "corners": None, "manual": False, "mode": DEFAULT_MODE,
            "clicks": []}


def ingest(files) -> None:
    """Append newly uploaded files to the page set (does not replace)."""
    added = 0
    for f in files:
        data = f.getvalue()
        st.session_state.pages.append(new_page(f.name, data))
        added += 1
    if added:
        if st.session_state.sel_id is None:
            st.session_state.sel_id = st.session_state.pages[0]["id"]
        st.session_state.up_round += 1      # reset the uploader widget
        st.toast(f"Added {added} page(s)", icon="➕")
        st.rerun()


def get_page(pid):
    for p in st.session_state.pages:
        if p["id"] == pid:
            return p
    return None


def remove_page(pid) -> None:
    pages = st.session_state.pages
    idx = next((i for i, p in enumerate(pages) if p["id"] == pid), None)
    if idx is None:
        return
    pages.pop(idx)
    if st.session_state.sel_id == pid:
        st.session_state.sel_id = pages[min(idx, len(pages) - 1)]["id"] if pages else None


def move_page(pid, delta) -> None:
    pages = st.session_state.pages
    idx = next((i for i, p in enumerate(pages) if p["id"] == pid), None)
    if idx is None:
        return
    j = idx + delta
    if 0 <= j < len(pages):
        pages[idx], pages[j] = pages[j], pages[idx]


# --------------------------------------------------------------------------- #
# Processing (cached)
# --------------------------------------------------------------------------- #
def apply_rotation(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


@st.cache_data(show_spinner=False, max_entries=64)
def process_cached(img_bytes: bytes, rotation: int, mode: str,
                   corners_key: tuple | None, process_height: int):
    """Decode -> rotate -> run the pipeline. Cached so reruns are instant."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = apply_rotation(img, rotation)
    corners = (np.array(corners_key, np.float32).reshape(4, 2)
               if corners_key else None)
    return scan_image(img, mode=mode, process_height=process_height,
                      corners_override=corners)


def process(page: dict, process_height: int):
    ckey = tuple(np.asarray(page["corners"]).ravel().tolist()) if page["corners"] else None
    return process_cached(page["bytes"], page["rotation"], page["mode"],
                          ckey, process_height)


def page_status(page: dict, result) -> tuple[str, str, str]:
    """Return (emoji, pill_class, label) for a page's detection state."""
    if page["manual"]:
        return "✋", "pill-blue", "Manual corners"
    if result.found and result.confidence >= CONFIDENCE_OK:
        return "✅", "pill-green", f"Auto-detected ({result.method})"
    return "⚠️", "pill-yellow", "Low confidence: please adjust corners"


def capped_width(img: np.ndarray, target_h: int) -> int:
    h, w = img.shape[:2]
    return max(1, int(round(target_h * w / h)))


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #
def _cv_to_pil(img: np.ndarray, quality: int) -> Image.Image:
    """Convert an OpenCV image to PIL, optionally JPEG-compressed for size."""
    if quality < 100:
        enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


PAGE_SIZES_PX = {"A4": (1654, 2339), "Letter": (1700, 2200)}


def _fit_on_page(pil_img: Image.Image, size_px: tuple[int, int]) -> Image.Image:
    """Place an image centred on a white page of the given pixel size."""
    page = Image.new("RGB", size_px, "white")
    iw, ih = pil_img.size
    scale = min((size_px[0] * 0.92) / iw, (size_px[1] * 0.92) / ih)
    new = pil_img.resize((max(1, int(iw * scale)), max(1, int(ih * scale))))
    page.paste(new, ((size_px[0] - new.size[0]) // 2, (size_px[1] - new.size[1]) // 2))
    return page


def build_pdf(images: list[np.ndarray], page_size: str, quality: int) -> bytes:
    pages = [_cv_to_pil(im, quality) for im in images]
    if page_size in PAGE_SIZES_PX:
        pages = [_fit_on_page(p, PAGE_SIZES_PX[page_size]) for p in pages]
    buf = io.BytesIO()
    pages[0].save(buf, "PDF", save_all=True, append_images=pages[1:], resolution=200.0)
    return buf.getvalue()


def build_zip(items: list[tuple[str, np.ndarray]], fmt: str, quality: int) -> bytes:
    buf = io.BytesIO()
    ext = ".jpg" if fmt == "JPG" else ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "JPG" else []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (name, img) in enumerate(items, 1):
            ok, enc = cv2.imencode(ext, img, params)
            if ok:
                zf.writestr(f"{i:02d}_{name}{ext}", enc.tobytes())
    return buf.getvalue()


def human_size(n: int) -> str:
    return f"{n/1024:.0f} KB" if n < 1024 * 1024 else f"{n/1024/1024:.1f} MB"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
init_state()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    process_height = st.slider("Detection resolution (px)", 300, 900, 450, 50,
                               help="Detection runs on a down-scaled copy; the "
                                    "final warp always uses full resolution.")
    show_stages = st.toggle("Show pipeline stages", value=True)

    st.markdown("### ℹ️ About")
    st.markdown(
        "- **Classical CV** — no deep learning\n"
        "- Canny edges → contour quad → homography warp\n"
        "- Adaptive threshold for crisp B&W\n"
        "- Runs in real time on a CPU")

    st.markdown("### 💡 Tips")
    st.markdown(
        "- Shoot on a **dark, plain** background\n"
        "- Keep **all 4 corners** in the frame\n"
        "- Use **Rotate** if the page is sideways\n"
        "- Low confidence? **click the 4 corners**")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="app-header"><span class="app-title">📄 Automatic Document '
    'Scanner</span><span class="app-badge">Classical CV · OpenCV</span></div>',
    unsafe_allow_html=True)
st.markdown('<p class="app-tag">Turn a photo of a document into a flat, clean, '
            'enhanced scan — every step is explainable computer vision.</p>',
            unsafe_allow_html=True)
st.divider()


# --------------------------------------------------------------------------- #
# STEP 1 — Upload
# --------------------------------------------------------------------------- #
st.markdown('<div class="step-head">① Upload</div>', unsafe_allow_html=True)

label = ("📥 Drag photos here or click to browse · JPG/PNG"
         if not st.session_state.pages
         else "➕ Add more photos (appends to your current set)")
uploaded = st.file_uploader(
    label, type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
    accept_multiple_files=True, key=f"uploader_{st.session_state.up_round}")
if uploaded:
    ingest(uploaded)

if not st.session_state.pages:
    st.info("⬆️ Upload at least one document photo to begin. "
            "Tip: a dark, plain background makes the page edges easy to detect.")
    st.stop()

# Thumbnail strip of all pages with status + controls.
st.caption(f"{len(st.session_state.pages)} page(s) · click **Select** to edit, "
           "reorder for the PDF with ⬆ ⬇")
cols = st.columns(max(1, min(len(st.session_state.pages), 6)))
for i, page in enumerate(list(st.session_state.pages)):
    with cols[i % len(cols)]:
        res = process(page, process_height)
        emoji, _, _ = page_status(page, res)
        thumb = to_display(res.enhanced)
        st.image(thumb, width=capped_width(res.enhanced, THUMB_H))
        is_sel = page["id"] == st.session_state.sel_id
        st.markdown(f'<div class="thumb-name">{emoji} '
                    f'{"▶ " if is_sel else ""}{page["name"]}</div>',
                    unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)
        b1.button("👁", key=f"sel{page['id']}", help="Select",
                  on_click=lambda pid=page["id"]: st.session_state.update(sel_id=pid))
        b2.button("⬆", key=f"up{page['id']}", help="Move up",
                  on_click=move_page, args=(page["id"], -1))
        b3.button("⬇", key=f"dn{page['id']}", help="Move down",
                  on_click=move_page, args=(page["id"], 1))
        b4.button("✕", key=f"rm{page['id']}", help="Remove",
                  on_click=remove_page, args=(page["id"],))

if st.session_state.sel_id is None and st.session_state.pages:
    st.session_state.sel_id = st.session_state.pages[0]["id"]

page = get_page(st.session_state.sel_id)
st.divider()


# --------------------------------------------------------------------------- #
# STEP 2 — Review & adjust
# --------------------------------------------------------------------------- #
st.markdown('<div class="step-head">② Review &amp; adjust</div>', unsafe_allow_html=True)
st.caption(f"Editing: **{page['name']}**")

# ---- Toolbar ----
t1, t2, t3, t4, t5 = st.columns([1, 1, 3.2, 1.2, 1])
with t1:
    if st.button("⟲ Rotate L", use_container_width=True):
        page["rotation"] = (page["rotation"] + 270) % 360
        page["corners"], page["manual"], page["clicks"] = None, False, []
        st.rerun()
with t2:
    if st.button("⟳ Rotate R", use_container_width=True):
        page["rotation"] = (page["rotation"] + 90) % 360
        page["corners"], page["manual"], page["clicks"] = None, False, []
        st.rerun()
with t3:
    chosen = st.segmented_control(
        "Enhancement mode", options=TOOLBAR_MODES, default=page["mode"],
        format_func=lambda m: MODE_LABELS[m], key=f"mode{page['id']}",
        label_visibility="collapsed")
    if chosen and chosen != page["mode"]:
        page["mode"] = chosen
        st.rerun()
with t4:
    if st.button("🔄 Auto-detect", use_container_width=True,
                 help="Clear manual corners and re-run automatic detection"):
        page["corners"], page["manual"], page["clicks"] = None, False, []
        st.rerun()
with t5:
    if st.button("♻ Reset", use_container_width=True, help="Reset this page"):
        page.update(rotation=0, corners=None, manual=False, clicks=[], mode=DEFAULT_MODE)
        st.rerun()

if st.button("📑 Apply this mode to all pages"):
    for p in st.session_state.pages:
        p["mode"] = page["mode"]
    st.toast(f"Applied '{MODE_LABELS[page['mode']]}' to all pages", icon="📑")

# ---- Process selected page ----
with st.spinner("Processing…"):
    result = process(page, process_height)

emoji, pill_class, pill_label = page_status(page, result)
pill_html = f'<span class="pill {pill_class}">{pill_label}'
if not page["manual"] and result.found and result.confidence >= CONFIDENCE_OK:
    pill_html += f' · {result.confidence*100:.0f}%'
pill_html += "</span>"
st.markdown(pill_html, unsafe_allow_html=True)
if result.clipped:
    st.info("📐 Page extends beyond the photo: some corners were estimated. "
            "Corners outside the frame can't be fully recovered — adjust them "
            "manually if needed.")
st.write("")

# ---- Main side-by-side cards ----
cL, cR = st.columns(2)
with cL:
    with st.container(border=True):
        st.markdown('<div class="card-title">Original · detected outline</div>',
                    unsafe_allow_html=True)
        st.image(to_display(result.contour_overlay),
                 width=capped_width(result.contour_overlay, PREVIEW_H))
with cR:
    with st.container(border=True):
        st.markdown(f'<div class="card-title">Final scan · {MODE_LABELS[page["mode"]]}'
                    '</div>', unsafe_allow_html=True)
        st.image(to_display(result.enhanced),
                 width=capped_width(result.enhanced, PREVIEW_H))

# ---- Manual corners: click directly on the image ----
needs = page["manual"] or not (result.found and result.confidence >= CONFIDENCE_OK)
with st.expander("✋ Set corners manually — click the 4 corners on the image",
                 expanded=needs and _HAS_COORDS):
    base = result.original
    oh, ow = base.shape[:2]
    disp_h = 440
    disp_w = capped_width(base, disp_h)

    overlay = base.copy()
    scale = ow / disp_w
    for (x, y) in page["clicks"]:
        cv2.circle(overlay, (int(x), int(y)), max(4, int(6 * scale)), (0, 0, 255), -1)
    if page["corners"] is not None:
        pts = np.array(page["corners"], np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), max(2, int(3 * scale)))

    if _HAS_COORDS:
        st.caption("Click **top-left → top-right → bottom-right → bottom-left**. "
                   "Four clicks set the page; use Reset to start over.")
        disp_rgb = cv2.cvtColor(cv2.resize(overlay, (disp_w, disp_h)), cv2.COLOR_BGR2RGB)
        val = streamlit_image_coordinates(
            Image.fromarray(disp_rgb), key=f"clicker_{page['id']}_{page['rotation']}")
        last_key = f"lastclk_{page['id']}"
        if val is not None:
            cur = (val["x"], val["y"])
            if st.session_state.get(last_key) != cur and len(page["clicks"]) < 4:
                st.session_state[last_key] = cur
                fx, fy = cur[0] * ow / disp_w, cur[1] * oh / disp_h
                page["clicks"].append([fx, fy])
                if len(page["clicks"]) == 4:
                    page["corners"] = order_points(
                        np.array(page["clicks"], np.float32)).tolist()
                    page["manual"] = True
                    st.toast("Manual corners set", icon="✋")
                st.rerun()
        c1, c2 = st.columns(2)
        c1.write(f"Points placed: **{len(page['clicks'])}/4**")
        if c2.button("↺ Reset corners", key=f"rc{page['id']}"):
            page["corners"], page["manual"], page["clicks"] = None, False, []
            st.session_state.pop(last_key, None)
            st.rerun()
    else:
        st.info("Install `streamlit-image-coordinates` to click corners. "
                "Use the sliders below to adjust them instead.")

    # Fine-tune sliders (secondary).
    with st.expander("🎚️ Fine-tune corners with sliders"):
        seed = (np.array(page["corners"], np.float32) if page["corners"] is not None
                else result.corners_full).astype(int)
        labels = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
        scols = st.columns(4)
        newp = []
        for i, (lab, col) in enumerate(zip(labels, scols)):
            with col:
                st.markdown(f"**{lab}**")
                x = st.slider(f"{lab} x", 0, ow - 1, int(seed[i][0]), key=f"sx{page['id']}{i}")
                y = st.slider(f"{lab} y", 0, oh - 1, int(seed[i][1]), key=f"sy{page['id']}{i}")
                newp.append([x, y])
        if st.button("Apply slider corners", key=f"apsl{page['id']}"):
            page["corners"] = order_points(np.array(newp, np.float32)).tolist()
            page["manual"] = True
            st.rerun()

# ---- Pipeline stages (collapsed) ----
if show_stages:
    with st.expander("🔍 Show how it works: pipeline stages"):
        stage_imgs = [result.gray, result.blurred, result.edges, result.closed,
                      result.contour_overlay, result.warped]
        grid = st.columns(3)
        for i, (img, (title, cap)) in enumerate(zip(stage_imgs, STAGE_CAPTIONS)):
            with grid[i % 3]:
                st.image(to_display(img), width=capped_width(
                    img if img.ndim == 2 else img, 190))
                st.markdown(f"**{i+1}. {title}**  \n<span style='color:#9aa4b2;"
                            f"font-size:.78rem'>{cap}</span>", unsafe_allow_html=True)

st.divider()


# --------------------------------------------------------------------------- #
# STEP 3 — Export
# --------------------------------------------------------------------------- #
st.markdown('<div class="step-head">③ Export</div>', unsafe_allow_html=True)

# Build enhanced images for ALL pages in current order.
ordered = st.session_state.pages
all_imgs = [process(p, process_height).enhanced for p in ordered]
cur_result = result

opt1, opt2, opt3 = st.columns(3)
with opt1:
    page_size = st.selectbox("PDF page size", ["Original", "A4", "Letter"])
with opt2:
    quality = st.slider("Image quality", 40, 100, 90,
                        help="Lower = smaller files (more JPEG compression).")
with opt3:
    default_name = f"scan_{datetime.now():%Y%m%d}"
    fname = st.text_input("File name", value=default_name)
fname = (fname.strip() or default_name).replace(" ", "_")

b1, b2, b3 = st.columns(3)
with b1:
    with st.spinner("Building PDF…"):
        pdf_bytes = build_pdf(all_imgs, page_size, quality)
    st.download_button(f"📄 Download PDF · {len(all_imgs)} page(s)", pdf_bytes,
                       file_name=f"{fname}.pdf", mime="application/pdf",
                       use_container_width=True, type="primary")
    st.caption(f"Estimated size: {human_size(len(pdf_bytes))}")
with b2:
    png_bytes = cv2.imencode(".png", cur_result.enhanced)[1].tobytes()
    st.download_button("🖼️ Current page (PNG)", png_bytes,
                       file_name=f"{fname}_{page['name'].rsplit('.',1)[0]}.png",
                       mime="image/png", use_container_width=True)
    st.caption(f"Estimated size: {human_size(len(png_bytes))}")
with b3:
    zip_fmt = "PNG" if quality >= 100 else "JPG"
    zip_bytes = build_zip([(p["name"].rsplit(".", 1)[0], process(p, process_height).enhanced)
                           for p in ordered], zip_fmt, quality)
    st.download_button(f"🗜️ All as ZIP ({zip_fmt})", zip_bytes,
                       file_name=f"{fname}.zip", mime="application/zip",
                       use_container_width=True)
    st.caption(f"Estimated size: {human_size(len(zip_bytes))}")

# ---- OCR ----
with st.expander("📝 Extract text (OCR) from the current page"):
    if not tesseract_available():
        st.warning(
            "Tesseract OCR engine not found. Install it to enable text extraction:\n\n"
            "- **Windows:** https://github.com/UB-Mannheim/tesseract/wiki , then add "
            "`tesseract.exe` to your PATH\n"
            "- **macOS:** `brew install tesseract`\n"
            "- **Linux:** `sudo apt install tesseract-ocr`\n\n"
            "The scanner works fully without it; OCR is optional.")
    else:
        if st.button("Run OCR on current page"):
            with st.spinner("Recognising text…"):
                st.session_state[f"ocr_{page['id']}"] = extract_text(cur_result.enhanced)
        text = st.session_state.get(f"ocr_{page['id']}")
        if text is not None:
            st.caption("Recognised text (use the copy icon in the top-right):")
            st.code(text or "(no text found)", language=None)
            st.download_button("📥 Download .txt", (text or "").encode("utf-8"),
                               file_name=f"{fname}_{page['name'].rsplit('.',1)[0]}.txt",
                               mime="text/plain")

st.caption("Built with OpenCV + Streamlit · classical computer vision, no deep "
           "learning, no paid APIs.")
