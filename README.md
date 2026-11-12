# Automatic Document Scanner using Computer Vision

Turn a photo of a document into a flat, cropped, enhanced **scan** (PNG/JPG or
PDF) — like CamScanner or Adobe Scan, but built from scratch with **classical
computer vision** in OpenCV. It detects the page edges, corrects the perspective
with a homography, and cleans the result with adaptive thresholding. No deep
learning, no paid APIs, runs in real time on a normal laptop — and every step is
explainable.

![Pipeline stages](docs/images/pipeline_stages.png)

---

## ✨ Features

- **Full classical CV pipeline:** grayscale → Gaussian blur → auto-Canny →
  morphological close → contour/`approxPolyDP` quad detection → 4-point
  perspective warp → enhancement.
- **Robust multi-cue detection:** candidates from contour search (grayscale **and**
  LAB/HSV colour + morphological-gradient edges), Hough lines, and a **GrabCut**
  segmentation fallback are each **validated** (area, convexity, angles, aspect)
  and **scored** (area, rectangularity, edge support, inside/outside contrast) —
  the best wins, and the UI shows which method. Handles a **white page on a white
  desk**, and is **border-aware** (flags a page that runs off the frame). Manual
  corner picking is the final fallback.
- **Four enhancement modes:** Original colour, Grayscale, **Magic colour**
  (CLAHE contrast boost) and **Black & White** (adaptive threshold), with Otsu
  as a comparison baseline.
- **Three interfaces:** command line, a polished **Streamlit** web app, and a
  **live webcam** mode.
- **Large preview modal:** inspect any page full-resolution with zoom/pan,
  Original/Final/Side-by-side views, page navigation and download.
- **Optional OCR** via Tesseract (degrades gracefully if not installed).
- **Export** to PNG/JPG or a combined multi-page **PDF**.
- **Quantitative evaluation:** success rate, corner **IoU**, timing, and
  adaptive-vs-Otsu comparison, with Matplotlib charts.
- **Reproducible:** synthetic test-set generator so everything runs end to end
  before you add your own photos.

---

## 🧭 Pipeline overview

```mermaid
flowchart LR
    A[Photo] --> B[Preprocess<br/>gray, blur, Canny, close]
    B --> C[Detect page<br/>contours + approxPolyDP]
    C -->|fallback| D[Hough lines / manual]
    C --> E[Order 4 corners]
    D --> E
    E --> F[Perspective warp<br/>homography]
    F --> G[Enhance<br/>bw / magic / gray]
    G --> H[Export PNG / PDF + OCR]
```

Full theory for each step is in [`docs/report.md`](docs/report.md).

---

## 🚀 Setup

Requires **Python 3.11+** (developed on 3.14). From the project folder:

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (cmd):
venv\Scripts\activate.bat
# macOS / Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

**Optional — OCR:** install the Tesseract engine and make sure `tesseract` is on
your PATH. Windows build: <https://github.com/UB-Mannheim/tesseract/wiki>. The
app works fine without it; OCR simply reports that it's unavailable.

---

## 🖥️ How to run

### 1. Streamlit web app (best for a live demo)
```bash
streamlit run app.py
```
A 3-step workflow: **① Upload** one or more photos (reorder them for the PDF),
**② Review & adjust** — rotate, switch enhancement mode with a segmented control,
and if auto-detection is low-confidence **click the 4 corners directly on the
image** (or fine-tune with sliders), **③ Export** to PDF (A4/Letter/Original),
PNG, or a ZIP, with optional OCR. Expand *"Show how it works"* to see every
pipeline stage with captions — great for the demo.

### 2. Command line (`scan.py`)
```bash
# Single image
python scan.py --input photo.jpg --output scan.png --mode bw

# A whole folder -> one scan per image
python scan.py --input ./data/test_images --output ./results/scans

# A whole folder -> one combined PDF
python scan.py --input ./data/test_images --output ./results/doc.pdf --mode magic

# Also run OCR
python scan.py --input photo.jpg --output scan.png --ocr
```
Modes: `original | gray | magic | bw (default) | otsu`.

### 3. Live webcam (`webcam.py`)
```bash
python webcam.py            # SPACE/C = capture, M = cycle mode, Q/ESC = quit
```

### 4. Evaluation
```bash
# Generate a synthetic labelled test set (first time / to refresh)
python data/make_synthetic.py --per 8

# Run the evaluation -> charts + metrics in results/
python evaluate.py

# Label your OWN photos (drop them in data/test_images first)
python label_tool.py
```

### 5. Tests
```bash
pytest -q
```

---

## 📊 Results

Measured on a 32-image synthetic test set (plain / cluttered / low-light /
tilted) with known ground-truth corners:

| Metric | Value |
|--------|-------|
| Detection success rate (IoU ≥ 0.85) | **90.6%** |
| Mean corner IoU | **93.4%** |
| Average processing time | **~70 ms/image** (common pages ~10 ms; GrabCut on hard cases) |
| Binarisation accuracy (adaptive vs Otsu) | **97.4%** vs **95.7%** |

| Condition | Success rate | Mean IoU |
|-----------|--------------|----------|
| Plain | 100% | 96.4% |
| Tilted | 100% | 95.1% |
| Low-light | 100% | 96.3% |
| Cluttered | 62.5% | 85.8% |

![Detection by condition](results/success_by_condition.png)
![Adaptive vs Otsu](results/threshold_comparison.png)

The detector is very robust to **tilt** and **low light**; **cluttered**
backgrounds and **dense-text pages that fill the frame** are the main challenge.
Every detected quad is validated (area / convexity / angles / aspect) and scored
for **confidence**; low-confidence cases show a warning and invite manual corner
adjustment instead of reporting a false success (future work: a learned page
segmenter).

---

## 📸 Screenshots

**The Streamlit web app** — a clean ①Upload → ②Review & adjust → ③Export
workflow with live detection, a confidence pill, and side-by-side preview:

![Streamlit UI](docs/images/streamlit_ui.png)

**Large preview modal** — click *🔍 View large* (or a thumbnail) to inspect a
page full-resolution: Original / Final scan / Side-by-side, a zoom slider
(Fit / 100 / 150 / 200%) in a scrollable pane, page navigation, and download:

![Large preview](docs/images/streamlit_preview.png)

| Pipeline stages | Before → after |
|-----------------|----------------|
| ![stages](docs/images/pipeline_stages.png) | ![before/after](docs/images/before_after.png) |

---

## 🗂️ Project structure

```
Automatic document scanner/
├── scanner/              # core CV package (one module per pipeline stage)
│   ├── preprocess.py     # gray, blur, auto-Canny, morphological close
│   ├── detect.py         # multi-cue candidates (colour edges + Hough + GrabCut), scored
│   ├── transform.py      # corner ordering + perspective warp (homography)
│   ├── enhance.py        # original / gray / magic (CLAHE) / bw / otsu
│   ├── ocr.py            # optional pytesseract wrapper
│   ├── export.py         # PNG/JPG save + multi-page PDF
│   ├── pipeline.py       # orchestration; returns every intermediate image
│   └── utils.py          # resize-with-ratio, IO, timing
├── app.py                # Streamlit web app
├── scan.py               # command-line interface
├── webcam.py             # live webcam scanning
├── evaluate.py           # metrics + charts
├── label_tool.py         # click 4 corners to label your own photos
├── data/
│   ├── make_synthetic.py # synthetic test-set generator
│   ├── test_images/      # the test photos
│   ├── pages/            # clean reference pages (for threshold scoring)
│   └── ground_truth.json # labelled corners
├── tests/                # pytest suite
├── docs/
│   ├── report.md         # full project report
│   ├── ppt_outline.md    # slide-by-slide presentation plan
│   ├── viva_questions.md # 46 viva Q&A
│   └── images/           # figures
├── results/              # charts + metrics (generated)
└── requirements.txt
```

---

## 📚 Academic deliverables

- **[docs/report.md](docs/report.md)** — full report (Abstract → References, with
  architecture diagram, methodology theory and results).
- **[docs/ppt_outline.md](docs/ppt_outline.md)** — 10–12 slide outline + live-demo
  plan.
- **[docs/viva_questions.md](docs/viva_questions.md)** — 46 likely viva questions
  with concise answers.

---

## ⚠️ Notes & limitations

- Works best when the document is the largest, fully-visible object on a
  contrasting background.
- Cluttered/low-contrast backgrounds and occluded edges are the main failure
  mode.
- Headline numbers are from synthetic data; use `label_tool.py` + `evaluate.py`
  to measure on your own photos.

## License

Educational project. Built with OpenCV, NumPy, Pillow, Streamlit and Matplotlib.
