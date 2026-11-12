# Presentation Outline — Automatic Document Scanner using Computer Vision

### 10–12 slide deck, ~8–10 minutes + live demo

Each slide lists **content** (what to say / show) and **visual** (what to put on
the slide). Keep text on slides minimal — speak the detail.

---

### Slide 1 — Title

- **Content:** Project title, your name, roll number, course, guide's name,
  institution, date.
- **Visual:** A clean before→after image (photo of a tilted page → flat B&W
  scan), `docs/images/before_after.png`; or a screenshot of the polished web app,
  `docs/images/streamlit_ui.png`.

---

### Slide 2 — Motivation & Problem

- **Content:** Phones replaced flatbed scanners (CamScanner, Adobe Scan). The
  problem: a photo of a page is tilted, perspective-distorted and unevenly lit.
  Goal — automatically detect the page, flatten it, and clean it up.
- **Visual:** Side-by-side of a messy phone photo vs. the clean scan we want.

---

### Slide 3 — Objectives

- **Content:** The five objectives: detect corners → correct perspective →
  enhance → provide interfaces (CLI/web/webcam) → evaluate. Stress: **classical
  CV, no deep learning, runs on a laptop, fully explainable.**
- **Visual:** A simple icon/bullet list.

---

### Slide 4 — Why classical CV (not deep learning)

- **Content:** Comparison table — transparency, no training data, runs on CPU in
  ms, perfect for a CV course. Deep learning is more robust to clutter but is a
  black box needing data/GPU. We use classical as the core and note ML as future
  work for the hard case.
- **Visual:** The classical-vs-DL comparison table from the report (§4).

---

### Slide 5 — System Architecture / Pipeline

- **Content:** Walk the block diagram left to right: preprocess → detect → warp →
  enhance → export. Mention the modular `scanner/` package mirrors the diagram.
- **Visual:** The **Mermaid block diagram** (render it to an image) from the
  report §5.

---

### Slide 6 — Step 1–2: Preprocess & Edge Detection

- **Content:** Grayscale (intensity is what matters) → Gaussian blur (gradients
  amplify noise) → auto-Canny (thresholds from the median, adapts to lighting) →
  morphological close (join broken edges so contours are closed).
- **Visual:** The first row of `docs/images/pipeline_stages.png`
  (grayscale, blur, Canny, close).

---

### Slide 7 — Step 3: Document Detection

- **Content:** Find contours → sort by area → `approxPolyDP`
  (Douglas–Peucker) simplifies to a polygon → keep the largest **convex 4-point**
  shape. Fallbacks: Hough lines + intersection, then manual corners. Emphasise
  **why 4 points**.
- **Visual:** "Detected page" panel (green outline + red corners) from the montage.

---

### Slide 8 — Step 4: Perspective Transform (the key idea)

- **Content:** A camera maps the planar page by a **homography** (3×3, 8 DOF).
  Four corner correspondences give 8 equations → exactly solvable.
  `getPerspectiveTransform` + `warpPerspective` flatten the page; output size
  from the quad's side lengths. One-line homography equation.
- **Visual:** Tilted quad → flat rectangle animation/arrow; the homography
  matrix equation.

---

### Slide 9 — Step 5: Enhancement Modes

- **Content:** Original / Grayscale / **Magic colour (CLAHE)** — local contrast
  without hue shift / **B&W adaptive threshold** — local thresholds beat a global
  cut under uneven light. Mention Otsu as the global baseline we compare against.
- **Visual:** The same warped page shown in all four modes.

---

### Slide 10 — Results & Evaluation

- **Content:** 32-image labelled test set, IoU-based success. Headline: **87.5%
  success, 85.4% mean IoU, 8.4 ms/image**; adaptive **97.4%** vs Otsu **95.7%**.
  Robust to tilt & low light; cluttered is the hard case (50%).
- **Visual:** `results/success_by_condition.png` and
  `results/threshold_comparison.png`.

---

### Slide 11 — Limitations & Future Scope

- **Content:** Limits — cluttered/low-contrast backgrounds, occlusion, extreme
  angles, synthetic benchmark. Future — learned page segmentation (U-Net/HED),
  shadow/glare removal, mobile deployment, better OCR/searchable PDF.
- **Visual:** A failure example (a cluttered image) next to the "future work"
  bullets.

---

### Slide 12 — Conclusion + Live Demo / Q&A

- **Content:** Recap: a complete, real-time, explainable scanner with CLI, web
  app and webcam, plus a reproducible evaluation. Thank the audience; invite
  questions. Then run the demo.
- **Visual:** Project GitHub link + a QR code; "Demo" prompt.

---

## Live-demo plan (rehearse this!)

> Have the Streamlit app already running before you start to avoid cold-start
> lag: `streamlit run app.py`. The UI is a clean **① Upload → ② Review & adjust →
> ③ Export** workflow (see `docs/images/streamlit_ui.png`).

1. **Web app (main demo).**
   - **① Upload** a tilted sample photo (`data/test_images/tilted_0.png`). Point
     out the thumbnail strip with its ✅ detection status and the green
     **"Auto-detected (contour) · 90%"** confidence pill.
   - **② Review:** show the side-by-side _Original · detected outline_ vs
     _Final scan_ cards. Switch enhancement mode with the segmented control
     (Magic colour → B&W) so the audience sees the difference instantly. Expand
     **"🔍 Show how it works: pipeline stages"** to walk the captioned stages.
   - Upload a **cluttered** image to honestly show a low-confidence case (yellow
     pill), then **click the 4 corners directly on the image** to fix it — this
     shows you understand the limitation _and_ the manual fallback. Use the
     **Rotate** buttons if a photo is sideways.
   - **③ Export:** choose A4, then click **📄 Download PDF** to prove export
     works (note the estimated file size).
2. **CLI (quick).** In a terminal:
   `python scan.py -i data/test_images -o results/demo.pdf -m bw`
   — show a whole folder scanned into one PDF in one command.
3. **Webcam (if a camera is available).** `python webcam.py` — hold up a real
   page, show the live outline tracking it, press SPACE to capture.
4. **Evaluation (optional).** `python evaluate.py` — show the numbers and charts
   regenerating live.

**Backup:** keep screenshots of each step in the slides in case the laptop/camera
misbehaves. Pre-generate `results/demo.pdf` beforehand as a fallback.

**Timing:** aim for ~2 minutes on the demo; don't let it overrun the talk.
