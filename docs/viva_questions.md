# Viva Questions & Answers — Automatic Document Scanner

Short, defensible answers to the questions most likely to come up. Grouped by
topic. (46 questions.)

---

## A. Overview & design choices

**1. What does your project do, in one sentence?**
It takes a photograph of a document and automatically detects the page, removes
the perspective distortion so it looks scanned from above, and enhances it into a
clean image/PDF — using only classical computer vision.

**2. Why classical computer vision instead of deep learning?**
It's transparent (I can explain and defend every step), needs no training data or
GPU, runs in milliseconds on a CPU, and geometry solves the core problem exactly.
Deep learning would be a black box and overkill here; I note it as future work for
the hardest case (cluttered backgrounds).

**3. Walk me through your full pipeline.**
Resize (keep the scale ratio) → preprocess (grayscale, Gaussian blur, auto-Canny,
morphological close) → detect the document (contours + `approxPolyDP`, with Hough
and manual fallbacks) → order the 4 corners → perspective warp the full-resolution
image → enhance (adaptive threshold etc.) → export PNG/PDF, with optional OCR.

**4. Why resize the image for detection but warp the original?**
Detection is faster and more robust on a small image (noise/texture shrink away).
But quality needs full resolution, so I remember the scale ratio, detect on the
small copy, scale the corners back up, and warp the original pixels. Best of both.

**5. What are the main modules of your code?**
A `scanner/` package: `preprocess`, `detect`, `transform`, `enhance`, `ocr`,
`export`, `pipeline`, `utils` — each stage isolated with docstrings — plus
interfaces `scan.py` (CLI), `app.py` (Streamlit) and `webcam.py`.

---

## B. Preprocessing

**6. Why convert to grayscale first?**
Edges come from intensity changes, not colour. Grayscale drops information I don't
need, is ~3× cheaper, and Canny requires a single-channel input.

**7. Why apply Gaussian blur before Canny? (classic question)**
Canny differentiates the image to find gradients, and differentiation amplifies
noise — one noisy pixel can create a false edge. Blurring is a low-pass filter
that removes high-frequency noise and paper texture so Canny responds to the real
page boundary.

**8. What is a Gaussian blur / why Gaussian specifically?**
It convolves the image with a Gaussian kernel, a weighted average that weights
nearby pixels more. It's smooth, separable (fast), and optimally trades off
spatial and frequency localisation, so it denoises without creating ringing
artefacts.

**9. What does Canny edge detection do, step by step?**
(1) Compute gradient magnitude and direction (Sobel); (2) non-maximum suppression
thins edges to 1 px; (3) double-threshold hysteresis keeps strong edges and the
weak edges connected to them, discarding isolated weak ones.

**10. Why do you compute Canny thresholds automatically?**
Fixed thresholds fail when brightness changes. I set them from the image's median
intensity: `lower=(1−σ)·median`, `upper=(1+σ)·median` (σ=0.33). They adapt per
image to bright or dark photos.

**11. What is hysteresis thresholding in Canny?**
Two thresholds: pixels above the high one are definite edges; pixels between low
and high are kept only if connected to a definite edge. This links genuine edges
while rejecting noise.

**12. Why dilation and morphological closing after Canny?**
Canny edges often have small gaps. `findContours` only traces closed curves, so
gaps fragment the page outline. Dilation thickens edges and closing
(dilate→erode) bridges small breaks, making the boundary one closed contour.

**13. What's the difference between dilation, erosion, opening and closing?**
Dilation grows white regions; erosion shrinks them. Opening = erode then dilate
(removes small white noise). Closing = dilate then erode (fills small holes/gaps)
— which is what I need to seal edge gaps.

---

## C. Detection

**14. How do you find the document among all the contours?**
`findContours` returns every closed curve; I sort by area (the page is normally
the biggest object), approximate each with `approxPolyDP`, and pick the largest
one that is a convex quadrilateral above a minimum area.

**15. What is `approxPolyDP` / the Douglas–Peucker algorithm?**
It simplifies a curve to fewer points: recursively, it keeps the point farthest
from the current chord if that distance exceeds `epsilon`, else discards all
in-between points. It collapses a wiggly contour into a clean polygon.

**16. How do you choose epsilon?**
`epsilon = 0.02 × perimeter` — proportional to contour size, so it scales with the
document. Larger epsilon → fewer vertices (too large loses real corners); smaller
→ keeps noise. 2% of the perimeter is a well-known sweet spot for pages.

**17. Why do you require exactly 4 points and convexity?**
A rectangular page seen in perspective is a convex quadrilateral — 4 corners.
Requiring 4 points and convexity rejects text blocks, logos and noisy contours,
and gives exactly the four correspondences the perspective transform needs.

**18. What if no 4-point contour is found?**
Fallbacks: (1) Hough line transform — detect straight lines and intersect the
outermost horizontal/vertical pair to get corners; (2) if that fails, use the
full frame, and in the web app the user can drag the four corners manually.

**19. What is the Hough transform?**
A voting scheme: each edge pixel votes for all lines through it, parameterised as
`x·cosθ + y·sinθ = ρ`. Peaks in (ρ, θ) accumulator space are the dominant
straight lines — useful to recover page edges when the contour is broken.

**20. How do you get corners from Hough lines?**
Split lines into near-vertical and near-horizontal by θ, take the outermost line
of each group (extreme ρ), and solve the 2×2 linear system for the intersection
of each horizontal with each vertical — four intersections = four corners.

---

## D. Perspective transform (the core)

**21. What is a homography?**
A 3×3 projective transformation (8 degrees of freedom) that maps one plane to
another in homogeneous coordinates. Because the document is planar, the mapping
from the photographed page to a flat rectangle is exactly a homography.

**22. Why do you need exactly four points for the transform?**
A homography has 8 unknowns. Each point correspondence gives 2 equations, so 4
corner pairs give 8 equations — just enough to solve for H uniquely. That's the
geometric reason a scanner detects four corners.

**23. Difference between affine and perspective transforms?**
Affine (6 DOF) preserves parallelism — it can't model a page where parallel edges
appear to converge. Perspective/projective (8 DOF) models that convergence
(vanishing points), which is exactly what camera tilt produces, so I need the
perspective transform.

**24. How is the output size of the scan decided?**
From the detected quad's own geometry: width = max(top edge, bottom edge),
height = max(left edge, right edge), using Euclidean distances between corners.
Taking the max avoids cropping the longer (nearer) edge.

**25. Why must corners be ordered before the transform?**
`getPerspectiveTransform` maps source point i to destination point i. If the order
is wrong, the page is rotated, mirrored or twisted. I order them canonically as
TL, TR, BR, BL.

**26. How do you order the corners?**
Using sums and differences of coordinates: top-left has the smallest x+y,
bottom-right the largest; top-right has the smallest y−x, bottom-left the largest.
Robust to moderate rotation.

**27. What does `warpPerspective` actually do?**
For each output pixel it applies the inverse homography to find the source
location and interpolates the colour there (inverse mapping + bilinear
interpolation), filling the destination rectangle with the flattened page.

---

## E. Enhancement & thresholding

**28. Global vs adaptive thresholding — what's the difference and why adaptive?**
Global uses one threshold for the whole image; it fails under uneven lighting.
Adaptive computes a local threshold per pixel from its neighbourhood, so text
stays legible even when one side of the page is darker — the usual case for phone
photos.

**29. What is Otsu's method?**
It automatically chooses the single global threshold that minimises the
within-class variance (equivalently maximises between-class variance) of the two
pixel groups. Great under even lighting; I use it as a baseline to compare
against adaptive thresholding.

**30. Your results: adaptive 97.4% vs Otsu 95.7% — what does that show?**
On unevenly-lit pages the local (adaptive) threshold reconstructs the text more
accurately than the single global Otsu cut, confirming the theory. Otsu is close
on evenly-lit pages but degrades when lighting varies.

**31. What is CLAHE and why use it for "magic colour"?**
Contrast Limited Adaptive Histogram Equalisation. It equalises small tiles
independently (local contrast), clips the histogram to avoid amplifying noise, and
interpolates between tiles. I apply it to the L channel in LAB space to boost
contrast without changing the colours.

**32. Why LAB colour space for CLAHE and not RGB/BGR?**
LAB separates lightness (L) from colour (a, b). Enhancing only L changes
brightness/contrast while leaving hue untouched, so the page cleans up without
colour casts.

---

## F. Evaluation, OCR, limitations

**33. How did you evaluate the system without lots of real photos?**
I generate a synthetic test set: project a known clean page through a known
perspective onto varied backgrounds — so the true corners are known for free.
`label_tool.py` lets me label real photos the same way. Then `evaluate.py`
reports success rate, IoU, time and binarisation accuracy.

**34. What is IoU and why use it for corner accuracy?**
Intersection-over-Union: the overlap area of the detected quad and the ground-
truth quad divided by their union. It's a single 0–1 score for how well the
detected page matches the true page — standard in detection/segmentation.

**35. What are your system's main failure cases?**
Cluttered or low-contrast backgrounds and occluded page edges (my cluttered
success was 50%), very extreme angles, and folded/curled pages. The page must be
the dominant, fully-visible object.

**36. How would you improve the cluttered case?**
Add a learned page-segmentation model (U-Net / HED / a light MobileNet) to predict
the page mask, then keep my geometric warp — combining robustness with
explainable geometry. Also shadow/glare removal and better edge grouping.

**37. How does OCR work here, and what if Tesseract isn't installed?**
`pytesseract` calls the Tesseract engine on the cleaned scan to extract text. My
wrapper checks whether Tesseract is available first and returns a clear message
instead of crashing, so OCR is an optional bonus, never a hard dependency.

**38. Tell me about a real failure you found and how you fixed it.**
On a close-up photo of a **newspaper that filled the frame** (dense text, page
edges near or cut off at the border), the Hough fallback latched onto a thin
**sliver** of text columns, the warp produced a long thin strip, and the B&W
output was nearly all white — yet the UI claimed success. I fixed it by adding a
shared **quad-validation** step used by both detectors: a quad is rejected if it
covers < 20% of the frame, is non-convex, has an interior angle outside 45–135°,
or warps to an aspect ratio > 4:1. If nothing validates, the system falls back to
the full frame with `found = False` and **confidence 0**, and the UI shows a
*low-confidence* warning (never a green success) and asks the user to adjust the
corners. I also made preprocessing more robust to text by adding a **median blur**
before Canny and running detection at a **lower resolution**, so fine character
edges fade while the page border survives, and I added a **Rotate 90°** control
for sideways photos. A regression test now asserts that a sliver quad is rejected.

**39. Why a median blur specifically for the newspaper case (vs. Gaussian)?**
A median filter replaces each pixel with the median of its neighbourhood, so a
handful of dark text pixels are ignored entirely — it removes small high-contrast
speckle (letters) while keeping long continuous edges (the page border). A
Gaussian just smooths everything and leaves faint text edges that Canny can still
pick up. So median blur is the better tool for erasing dense text before edge
detection.

**40. What does your confidence score mean and how is it computed?**
It's a 0–1 estimate of how "page-like" the detected quad is, blending four
sub-scores: how much of the frame it fills, how close its interior angles are to
90°, how rectangular (non-stretched) it is, and whether it's convex. It drives the
UI: high confidence → green success; low confidence or a fallback → yellow warning
with manual adjustment. It turns a silent failure into an honest, actionable one.

**41. How do you detect a white page on a white (or light) table?**
Grayscale Canny fails because paper and desk have almost the same brightness. So
I detect edges on **multiple cues** and combine them: besides grayscale, I run
Canny on the **LAB a and b colour channels** and the **HSV saturation** channel,
plus a **morphological gradient**, and OR the results. Paper and desk almost
always differ in hue or saturation even when brightness matches, so the colour
channels reveal a boundary the grayscale channel cannot see.

**42. Why LAB and HSV channels specifically?**
LAB separates lightness (L) from the two colour axes (a = green–red, b =
blue–yellow); the a/b channels carry colour difference independent of brightness.
HSV's S (saturation) channel captures how "colourful" a pixel is. A faint cream
page on a white desk has near-identical L but different a/b and S, so edges show
up there.

**43. What is GrabCut and when do you use it?**
GrabCut is an interactive foreground-segmentation algorithm: you give it a
rectangle around the object, it models foreground and background colours with
Gaussian mixture models and refines a segmentation by iterated graph cuts
(min-cut/max-flow). I use it as a **fallback** when no edge-based quad scores well
— it separates the page from the background by colour statistics rather than
edges, then I fit a quad to the largest foreground region. It's the expensive
method, so it only runs on hard cases.

**44. You have several detection methods — how do you choose between them?**
I generate candidate quads from every method (contour on grayscale, contour on
the multi-cue edges, Hough, GrabCut), **validate** each (area, convexity, angles,
aspect), and **score** the survivors with a blend of area, rectangularity, **edge
support** (how much of the quad's outline lies on real edges) and **inside-vs-
outside contrast** (LAB colour difference across the border). The highest-scoring
valid candidate wins, and the UI shows which method produced it.

**45. What happens when the page runs off the edge of the photo?**
The scanner is **border-aware**: when the segmented page presses against the image
border, I snap those corners to the edge and flag the result as *clipped* with the
message "some corners were estimated", instead of discarding it. Honest caveat:
a corner that is outside the frame **cannot truly be recovered** — it isn't in the
image — so the clipped warp is approximate and the best fix is to re-shoot with
the whole page in frame. Manual corner clicking is the final fallback.

**46. Doesn't all this extra processing make it slow?**
Not for normal photos: the cheap contour method handles them in ~10 ms and the
GrabCut fallback is **skipped** when a candidate already scores well. GrabCut only
runs on genuinely hard images, which raises the *average* to ~70 ms on the test
set — still interactive. In the web app results are cached, so adjusting a slider
doesn't re-run detection.
