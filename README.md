# Automatic Document Scanner using Computer Vision

Turns a photo of a document into a flat, clean scan (PDF or image) using OpenCV — detects the page edges, corrects perspective, and cleans up the result.

## Setup
```
pip install -r requirements.txt
```

## Use
```
python scanner.py photo.jpg                # -> photo_scan.pdf
python scanner.py photo.jpg -o scan.png    # image output
python scanner.py a.jpg b.jpg -o doc.pdf   # multi-page PDF
python scanner.py photo.jpg --mode color   # bw (default), gray, color
python scanner.py --camera                 # webcam: SPACE capture, ENTER save, Q quit
```

Tip: photograph the page on a dark, plain surface so the edges are easy to detect.
