"""
Automatic Document Scanner - classical computer-vision pipeline.

This package turns a photograph of a document into a flat, cropped, enhanced
"scan" using only classical OpenCV techniques (no deep learning). Each step is
deliberately kept small and explainable so the full pipeline can be reasoned
about and defended:

    preprocess -> detect -> transform (warp) -> enhance -> (ocr) -> export

Public helpers are re-exported here so callers can simply do, e.g.:

    from scanner import scan_image
"""

from .pipeline import scan_image, PipelineResult

__all__ = ["scan_image", "PipelineResult"]

__version__ = "1.0.0"
