"""Tests for still-image (file) processing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_image(w: int = 640, h: int = 480) -> np.ndarray:
    """Create a simple BGR test image with a fake plate region."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # White plate background
    cv2.rectangle(frame, (200, 200), (440, 260), (255, 255, 255), -1)
    cv2.putText(frame, "AB12 CDE", (210, 248),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return frame


def _save_test_image(fmt: str = ".jpg") -> Path:
    """Save a test image to a temporary file and return its path."""
    frame = _make_test_image()
    tmp = tempfile.NamedTemporaryFile(suffix=fmt, delete=False)
    cv2.imwrite(tmp.name, frame)
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Pipeline: process_image (still file)
# ---------------------------------------------------------------------------

class TestPipelineImageFile:
    """Verify that the pipeline can process a still image loaded from disk."""

    def test_process_jpeg(self):
        from src.lpr.pipeline import LPRPipeline

        img_path = _save_test_image(".jpg")
        try:
            frame = cv2.imread(str(img_path))
            assert frame is not None, "cv2.imread returned None for a valid JPEG"

            with LPRPipeline() as pipeline:
                # process() accepts any BGR ndarray regardless of its origin
                detections = pipeline.process(frame)
                annotated = pipeline.annotate(frame, detections)

            assert annotated.shape == frame.shape
            assert isinstance(detections, list)
        finally:
            img_path.unlink(missing_ok=True)

    def test_process_png(self):
        from src.lpr.pipeline import LPRPipeline

        img_path = _save_test_image(".png")
        try:
            frame = cv2.imread(str(img_path))
            assert frame is not None

            with LPRPipeline() as pipeline:
                detections = pipeline.process(frame)

            assert isinstance(detections, list)
        finally:
            img_path.unlink(missing_ok=True)

    def test_annotate_still_image(self):
        """annotate() must not mutate the original frame."""
        from src.lpr.pipeline import LPRPipeline, PlateDetection

        frame = _make_test_image()
        original = frame.copy()
        det = PlateDetection(bbox=(200, 200, 240, 60), text="AB12 CDE", confidence=0.90)

        with LPRPipeline() as pipeline:
            annotated = pipeline.annotate(frame, [det])

        assert np.array_equal(frame, original), "annotate() must not modify the input frame"
        assert not np.array_equal(annotated, original), "annotated frame should differ"

    def test_invalid_file_returns_none(self, tmp_path):
        """cv2.imread should return None for a corrupt or non-image file."""
        bad_file = tmp_path / "not_an_image.jpg"
        bad_file.write_bytes(b"this is not image data")

        result = cv2.imread(str(bad_file))
        assert result is None


# ---------------------------------------------------------------------------
# Image format coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", [".jpg", ".png", ".bmp"])
def test_roundtrip_formats(fmt, tmp_path):
    """Write and re-read images in various formats; pipeline must accept them."""
    from src.lpr.pipeline import LPRPipeline

    frame = _make_test_image()
    img_path = tmp_path / f"plate{fmt}"
    cv2.imwrite(str(img_path), frame)

    reloaded = cv2.imread(str(img_path))
    assert reloaded is not None, f"Could not read back {fmt} image"

    with LPRPipeline() as pipeline:
        detections = pipeline.process(reloaded)

    assert isinstance(detections, list)
