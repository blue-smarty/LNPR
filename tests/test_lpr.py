"""Tests for the LPR pipeline (mock / demo mode)."""

from __future__ import annotations

import numpy as np
import pytest


class TestPlateDetection:
    def test_str_with_text(self):
        from src.lpr.pipeline import PlateDetection

        det = PlateDetection(bbox=(10, 20, 200, 60), text="AB12 CDE", confidence=0.92)
        s = str(det)
        assert "AB12 CDE" in s
        assert "92%" in s

    def test_str_without_text(self):
        from src.lpr.pipeline import PlateDetection

        det = PlateDetection(bbox=(0, 0, 100, 30), text="", confidence=0.75)
        assert "???" in str(det)


class TestLPRPipeline:
    """Tests that work in mock/demo mode (no Hailo hardware required)."""

    def _make_frame(self, w=640, h=480):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_open_close_mock(self):
        from src.lpr.pipeline import LPRPipeline

        pipeline = LPRPipeline()
        pipeline.open()
        assert pipeline._mock  # should be mock when no HEF is present
        pipeline.close()

    def test_process_returns_list(self):
        from src.lpr.pipeline import LPRPipeline

        with LPRPipeline() as pipeline:
            result = pipeline.process(self._make_frame())
        assert isinstance(result, list)

    def test_annotate_does_not_crash(self):
        from src.lpr.pipeline import LPRPipeline, PlateDetection

        frame = self._make_frame()
        detections = [PlateDetection(bbox=(10, 10, 100, 30), text="XY99 ZZZ", confidence=0.88)]
        with LPRPipeline() as pipeline:
            annotated = pipeline.annotate(frame, detections)
        assert annotated.shape == frame.shape

    def test_mock_periodic_detection(self):
        from src.lpr.pipeline import LPRPipeline

        frame = self._make_frame()
        with LPRPipeline() as pipeline:
            # In mock mode a detection is returned every ~30 frames
            results_over_60 = []
            for _ in range(60):
                results_over_60.extend(pipeline.process(frame))

        # Should have gotten at least one detection in 60 frames
        assert len(results_over_60) >= 1


class TestYoloPostprocess:
    def test_parse_valid_boxes(self):
        from src.lpr.pipeline import _parse_yolo_detections

        # 3 boxes: (cx, cy, w, h, conf)
        raw = np.array([
            [0.5, 0.5, 0.3, 0.1, 0.9, 0.0],
            [0.1, 0.1, 0.05, 0.05, 0.2, 0.0],   # low conf – should be filtered
            [0.7, 0.7, 0.2, 0.15, 0.85, 0.0],
        ], dtype=np.float32)

        boxes = _parse_yolo_detections(raw, frame_w=640, frame_h=480, conf_threshold=0.40)
        assert len(boxes) == 2
        for x, y, w, h, conf in boxes:
            assert conf >= 0.40
            assert w > 0 and h > 0

    def test_empty_input(self):
        from src.lpr.pipeline import _parse_yolo_detections

        boxes = _parse_yolo_detections(np.zeros((0, 6), dtype=np.float32), 640, 480)
        assert boxes == []


class TestLPRNetDecoder:
    def test_decode_simple(self):
        from src.lpr.pipeline import _decode_lprnet_output, _CHAR_LIST

        # Build a fake output: one-hot encoding of "AB1"
        def char_idx(c):
            return _CHAR_LIST.index(c)

        T = 3
        num_classes = len(_CHAR_LIST)
        output = np.zeros((T, num_classes), dtype=np.float32)
        output[0, char_idx("A")] = 1.0
        output[1, char_idx("B")] = 1.0
        output[2, char_idx("1")] = 1.0

        result = _decode_lprnet_output(output)
        assert result == "AB1"

    def test_decode_with_blanks(self):
        from src.lpr.pipeline import _decode_lprnet_output, _CHAR_LIST

        # Space character acts as CTC blank
        num_classes = len(_CHAR_LIST)
        output = np.zeros((4, num_classes), dtype=np.float32)
        output[0, _CHAR_LIST.index("A")] = 1.0
        output[1, _CHAR_LIST.index(" ")] = 1.0   # blank
        output[2, _CHAR_LIST.index("B")] = 1.0
        output[3, _CHAR_LIST.index(" ")] = 1.0   # blank

        result = _decode_lprnet_output(output)
        assert result == "AB"
