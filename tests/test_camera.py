"""Tests for the camera abstraction layer."""

from __future__ import annotations

import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# DemoCamera – runs without hardware
# ---------------------------------------------------------------------------

class TestDemoCamera:
    def test_open_and_read_synthetic(self):
        from src.camera.demo_camera import DemoCamera

        cam = DemoCamera(width=640, height=480, fps=60)
        cam.open()
        assert cam.is_open

        frame = cam.read()
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8

        cam.release()
        assert not cam.is_open

    def test_context_manager(self):
        from src.camera.demo_camera import DemoCamera

        with DemoCamera(width=320, height=240, fps=60) as cam:
            assert cam.is_open
            frame = cam.read()
            assert frame is not None
        assert not cam.is_open

    def test_multiple_frames(self):
        from src.camera.demo_camera import DemoCamera

        with DemoCamera(width=320, height=240, fps=120) as cam:
            frames = [cam.read() for _ in range(5)]

        assert all(f is not None for f in frames)
        # Frames should differ (animated background)
        assert not np.array_equal(frames[0], frames[-1])

    def test_video_file_not_found(self):
        from src.camera.demo_camera import DemoCamera
        from src.camera.base import CameraError

        cam = DemoCamera(video_path="/nonexistent/video.mp4")
        with pytest.raises(CameraError):
            cam.open()

    def test_repr(self):
        from src.camera.demo_camera import DemoCamera

        cam = DemoCamera(width=800, height=600, fps=25)
        assert "800" in repr(cam)
        assert "600" in repr(cam)


# ---------------------------------------------------------------------------
# CameraBase abstract interface
# ---------------------------------------------------------------------------

class TestCameraBase:
    def test_cannot_instantiate_abstract(self):
        from src.camera.base import CameraBase

        with pytest.raises(TypeError):
            CameraBase()  # type: ignore[abstract]

    def test_concrete_subclass(self):
        from src.camera.base import CameraBase

        class _Concrete(CameraBase):
            def open(self): self._opened = True
            def read(self): return np.zeros((10, 10, 3), dtype=np.uint8)
            def release(self): self._opened = False

        cam = _Concrete(width=100, height=50, fps=15)
        assert cam.width == 100
        assert not cam.is_open
        cam.open()
        assert cam.is_open
        frame = cam.read()
        assert frame.shape == (10, 10, 3)
        cam.release()
        assert not cam.is_open


class TestRTSPHelpers:
    def test_parse_rtsp_urls_comma_and_newline(self):
        from src.camera.rtsp_camera import parse_rtsp_urls

        urls = (
            "rtsp://cam-a/stream1, rtsp://cam-b/stream1\n"
            "rtsp://cam-c/stream1"
        )
        parsed = parse_rtsp_urls(urls)
        assert parsed == [
            "rtsp://cam-a/stream1",
            "rtsp://cam-b/stream1",
            "rtsp://cam-c/stream1",
        ]

    def test_parse_rtsp_urls_empty(self):
        from src.camera.rtsp_camera import parse_rtsp_urls

        assert parse_rtsp_urls("") == []
