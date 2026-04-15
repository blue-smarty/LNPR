"""RTSP / network stream source using OpenCV / GStreamer."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .base import CameraBase, CameraError

logger = logging.getLogger(__name__)


class RTSPCamera(CameraBase):
    """Capture from an RTSP (or any URL-addressable) stream.

    OpenCV's :class:`cv2.VideoCapture` is used with the ``FFMPEG`` back-end
    when available, falling back to the default back-end.

    Args:
        url: RTSP URL, e.g. ``rtsp://user:pass@192.168.1.10:554/stream1``.
        width: Requested frame width (best-effort; the stream may override).
        height: Requested frame height.
        fps: Not enforced for RTSP; kept for API consistency.
        latency: GStreamer ``rtspsrc`` latency in milliseconds (if using
            the GStreamer pipeline variant).
    """

    def __init__(
        self,
        url: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        latency: int = 200,
    ) -> None:
        super().__init__(width=width, height=height, fps=fps)
        self.url = url
        self.latency = latency
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------

    def _build_gst_pipeline(self) -> str:
        return (
            f"rtspsrc location={self.url} latency={self.latency} ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! appsink max-buffers=1 drop=true"
        )

    def open(self) -> None:
        logger.info("Opening RTSP stream: %s", self.url)
        # Try plain URL first (FFMPEG or OS default back-end)
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            # Fallback: GStreamer pipeline
            logger.debug("FFMPEG back-end failed, trying GStreamer pipeline")
            self._cap = cv2.VideoCapture(self._build_gst_pipeline(), cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise CameraError(f"Cannot open RTSP stream: {self.url}")
        self._opened = True
        logger.info("RTSP stream opened")

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False
        logger.info("RTSP stream released")
