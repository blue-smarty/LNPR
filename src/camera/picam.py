"""Raspberry Pi Camera source using picamera2."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .base import CameraBase, CameraError

logger = logging.getLogger(__name__)


class PiCamera2Camera(CameraBase):
    """Capture from the Raspberry Pi camera module via *picamera2*.

    *picamera2* must be installed (``sudo apt install python3-picamera2``).
    Falls back gracefully with :class:`.CameraError` when not available.

    Args:
        camera_num: Index of the camera (0 for the first module).
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Target frame rate.
    """

    def __init__(
        self,
        camera_num: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        super().__init__(width=width, height=height, fps=fps)
        self.camera_num = camera_num
        self._cam = None

    # ------------------------------------------------------------------

    def open(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise CameraError(
                "picamera2 is not installed. Run: sudo apt install python3-picamera2"
            ) from exc

        logger.info("Opening PiCamera2 (camera %d)", self.camera_num)
        self._cam = Picamera2(self.camera_num)
        config = self._cam.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": float(self.fps)},
        )
        self._cam.configure(config)
        self._cam.start()
        self._opened = True
        logger.info("PiCamera2 started")

    def read(self) -> Optional[np.ndarray]:
        if self._cam is None:
            return None
        import cv2  # local import to avoid cost when not needed
        frame_rgb = self._cam.capture_array()
        # picamera2 returns RGB; convert to BGR for OpenCV / rest of pipeline
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def release(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam.close()
            self._cam = None
        self._opened = False
        logger.info("PiCamera2 released")
