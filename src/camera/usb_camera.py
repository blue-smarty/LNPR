"""USB / V4L2 camera source using OpenCV."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .base import CameraBase, CameraError

logger = logging.getLogger(__name__)


class USBCamera(CameraBase):
    """Capture from a USB or V4L2 camera device.

    Args:
        device_index: Integer index (e.g. 0 for ``/dev/video0``) or a device
            path string such as ``/dev/video2``.
        width: Requested frame width in pixels.
        height: Requested frame height in pixels.
        fps: Requested frames per second.
    """

    def __init__(
        self,
        device_index: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        super().__init__(width=width, height=height, fps=fps)
        self.device_index = device_index
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------

    def open(self) -> None:
        logger.info("Opening USB camera: %s", self.device_index)
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise CameraError(f"Cannot open USB camera: {self.device_index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._opened = True
        logger.info("USB camera opened (actual: %dx%d @ %.1f fps)",
                    int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    self._cap.get(cv2.CAP_PROP_FPS))

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
        logger.info("USB camera released")

    def get_property(self, name: str):  # noqa: ANN201
        prop = getattr(cv2, f"CAP_PROP_{name.upper()}", None)
        if prop is None or self._cap is None:
            raise ValueError(f"Unknown property: {name}")
        return self._cap.get(prop)

    def set_property(self, name: str, value) -> None:  # noqa: ANN001
        prop = getattr(cv2, f"CAP_PROP_{name.upper()}", None)
        if prop is None or self._cap is None:
            raise ValueError(f"Unknown property: {name}")
        self._cap.set(prop, value)
