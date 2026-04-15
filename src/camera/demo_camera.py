"""Demo / test camera that generates synthetic frames or plays a video file.

This source requires **no hardware** and is suitable for UI demonstrations,
CI testing, and development on non-Pi machines.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import CameraBase, CameraError

logger = logging.getLogger(__name__)

# Plate strings rendered in demo frames
_DEMO_PLATES = ["AB12 CDE", "XY34 FGH", "LM56 NOP", "QR78 STU", "VW90 XYZ"]


def _make_synthetic_frame(
    width: int, height: int, frame_idx: int
) -> np.ndarray:
    """Generate a synthetic BGR frame with a fake number plate overlay."""
    # Scrolling gradient background
    hue = int((frame_idx * 2) % 180)
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = 80
    hsv[:, :, 2] = 200
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Simulated licence plate
    plate_text = _DEMO_PLATES[frame_idx % len(_DEMO_PLATES)]
    plate_w, plate_h = 320, 80
    px = (width - plate_w) // 2
    py = (height - plate_h) // 2 + int(30 * np.sin(frame_idx / 15))
    py = max(0, min(py, height - plate_h))

    cv2.rectangle(frame, (px, py), (px + plate_w, py + plate_h), (0, 200, 255), -1)
    cv2.rectangle(frame, (px, py), (px + plate_w, py + plate_h), (0, 0, 0), 3)
    cv2.putText(
        frame,
        plate_text,
        (px + 20, py + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    # Frame counter / watermark
    cv2.putText(
        frame,
        f"DEMO  frame {frame_idx:05d}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


class DemoCamera(CameraBase):
    """Synthetic or video-file demo camera.

    Args:
        video_path: Path to a video file. When *None* synthetic frames are
            generated instead.
        width: Output frame width.
        height: Output frame height.
        fps: Target frame rate (throttled via :func:`time.sleep`).
        loop: Whether to loop a video file indefinitely.
    """

    def __init__(
        self,
        video_path: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        loop: bool = True,
    ) -> None:
        super().__init__(width=width, height=height, fps=fps)
        self.video_path = video_path
        self.loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx: int = 0
        self._last_time: float = 0.0
        self._synthetic = video_path is None

    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._synthetic:
            logger.info("DemoCamera: synthetic mode (%dx%d @ %d fps)", self.width, self.height, self.fps)
        else:
            path = Path(self.video_path)  # type: ignore[arg-type]
            if not path.exists():
                raise CameraError(f"Demo video not found: {path}")
            self._cap = cv2.VideoCapture(str(path))
            if not self._cap.isOpened():
                raise CameraError(f"Cannot open demo video: {path}")
            logger.info("DemoCamera: playing %s", path)
        self._frame_idx = 0
        self._last_time = time.monotonic()
        self._opened = True

    def read(self) -> Optional[np.ndarray]:
        # Throttle to requested FPS
        now = time.monotonic()
        elapsed = now - self._last_time
        delay = 1.0 / self.fps - elapsed
        if delay > 0:
            time.sleep(delay)
        self._last_time = time.monotonic()

        if self._synthetic:
            frame = _make_synthetic_frame(self.width, self.height, self._frame_idx)
            self._frame_idx += 1
            return frame

        # Video file mode
        assert self._cap is not None
        ret, frame = self._cap.read()
        if not ret:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return None
            else:
                return None
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self._frame_idx += 1
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False
        logger.info("DemoCamera released")
