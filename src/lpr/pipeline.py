"""Full LPR pipeline: detection → crop → recognition.

Architecture
------------
1. **Detection stage**: a YOLOv5-based licence-plate detector (Hailo HEF)
   returns bounding boxes for all plates in the frame.
2. **Recognition stage**: an LPRNet model (Hailo HEF) reads the text from
   each cropped plate region.

When the Hailo SDK is unavailable (mock mode) a simple OpenCV-based
heuristic is used to simulate plausible output for demo purposes.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..inference.hailo_inference import HailoInference

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PlateDetection:
    """A single detected licence plate.

    Attributes:
        bbox: Bounding box ``(x, y, w, h)`` in pixel coordinates.
        text: Recognised plate text (empty string if recognition failed).
        confidence: Detection confidence in ``[0, 1]``.
        timestamp: Unix timestamp when the detection was made.
    """
    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float
    timestamp: float = dataclasses.field(default_factory=time.time)

    def __str__(self) -> str:
        return f"{self.text or '???'} ({self.confidence:.0%})"


# ---------------------------------------------------------------------------
# LPD post-processing
# ---------------------------------------------------------------------------

_CHAR_LIST = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ")

def _decode_lprnet_output(output: np.ndarray) -> str:
    """Greedy CTC decode of LPRNet output tensor.

    Args:
        output: Shape ``(T, num_classes)`` float32 tensor.

    Returns:
        Decoded plate string.
    """
    indices = np.argmax(output, axis=-1)
    chars = []
    prev = -1
    for idx in indices:
        if idx != prev and idx < len(_CHAR_LIST):
            ch = _CHAR_LIST[idx]
            if ch != " ":
                chars.append(ch)
        prev = idx
    return "".join(chars)


def _parse_yolo_detections(
    raw: np.ndarray,
    frame_w: int,
    frame_h: int,
    conf_threshold: float = 0.40,
) -> List[Tuple[int, int, int, int, float]]:
    """Parse a flat YOLO output tensor into bounding boxes.

    Supports both ``(num_boxes, 5+)`` and flat layouts.
    Returns a list of ``(x, y, w, h, confidence)`` tuples in pixel space.
    """
    detections: List[Tuple[int, int, int, int, float]] = []
    if raw.ndim == 1:
        # Reshape assuming 6 values per box: [cx, cy, w, h, obj_conf, class_conf]
        if raw.size % 6 == 0:
            raw = raw.reshape(-1, 6)
        else:
            return detections

    for row in raw:
        if len(row) < 5:
            continue
        cx, cy, w, h = row[:4]
        obj_conf = float(row[4])
        if obj_conf < conf_threshold:
            continue
        # Clamp to [0, 1] – model outputs are typically normalised
        if cx > 1.0:
            cx /= frame_w; cy /= frame_h; w /= frame_w; h /= frame_h

        x = int((cx - w / 2) * frame_w)
        y = int((cy - h / 2) * frame_h)
        bw = int(w * frame_w)
        bh = int(h * frame_h)
        x = max(0, x); y = max(0, y)
        bw = min(bw, frame_w - x); bh = min(bh, frame_h - y)
        if bw > 10 and bh > 4:
            detections.append((x, y, bw, bh, obj_conf))

    return detections


# ---------------------------------------------------------------------------
# Demo / mock helpers
# ---------------------------------------------------------------------------

_MOCK_PLATES = [
    "AB12 CDE", "XY34 FGH", "LM56 NOP", "QR78 STU", "VW90 XYZ",
    "AA00 AAA", "ZZ99 ZZZ",
]
_mock_counter = 0


def _mock_detect(frame: np.ndarray) -> List[PlateDetection]:
    """Return a single fake detection for demo/mock mode."""
    global _mock_counter  # noqa: PLW0603
    h, w = frame.shape[:2]
    plate_w, plate_h = w // 4, h // 10
    x = w // 2 - plate_w // 2
    y = h // 2 - plate_h // 2
    text = _MOCK_PLATES[_mock_counter % len(_MOCK_PLATES)]
    _mock_counter += 1
    return [PlateDetection(bbox=(x, y, plate_w, plate_h), text=text, confidence=0.95)]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LPRPipeline:
    """End-to-end licence plate recognition pipeline.

    Args:
        lpd_hef: Path to the licence-plate **detection** HEF model.
        lpr_hef: Path to the licence-plate **recognition** (LPRNet) HEF model.
            Can be *None* to skip the recognition stage.
        conf_threshold: Minimum detection confidence.
        device_count: Number of Hailo devices (``VDevice`` multi-device).
    """

    def __init__(
        self,
        lpd_hef: Optional[str | Path] = None,
        lpr_hef: Optional[str | Path] = None,
        conf_threshold: float = 0.45,
        device_count: int = 1,
    ) -> None:
        self.conf_threshold = conf_threshold
        self._lpd = HailoInference(lpd_hef or "models/lpd_yolov5s.hef")
        self._lpr = HailoInference(lpr_hef or "models/lprnet.hef") if lpr_hef else None
        self._mock = self._lpd.is_mock

    # ------------------------------------------------------------------

    def open(self) -> None:
        self._lpd.open()
        if self._lpr:
            self._lpr.open()
        if self._lpd.is_mock:
            self._mock = True
            logger.info("LPRPipeline running in mock/demo mode")

    def close(self) -> None:
        self._lpd.close()
        if self._lpr:
            self._lpr.close()

    def __enter__(self) -> "LPRPipeline":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> List[PlateDetection]:
        """Run the full LPD → crop → LPR pipeline on *frame*.

        Args:
            frame: BGR image as a NumPy array.

        Returns:
            List of :class:`PlateDetection` instances (may be empty).
        """
        if self._mock:
            # In mock mode return a detection every ~30 calls to simulate
            # realistic detection rate without flooding the UI
            if not hasattr(self, "_mock_frame_count"):
                self._mock_frame_count = 0
            self._mock_frame_count += 1
            if self._mock_frame_count % 30 == 1:
                return _mock_detect(frame)
            return []

        h, w = frame.shape[:2]

        # --- Detection ---
        lpd_out = self._lpd.infer(frame)
        if not lpd_out:
            return []

        raw = next(iter(lpd_out.values()))
        boxes = _parse_yolo_detections(raw, w, h, self.conf_threshold)
        if not boxes:
            return []

        detections: List[PlateDetection] = []
        for (x, y, bw, bh, conf) in boxes:
            text = ""
            if self._lpr is not None:
                crop = frame[y: y + bh, x: x + bw]
                lpr_out = self._lpr.infer(crop)
                if lpr_out:
                    tensor = next(iter(lpr_out.values()))
                    text = _decode_lprnet_output(tensor)

            detections.append(
                PlateDetection(bbox=(x, y, bw, bh), text=text, confidence=conf)
            )

        return detections

    def annotate(self, frame: np.ndarray, detections: List[PlateDetection]) -> np.ndarray:
        """Draw bounding boxes and plate text on a copy of *frame*."""
        out = frame.copy()
        for det in detections:
            x, y, w, h = det.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 50), 2)
            label = f"{det.text}  {det.confidence:.0%}" if det.text else f"{det.confidence:.0%}"
            cv2.putText(
                out, label, (x, max(y - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 50), 2, cv2.LINE_AA,
            )
        return out
