"""Hailo-8 inference wrapper.

This module wraps the Hailo Runtime Python API (``hailo_platform``) to provide
a simple synchronous interface for running HEF (Hailo Executable Format) models.

When the Hailo SDK is not installed the wrapper falls back to a **mock** mode
that returns empty results, allowing the rest of the application (including the
UI) to function without hardware.

References:
  - Hailo Developer Zone: https://hailo.ai/developer-zone/
  - hailo-ai/hailo-apps: https://github.com/hailo-ai/hailo-apps
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SDK import
# ---------------------------------------------------------------------------

try:
    from hailo_platform import (  # type: ignore
        HEF,
        ConfigureParams,
        FormatType,
        HailoSchedulingAlgorithm,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )
    _HAILO_AVAILABLE = True
except ImportError:
    _HAILO_AVAILABLE = False
    logger.warning(
        "hailo_platform not found – running in mock/demo mode. "
        "Install the Hailo SDK to enable Hailo-8 inference."
    )


class HailoInferenceError(RuntimeError):
    """Raised on Hailo device or model errors."""


class HailoInference:
    """High-level wrapper around the Hailo Runtime inference API.

    Usage::

        with HailoInference("models/lpd_yolov5s.hef") as hailo:
            results = hailo.infer(frame_bgr)

    Args:
        hef_path: Path to the compiled ``.hef`` model file.
        input_format: Hailo input format type (default ``UINT8``).
        batch_size: Inference batch size (default 1).
    """

    def __init__(
        self,
        hef_path: str | Path,
        input_format: str = "UINT8",
        batch_size: int = 1,
    ) -> None:
        self.hef_path = Path(hef_path)
        self.input_format = input_format
        self.batch_size = batch_size

        self._device = None
        self._network_group = None
        self._input_vstream_params = None
        self._output_vstream_params = None
        self._lock = threading.Lock()
        self._mock = not _HAILO_AVAILABLE

        if not self.hef_path.exists() and _HAILO_AVAILABLE:
            logger.warning("HEF not found at %s – falling back to mock mode", self.hef_path)
            self._mock = True

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "HailoInference":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the Hailo device and load the HEF model."""
        if self._mock:
            logger.info("HailoInference: mock mode active")
            return

        logger.info("Loading HEF: %s", self.hef_path)
        try:
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            self._device = VDevice(params=params)

            hef = HEF(str(self.hef_path))
            configure_params = ConfigureParams.create_from_hef(
                hef=hef, interface=HailoStreamInterface.PCIe
            )
            network_groups = self._device.configure(hef, configure_params)
            if not network_groups:
                raise HailoInferenceError("No network groups found in HEF")
            self._network_group = network_groups[0]

            fmt = getattr(FormatType, self.input_format)
            self._input_vstream_params = InputVStreamParams.make(
                self._network_group, format_type=fmt
            )
            self._output_vstream_params = OutputVStreamParams.make(
                self._network_group, format_type=FormatType.FLOAT32
            )
            logger.info("Hailo-8 model loaded successfully")
        except Exception as exc:
            raise HailoInferenceError(f"Failed to load HEF: {exc}") from exc

    def close(self) -> None:
        """Release the Hailo device."""
        if self._device is not None:
            self._device = None
            logger.info("Hailo device released")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on a single BGR frame.

        Args:
            frame: BGR image as a NumPy array ``(H, W, 3)``.

        Returns:
            Dictionary mapping output layer name → output tensor (float32).
            Returns an empty dict in mock mode.
        """
        if self._mock:
            return {}

        # Resize to model input dimensions
        input_shape = self._get_input_shape()
        if input_shape:
            h, w = input_shape[:2]
            if frame.shape[:2] != (h, w):
                import cv2
                frame = cv2.resize(frame, (w, h))

        batch = frame[np.newaxis, ...]  # (1, H, W, 3)

        with self._lock:
            try:
                with InferVStreams(
                    self._network_group,
                    self._input_vstream_params,
                    self._output_vstream_params,
                ) as pipeline:
                    input_data = {
                        list(pipeline.get_input_vstreams())[0].name: batch
                    }
                    with self._network_group.activate():
                        raw_results = pipeline.infer(input_data)
            except Exception as exc:
                logger.error("Inference failed: %s", exc)
                return {}

        return {k: v[0] for k, v in raw_results.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_input_shape(self) -> Optional[Tuple[int, ...]]:
        if self._input_vstream_params is None:
            return None
        try:
            params = next(iter(self._input_vstream_params))
            return params.shape
        except (StopIteration, AttributeError):
            return None

    @property
    def is_mock(self) -> bool:
        """True if running in mock/demo mode without real Hailo hardware."""
        return self._mock
