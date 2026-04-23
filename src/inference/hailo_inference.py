"""Hailo-8 inference wrapper.

This module wraps the Hailo Runtime Python API (``hailo_platform``) to provide
an async-backed interface for running HEF (Hailo Executable Format) models.

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
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SDK import
# ---------------------------------------------------------------------------

try:
    from hailo_platform import (  # type: ignore
        VDevice,
        FormatType,
        HailoSchedulingAlgorithm,
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

        with HailoInference("models/lpd.hef") as hailo:
            results = hailo.infer(frame_bgr)

    Args:
        hef_path: Path to the compiled ``.hef`` model file.
        input_format: Hailo input format type (default ``UINT8``).
        batch_size: Inference batch size (default 1).
        timeout_ms: Async inference timeout in milliseconds.
    """

    _shared_device: Optional[VDevice] = None
    _shared_refs = 0
    _shared_lock = threading.Lock()

    def __init__(
        self,
        hef_path: str | Path,
        input_format: str = "UINT8",
        batch_size: int = 1,
        timeout_ms: int = 1000,
    ) -> None:
        self.hef_path = Path(hef_path)
        self.input_format = input_format
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms

        self._device: Optional[VDevice] = None
        self._infer_model = None
        self._configured_infer_model = None
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
            with self._shared_lock:
                if self.__class__._shared_device is None:
                    params = VDevice.create_params()
                    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
                    self.__class__._shared_device = VDevice(params=params)
                self.__class__._shared_refs += 1
                self._device = self.__class__._shared_device

            if self._device is None:
                raise HailoInferenceError("Failed to acquire Hailo VDevice")

            self._infer_model = self._device.create_infer_model(str(self.hef_path))
            if hasattr(self._infer_model, "set_batch_size"):
                self._infer_model.set_batch_size(self.batch_size)

            fmt = getattr(FormatType, self.input_format)
            try:
                self._infer_model.input().set_format_type(fmt)
            except Exception:  # noqa: BLE001
                pass

            for output in self._get_outputs():
                try:
                    output.set_format_type(FormatType.FLOAT32)
                except Exception:  # noqa: BLE001
                    pass

            self._configured_infer_model = self._infer_model.configure()
            logger.info("Hailo-8 model loaded successfully")
        except Exception as exc:
            raise HailoInferenceError(f"Failed to load HEF: {exc}") from exc

    def close(self) -> None:
        """Release the Hailo device."""
        self._configured_infer_model = None
        self._infer_model = None

        if self._device is not None:
            with self._shared_lock:
                self.__class__._shared_refs = max(0, self.__class__._shared_refs - 1)
                if self.__class__._shared_refs == 0 and self.__class__._shared_device is not None:
                    try:
                        self.__class__._shared_device.release()  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001
                        pass
                    self.__class__._shared_device = None
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

        if self._configured_infer_model is None or self._infer_model is None:
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
                bindings = self._configured_infer_model.create_bindings()
                try:
                    bindings.input().set_buffer(batch)
                except Exception:  # noqa: BLE001
                    bindings.input(self._infer_model.input().name).set_buffer(batch)

                output_buffers: Dict[str, np.ndarray] = {}
                for output in self._get_outputs():
                    out_name = getattr(output, "name", "output")
                    buffer = np.empty(output.shape, dtype=np.float32)
                    output_buffers[out_name] = buffer
                    self._set_output_buffer(bindings, output, buffer)

                self._configured_infer_model.wait_for_async_ready(timeout_ms=self.timeout_ms)
                job = self._configured_infer_model.run_async([bindings])
                job.wait(self.timeout_ms)

                results: Dict[str, np.ndarray] = {}
                for output in self._get_outputs():
                    out_name = getattr(output, "name", "output")
                    results[out_name] = self._get_output_buffer(bindings, output)
                return results
            except Exception as exc:
                logger.error("Inference failed: %s", exc)
                return {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_outputs(self):
        if self._infer_model is None:
            return []
        if hasattr(self._infer_model, "outputs"):
            return list(self._infer_model.outputs())
        if hasattr(self._infer_model, "output"):
            return [self._infer_model.output()]
        return []

    def _set_output_buffer(self, bindings, output, buffer: np.ndarray) -> None:  # noqa: ANN001
        try:
            bindings.output(output.name).set_buffer(buffer)
        except Exception:  # noqa: BLE001
            bindings.output().set_buffer(buffer)

    def _get_output_buffer(self, bindings, output) -> np.ndarray:  # noqa: ANN001
        try:
            return bindings.output(output.name).get_buffer()
        except Exception:  # noqa: BLE001
            return bindings.output().get_buffer()

    def _get_input_shape(self) -> Optional[Tuple[int, ...]]:
        if self._infer_model is None:
            return None
        try:
            return tuple(self._infer_model.input().shape)
        except Exception:
            return None

    @property
    def is_mock(self) -> bool:
        """True if running in mock/demo mode without real Hailo hardware."""
        return self._mock
