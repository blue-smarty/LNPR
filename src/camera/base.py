"""Abstract base class for all camera sources."""

from __future__ import annotations

import abc
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Raised on camera configuration or runtime errors."""


class CameraBase(abc.ABC):
    """Unified interface for RTSP, USB, PiCam and demo sources.

    Subclasses must implement :meth:`open`, :meth:`read`, and :meth:`release`.
    The optional :meth:`get_property` / :meth:`set_property` methods allow
    source-specific tuning without breaking the abstraction.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._opened = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def open(self) -> None:
        """Open / start the camera source.

        Raises:
            CameraError: if the source cannot be opened.
        """

    @abc.abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame as a NumPy array, or *None* on EOF/error."""

    @abc.abstractmethod
    def release(self) -> None:
        """Stop capture and free resources."""

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraBase":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._opened

    def get_property(self, name: str):  # noqa: ANN201
        raise NotImplementedError(f"{self.__class__.__name__} does not support get_property")

    def set_property(self, name: str, value) -> None:  # noqa: ANN001
        raise NotImplementedError(f"{self.__class__.__name__} does not support set_property")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(width={self.width}, height={self.height}, fps={self.fps})"
