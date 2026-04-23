"""GStreamer + HailoRT (hailonet) camera source."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .base import CameraBase, CameraError

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)

logger = logging.getLogger(__name__)


def _element_available(name: str) -> bool:
    return Gst.ElementFactory.find(name) is not None


class GstHailoCamera(CameraBase):
    """GStreamer camera source with Hailo hailonet inference.

    The pipeline performs live inference using the hailonet element and draws
    overlays inside GStreamer (hailooverlay when available).
    """

    def __init__(
        self,
        source: str,
        url: Optional[str] = None,
        device: Optional[int | str] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        lpd_hef: str = "models/lpd.hef",
        latency: int = 200,
        overlay: bool = True,
    ) -> None:
        super().__init__(width=width, height=height, fps=fps)
        self.source = source
        self.url = url
        self.device = device
        self.lpd_hef = lpd_hef
        self.latency = latency
        self.overlay = overlay
        self._pipeline: Optional[Gst.Element] = None
        self._appsink: Optional[Gst.Element] = None

    def _build_source(self) -> str:
        if self.source == "rtsp":
            if not self.url:
                raise CameraError("RTSP URL is required for GStreamer RTSP source.")
            return (
                f"rtspsrc location=\"{self.url}\" latency={self.latency} ! "
                "rtph264depay ! h264parse ! avdec_h264"
            )
        if self.source == "usb":
            device = self.device if self.device is not None else 0
            device_str = f"/dev/video{device}" if isinstance(device, int) else str(device)
            return f"v4l2src device={device_str}"
        if self.source == "picam":
            if _element_available("libcamerasrc"):
                return "libcamerasrc"
            return "v4l2src"
        raise CameraError(f"Unsupported GStreamer source: {self.source}")

    def _build_pipeline(self) -> str:
        if not _element_available("hailonet"):
            raise CameraError(
                "GStreamer hailonet element not found. Install the Hailo GStreamer plugin."
            )

        source = self._build_source()
        caps = (
            f"video/x-raw,format=RGB,width={self.width},height={self.height},"
            f"framerate={self.fps}/1"
        )

        hailo_chain = f"hailonet hef-path=\"{self.lpd_hef}\" batch-size=1 is-active=true"
        if _element_available("hailofilter"):
            hailo_chain += " ! hailofilter"
        if self.overlay and _element_available("hailooverlay"):
            hailo_chain += " ! hailooverlay"
        elif self.overlay:
            logger.warning("hailooverlay not available; continuing without overlay")

        pipeline = (
            f"{source} ! queue ! videoconvert ! videoscale ! videorate ! {caps} ! "
            f"{hailo_chain} ! videoconvert ! video/x-raw,format=RGB ! "
            "appsink name=appsink sync=false max-buffers=1 drop=true"
        )
        return pipeline

    def open(self) -> None:
        pipeline_desc = self._build_pipeline()
        logger.info("Starting GStreamer pipeline: %s", pipeline_desc)
        self._pipeline = Gst.parse_launch(pipeline_desc)
        if self._pipeline is None:
            raise CameraError("Failed to create GStreamer pipeline")
        self._appsink = self._pipeline.get_by_name("appsink")
        if self._appsink is None:
            raise CameraError("GStreamer appsink not found in pipeline")

        self._pipeline.set_state(Gst.State.PLAYING)
        self._opened = True

    def read(self) -> Optional[np.ndarray]:
        if self._appsink is None:
            return None
        sample = self._appsink.emit("try-pull-sample", int(0.2 * Gst.SECOND))
        if sample is None:
            return None

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        if buffer is None or caps is None:
            return None

        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        fmt = structure.get_value("format") or "RGB"

        success, mapinfo = buffer.map(Gst.MapFlags.READ)
        if not success:
            return None
        try:
            channels = 3
            if fmt in {"RGBx", "BGRx"}:
                channels = 4
            data = np.frombuffer(mapinfo.data, dtype=np.uint8)
            frame = data.reshape((height, width, channels))
            frame = frame[:, :, :3]
            if fmt.startswith("RGB"):
                frame = frame[:, :, ::-1]
            return frame.copy()
        finally:
            buffer.unmap(mapinfo)

    def release(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            self._appsink = None
        self._opened = False
        logger.info("GStreamer pipeline released")
