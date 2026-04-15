"""Camera abstraction package."""
from .base import CameraBase, CameraError
from .usb_camera import USBCamera
from .rtsp_camera import RTSPCamera
from .demo_camera import DemoCamera

__all__ = ["CameraBase", "CameraError", "USBCamera", "RTSPCamera", "DemoCamera"]

try:
    from .picam import PiCamera2Camera
    __all__.append("PiCamera2Camera")
except ImportError:
    pass
