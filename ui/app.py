"""LNPR GTK application – main entry point for the UI layer.

The application is structured as a single :class:`Gtk.ApplicationWindow`
with three areas:

* **Toolbar**: source selector, Start / Stop button, Open Image button,
  Settings button.
* **Preview pane**: live video feed rendered via a ``Gtk.DrawingArea``.
* **Detections panel**: scrollable list of recent plate detections.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("GdkPixbuf", "2.0")

from gi.repository import GdkPixbuf, GLib, Gtk  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.camera.base import CameraBase, CameraError
from src.camera.demo_camera import DemoCamera
from src.camera.rtsp_camera import RTSPCamera
from src.camera.usb_camera import USBCamera
from src.lpr.pipeline import LPRPipeline, PlateDetection

logger = logging.getLogger(__name__)

# Try PiCam – may not be available on non-Pi hardware
try:
    from src.camera.picam import PiCamera2Camera
    _PICAM_OK = True
except ImportError:
    _PICAM_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_ID = "ai.hailo.lnpr"
MAX_DETECTIONS = 50        # keep last N detections in the list
PREVIEW_FPS = 25           # target UI refresh rate
FRAME_Q_MAXSIZE = 2        # drop frames rather than accumulate


# ---------------------------------------------------------------------------
# Helper: BGR ndarray → GdkPixbuf
# ---------------------------------------------------------------------------

def _bgr_to_pixbuf(frame: np.ndarray) -> GdkPixbuf.Pixbuf:
    """Convert an OpenCV BGR frame to a ``GdkPixbuf``."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return GdkPixbuf.Pixbuf.new_from_data(
        rgb.tobytes(),
        GdkPixbuf.Colorspace.RGB,
        False,
        8,
        w, h,
        w * ch,
    )


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

class SettingsDialog(Gtk.Dialog):
    """Minimal settings dialog for camera parameters."""

    def __init__(self, parent: Gtk.Window, settings: dict) -> None:
        super().__init__(
            title="Settings",
            parent=parent,
            flags=Gtk.DialogFlags.MODAL | Gtk.DialogFlags.DESTROY_WITH_PARENT,
        )
        self.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OK, Gtk.ResponseType.OK,
        )
        self.set_default_size(420, 300)
        self._settings = dict(settings)

        grid = Gtk.Grid(
            column_spacing=12,
            row_spacing=8,
            margin_top=12,
            margin_bottom=12,
            margin_start=12,
            margin_end=12,
        )
        box = self.get_content_area()
        box.add(grid)

        # --- Width ---
        grid.attach(Gtk.Label(label="Frame width:", xalign=0), 0, 0, 1, 1)
        self._width_spin = Gtk.SpinButton.new_with_range(320, 3840, 16)
        self._width_spin.set_value(settings.get("width", 1280))
        grid.attach(self._width_spin, 1, 0, 1, 1)

        # --- Height ---
        grid.attach(Gtk.Label(label="Frame height:", xalign=0), 0, 1, 1, 1)
        self._height_spin = Gtk.SpinButton.new_with_range(240, 2160, 16)
        self._height_spin.set_value(settings.get("height", 720))
        grid.attach(self._height_spin, 1, 1, 1, 1)

        # --- FPS ---
        grid.attach(Gtk.Label(label="FPS:", xalign=0), 0, 2, 1, 1)
        self._fps_spin = Gtk.SpinButton.new_with_range(1, 120, 1)
        self._fps_spin.set_value(settings.get("fps", 30))
        grid.attach(self._fps_spin, 1, 2, 1, 1)

        # --- Confidence threshold ---
        grid.attach(Gtk.Label(label="Detection threshold:", xalign=0), 0, 3, 1, 1)
        self._conf_spin = Gtk.SpinButton.new_with_range(0.1, 1.0, 0.05)
        self._conf_spin.set_digits(2)
        self._conf_spin.set_value(settings.get("conf_threshold", 0.45))
        grid.attach(self._conf_spin, 1, 3, 1, 1)

        # --- RTSP URL ---
        grid.attach(Gtk.Label(label="RTSP URL:", xalign=0), 0, 4, 1, 1)
        self._rtsp_entry = Gtk.Entry()
        self._rtsp_entry.set_text(settings.get("rtsp_url", "rtsp://"))
        self._rtsp_entry.set_hexpand(True)
        self._rtsp_entry.set_tooltip_text(
            "Format:  rtsp://[user:password@]<host>[:<port>]/<path>\n"
            "Examples:\n"
            "  rtsp://192.168.1.64/stream1\n"
            "  rtsp://admin:secret@192.168.1.64:554/h264Preview_01_main\n"
            "  rtsp://192.168.1.64:8554/live\n"
            "  rtsp://admin:pass@192.168.1.64:554/Streaming/Channels/101  (Hikvision)\n"
            "  rtsp://admin:pass@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0  (Dahua)"
        )
        grid.attach(self._rtsp_entry, 1, 4, 1, 1)

        rtsp_hint = Gtk.Label()
        rtsp_hint.set_markup(
            "<small><i>Format: rtsp://[user:pass@]host[:port]/path  "
            "(hover for examples)</i></small>"
        )
        rtsp_hint.set_xalign(0)
        grid.attach(rtsp_hint, 0, 5, 2, 1)

        # --- USB device index ---
        grid.attach(Gtk.Label(label="USB device index:", xalign=0), 0, 6, 1, 1)
        self._usb_spin = Gtk.SpinButton.new_with_range(0, 10, 1)
        self._usb_spin.set_value(settings.get("usb_device", 0))
        grid.attach(self._usb_spin, 1, 6, 1, 1)

        self.show_all()

    def get_settings(self) -> dict:
        return {
            "width": int(self._width_spin.get_value()),
            "height": int(self._height_spin.get_value()),
            "fps": int(self._fps_spin.get_value()),
            "conf_threshold": self._conf_spin.get_value(),
            "rtsp_url": self._rtsp_entry.get_text(),
            "usb_device": int(self._usb_spin.get_value()),
        }


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class LNPRWindow(Gtk.ApplicationWindow):
    """Main application window."""

    def __init__(self, app: Gtk.Application) -> None:
        super().__init__(application=app, title="LNPR – Licence Number Plate Recognition")
        self.set_default_size(1100, 700)

        # Application state
        self._camera: Optional[CameraBase] = None
        self._pipeline: Optional[LPRPipeline] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=FRAME_Q_MAXSIZE)
        self._detection_count = 0

        self._settings = {
            "width": 1280,
            "height": 720,
            "fps": 30,
            "conf_threshold": 0.45,
            "rtsp_url": "rtsp://",
            "usb_device": 0,
        }

        self._build_ui()

        # Periodic UI update
        GLib.timeout_add(1000 // PREVIEW_FPS, self._on_frame_tick)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(outer)

        # ── Toolbar ──────────────────────────────────────────────────────
        toolbar = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=8,
            margin_top=6,
            margin_bottom=6,
            margin_start=8,
            margin_end=8,
        )
        outer.pack_start(toolbar, False, False, 0)

        # Source selector
        toolbar.pack_start(Gtk.Label(label="Source:"), False, False, 0)
        self._source_combo = Gtk.ComboBoxText()
        self._source_combo.append("demo", "Demo (no hardware)")
        self._source_combo.append("usb", "USB Camera")
        if _PICAM_OK:
            self._source_combo.append("picam", "PiCamera2")
        self._source_combo.append("rtsp", "RTSP Stream")
        self._source_combo.set_active(0)
        toolbar.pack_start(self._source_combo, False, False, 0)

        # Start / Stop
        self._start_btn = Gtk.Button(label="▶  Start")
        self._start_btn.get_style_context().add_class("suggested-action")
        self._start_btn.connect("clicked", self._on_start_stop)
        toolbar.pack_start(self._start_btn, False, False, 0)

        # Open image for still-picture recognition
        open_img_btn = Gtk.Button(label="📂  Open Image")
        open_img_btn.connect("clicked", self._on_open_image)
        toolbar.pack_start(open_img_btn, False, False, 0)

        # Clear detections
        clear_btn = Gtk.Button(label="🗑  Clear")
        clear_btn.connect("clicked", self._on_clear)
        toolbar.pack_start(clear_btn, False, False, 0)

        # Settings
        settings_btn = Gtk.Button(label="⚙  Settings")
        settings_btn.connect("clicked", self._on_settings)
        toolbar.pack_end(settings_btn, False, False, 0)

        # Separator
        outer.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 0)

        # ── Content area ─────────────────────────────────────────────────
        content = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        outer.pack_start(content, True, True, 0)

        # ── Preview ──────────────────────────────────────────────────────
        preview_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        content.pack_start(preview_box, True, True, 0)

        self._preview = Gtk.Image()
        self._preview.set_vexpand(True)
        self._preview.set_hexpand(True)
        preview_scroll = Gtk.ScrolledWindow()
        preview_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        preview_scroll.add(self._preview)
        preview_box.pack_start(preview_scroll, True, True, 0)

        # Status bar
        self._status_bar = Gtk.Statusbar()
        self._status_ctx = self._status_bar.get_context_id("main")
        self._status_bar.push(self._status_ctx, "Ready.  Select a source and press Start.")
        outer.pack_end(self._status_bar, False, False, 0)

        # ── Detections panel ─────────────────────────────────────────────
        panel_width = 280
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        panel.set_size_request(panel_width, -1)
        content.pack_end(panel, False, False, 0)

        panel.pack_start(
            Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 0
        )
        hdr = Gtk.Label()
        hdr.set_markup("<b>Recent Detections</b>")
        hdr.set_margin_top(8)
        panel.pack_start(hdr, False, False, 0)

        self._det_list = Gtk.ListStore(str, str, str)  # text, confidence, time
        tree = Gtk.TreeView(model=self._det_list)
        tree.set_headers_visible(True)

        for idx, (col_title, col_width) in enumerate(
            [("Plate", 130), ("Conf.", 55), ("Time", 70)]
        ):
            renderer = Gtk.CellRendererText()
            col = Gtk.TreeViewColumn(col_title, renderer, text=idx)
            col.set_fixed_width(col_width)
            col.set_resizable(True)
            tree.append_column(col)

        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        sw.add(tree)
        sw.set_vexpand(True)
        panel.pack_start(sw, True, True, 0)

        self._det_count_lbl = Gtk.Label(label="0 detections")
        self._det_count_lbl.set_margin_bottom(4)
        panel.pack_start(self._det_count_lbl, False, False, 0)

        self.show_all()

    # ------------------------------------------------------------------
    # Camera / pipeline factory
    # ------------------------------------------------------------------

    def _make_camera(self) -> CameraBase:
        source = self._source_combo.get_active_id()
        w = self._settings["width"]
        h = self._settings["height"]
        fps = self._settings["fps"]

        if source == "usb":
            return USBCamera(device_index=self._settings["usb_device"], width=w, height=h, fps=fps)
        if source == "rtsp":
            url = self._settings["rtsp_url"]
            if not url or url == "rtsp://":
                raise CameraError("Please set an RTSP URL in Settings before starting.")
            return RTSPCamera(url=url, width=w, height=h, fps=fps)
        if source == "picam" and _PICAM_OK:
            return PiCamera2Camera(width=w, height=h, fps=fps)
        # Default: demo
        return DemoCamera(width=w, height=h, fps=fps)

    # ------------------------------------------------------------------
    # Background capture thread
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Runs in a background thread: reads frames and runs inference."""
        assert self._camera is not None
        assert self._pipeline is not None

        while self._running:
            frame = self._camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            detections = self._pipeline.process(frame)
            annotated = self._pipeline.annotate(frame, detections)

            # Push frame for UI update (drop if queue full)
            try:
                self._frame_queue.put_nowait((annotated, detections))
            except queue.Full:
                pass

        logger.debug("Capture loop exited")

    # ------------------------------------------------------------------
    # Periodic GTK callback
    # ------------------------------------------------------------------

    def _on_frame_tick(self) -> bool:
        """Called by GLib every ~40 ms to update the preview and detections."""
        try:
            frame, detections = self._frame_queue.get_nowait()
        except queue.Empty:
            return True  # keep timer alive

        # Update preview
        alloc = self._preview.get_allocation()
        display_w = max(alloc.width, 320)
        display_h = max(alloc.height, 240)

        h, w = frame.shape[:2]
        scale = min(display_w / w, display_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        pixbuf = _bgr_to_pixbuf(resized)
        self._preview.set_from_pixbuf(pixbuf)

        # Update detections list
        for det in detections:
            ts = time.strftime("%H:%M:%S", time.localtime(det.timestamp))
            self._det_list.prepend([det.text or "-", f"{det.confidence:.0%}", ts])
            self._detection_count += 1
            # Trim list
            while len(self._det_list) > MAX_DETECTIONS:
                self._det_list.remove(self._det_list.get_iter(len(self._det_list) - 1))

        if detections:
            self._det_count_lbl.set_text(f"{self._detection_count} detections")

        return True  # keep timer alive

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_start_stop(self, _btn: Gtk.Button) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        try:
            self._camera = self._make_camera()
            self._camera.open()
        except CameraError as exc:
            self._show_error("Camera Error", str(exc))
            return

        self._pipeline = LPRPipeline(conf_threshold=self._settings["conf_threshold"])
        self._pipeline.open()

        if self._pipeline.is_mock:
            self._set_status("Running in DEMO mode (no Hailo hardware detected)")
        else:
            self._set_status("Running – Hailo-8 active")

        self._running = True
        self._start_btn.set_label("⏹  Stop")
        self._start_btn.get_style_context().remove_class("suggested-action")
        self._start_btn.get_style_context().add_class("destructive-action")
        self._source_combo.set_sensitive(False)

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="capture"
        )
        self._capture_thread.start()

    def _stop(self) -> None:
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=3.0)
            self._capture_thread = None
        if self._camera:
            self._camera.release()
            self._camera = None
        if self._pipeline:
            self._pipeline.close()
            self._pipeline = None

        self._start_btn.set_label("▶  Start")
        self._start_btn.get_style_context().remove_class("destructive-action")
        self._start_btn.get_style_context().add_class("suggested-action")
        self._source_combo.set_sensitive(True)
        self._set_status("Stopped.")

    def _on_clear(self, _btn: Gtk.Button) -> None:
        self._det_list.clear()
        self._detection_count = 0
        self._det_count_lbl.set_text("0 detections")

    def _on_settings(self, _btn: Gtk.Button) -> None:
        dlg = SettingsDialog(self, self._settings)
        response = dlg.run()
        if response == Gtk.ResponseType.OK:
            self._settings = dlg.get_settings()
        dlg.destroy()

    def _on_open_image(self, _btn: Gtk.Button) -> None:
        """Open a still image file and run the LPR pipeline on it."""
        dlg = Gtk.FileChooserDialog(
            title="Open Image",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dlg.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK,
        )

        # Filter to common image formats
        img_filter = Gtk.FileFilter()
        img_filter.set_name("Images (JPEG, PNG, BMP, TIFF)")
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp"):
            img_filter.add_pattern(pattern)
            img_filter.add_pattern(pattern.upper())
        dlg.add_filter(img_filter)

        all_filter = Gtk.FileFilter()
        all_filter.set_name("All files")
        all_filter.add_pattern("*")
        dlg.add_filter(all_filter)

        response = dlg.run()
        filepath = dlg.get_filename()
        dlg.destroy()

        if response != Gtk.ResponseType.OK or not filepath:
            return

        self._process_image_file(filepath)

    def _process_image_file(self, filepath: str) -> None:
        """Load *filepath*, run LPR, display result and log detections."""
        frame = cv2.imread(filepath)
        if frame is None:
            self._show_error(
                "Cannot Read Image",
                f"OpenCV could not decode the file:\n{filepath}\n\n"
                "Supported formats: JPEG, PNG, BMP, TIFF, WebP.",
            )
            return

        self._set_status(f"Processing image: {Path(filepath).name} …")

        # Create a one-shot pipeline if none is running
        own_pipeline = self._pipeline is None
        pipeline = self._pipeline
        if own_pipeline:
            pipeline = LPRPipeline(conf_threshold=self._settings["conf_threshold"])
            pipeline.open()

        assert pipeline is not None
        detections = pipeline.process(frame)
        annotated = pipeline.annotate(frame, detections)

        if own_pipeline:
            pipeline.close()

        # Display the annotated image
        alloc = self._preview.get_allocation()
        display_w = max(alloc.width, 320)
        display_h = max(alloc.height, 240)
        h, w = annotated.shape[:2]
        scale = min(display_w / w, display_h / h)
        resized = cv2.resize(annotated, (int(w * scale), int(h * scale)))
        self._preview.set_from_pixbuf(_bgr_to_pixbuf(resized))

        # Log detections
        for det in detections:
            ts = time.strftime("%H:%M:%S", time.localtime(det.timestamp))
            self._det_list.prepend([det.text or "-", f"{det.confidence:.0%}", ts])
            self._detection_count += 1
            while len(self._det_list) > MAX_DETECTIONS:
                self._det_list.remove(self._det_list.get_iter(len(self._det_list) - 1))
        if detections:
            self._det_count_lbl.set_text(f"{self._detection_count} detections")

        name = Path(filepath).name
        if detections:
            plates = ", ".join(d.text or "?" for d in detections)
            self._set_status(f"{name}: {len(detections)} plate(s) found – {plates}")
        else:
            self._set_status(f"{name}: no plates detected.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self._status_bar.pop(self._status_ctx)
        self._status_bar.push(self._status_ctx, msg)

    def _show_error(self, title: str, msg: str) -> None:
        dlg = Gtk.MessageDialog(
            parent=self,
            flags=Gtk.DialogFlags.MODAL,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.CLOSE,
            text=title,
        )
        dlg.format_secondary_text(msg)
        dlg.run()
        dlg.destroy()

    def do_delete_event(self, _event) -> bool:
        self._stop()
        return False


# ---------------------------------------------------------------------------
# GTK Application
# ---------------------------------------------------------------------------

class LNPRApp(Gtk.Application):
    def __init__(self) -> None:
        super().__init__(application_id=APP_ID)
        self._cli_args = None

    def set_cli_args(self, args) -> None:  # noqa: ANN001
        """Store parsed CLI arguments to be applied on activation."""
        self._cli_args = args

    def do_activate(self) -> None:
        win = LNPRWindow(self)
        # Apply CLI-provided defaults if available
        if self._cli_args is not None:
            args = self._cli_args
            settings_patch: dict = {}
            if args.width:
                settings_patch["width"] = args.width
            if args.height:
                settings_patch["height"] = args.height
            if args.fps:
                settings_patch["fps"] = args.fps
            if args.conf_threshold:
                settings_patch["conf_threshold"] = args.conf_threshold
            if args.rtsp_url:
                settings_patch["rtsp_url"] = args.rtsp_url
            if args.usb_device is not None:
                settings_patch["usb_device"] = args.usb_device
            win._settings.update(settings_patch)  # noqa: SLF001

            if args.source:
                source_id = args.source
                combo = win._source_combo  # noqa: SLF001
                model = combo.get_model()
                for i, row in enumerate(model):
                    if row[0] == source_id:
                        combo.set_active(i)
                        break
        win.present()
