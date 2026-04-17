#!/usr/bin/env python3
"""LNPR – Licence Number Plate Recognition

Entry point.  Run with::

    python main.py [--demo] [--source usb|rtsp|picam|demo] [--debug]
"""

from __future__ import annotations

import argparse
import logging
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lnpr",
        description="Licence Number Plate Recognition – Raspberry Pi 5 / Hailo-8",
    )
    parser.add_argument(
        "--source",
        choices=["demo", "usb", "rtsp", "picam"],
        default=None,
        help="Pre-select camera source on startup.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force demo/synthetic mode (equivalent to --source demo).",
    )
    parser.add_argument(
        "--rtsp-url",
        default=None,
        help="RTSP stream URL (used when --source rtsp).",
    )
    parser.add_argument(
        "--usb-device",
        type=int,
        default=0,
        help="USB camera device index (default: 0).",
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Frame width (default: 1280)."
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Frame height (default: 720)."
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Target frame rate (default: 30)."
    )
    parser.add_argument(
        "--lpd-hef",
        default="models/lpd.hef",
        help="Path to licence-plate detection HEF model.",
    )
    parser.add_argument(
        "--lpr-hef",
        default=None,
        help="Path to LPRNet recognition HEF model (optional).",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.45,
        help="Detection confidence threshold (default: 0.45).",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG logging."
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import gi
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk  # noqa: F401

    from ui.app import LNPRApp

    app = LNPRApp()

    # Pass CLI settings into the app via the public API
    if args.demo:
        args.source = "demo"

    app.set_cli_args(args)
    return app.run(sys.argv[:1])  # pass only argv[0] to GTK


if __name__ == "__main__":
    sys.exit(main())
