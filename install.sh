#!/usr/bin/env bash
# install.sh – Set up LNPR on Raspberry Pi 5 (Debian Trixie / aarch64)
#
# Run as a normal user with sudo rights:
#   chmod +x install.sh
#   ./install.sh

set -euo pipefail

echo "============================================================"
echo "  LNPR – Licence Number Plate Recognition"
echo "  Installation script for Raspberry Pi 5 / Debian Trixie"
echo "============================================================"
echo ""

# ---- 1. System packages ----

echo "[1/5] Installing system dependencies ..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-3.0 \
    gir1.2-gdkpixbuf-2.0 \
    python3-opencv \
    python3-numpy \
    python3-picamera2 \
    libcamera-apps \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-rtsp \
    ffmpeg

echo ""
echo "[2/5] Creating Python virtual environment ..."
python3 -m venv --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo ""
echo "[3/5] Installing Python requirements ..."
pip install --upgrade pip -q
# OpenCV headless (system python3-opencv is preferred for RPi – this is a fallback)
pip install "numpy>=1.24" -q || true

# ---- 4. Hailo SDK ----

echo ""
echo "[4/5] Hailo SDK setup ..."
echo ""
echo "  The Hailo Runtime SDK must be installed from the Hailo Developer Zone:"
echo "  https://hailo.ai/developer-zone/"
echo ""
echo "  After downloading hailo_platform-*.whl, run:"
echo "    source .venv/bin/activate"
echo "    pip install /path/to/hailo_platform-*.whl"
echo ""
echo "  If the SDK is already installed system-wide (e.g. via the official"
echo "  Hailo installer), the --system-site-packages venv above will pick"
echo "  it up automatically."

# ---- 5. Models ----

echo ""
echo "[5/5] Downloading / compiling Hailo models ..."
chmod +x models/download_models.sh
./models/download_models.sh || echo "  (model download skipped – demo mode will be used)"

echo ""
echo "============================================================"
echo "  Installation complete!"
echo ""
echo "  To run LNPR:"
echo "    source .venv/bin/activate"
echo "    python main.py"
echo ""
echo "  To run in demo mode (no hardware required):"
echo "    python main.py --demo"
echo "============================================================"
