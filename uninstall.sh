#!/usr/bin/env bash
# uninstall.sh – Remove LNPR from a Raspberry Pi 5 (Debian Trixie / aarch64)
#
# Run from the LNPR project directory as a normal user with sudo rights:
#   chmod +x uninstall.sh
#   ./uninstall.sh
#
# What this script removes (in order):
#   1. Python virtual environment (.venv)
#   2. Downloaded Hailo model files (*.hef inside models/)
#   3. Optionally: apt packages that were installed exclusively for LNPR
#   4. Optionally: the LNPR project directory itself
#
# What this script does NOT remove:
#   • hailo_platform SDK (installed separately via pip / Hailo installer)
#   • System-wide packages that were already present before LNPR was installed
#     (the apt purge step is interactive and opt-in for that reason)

set -euo pipefail

# ---- Colour helpers ----
RED='\033[0;31m'
YLW='\033[1;33m'
GRN='\033[0;32m'
NC='\033[0m'  # no colour

info()    { echo -e "${GRN}[info]${NC}  $*"; }
warn()    { echo -e "${YLW}[warn]${NC}  $*"; }
confirm() {
    # confirm <prompt>  – returns 0 for yes, 1 for no
    local reply
    read -r -p "$1 [y/N] " reply
    [[ "${reply,,}" == "y" ]]
}

echo ""
echo "============================================================"
echo "  LNPR – Licence Number Plate Recognition"
echo "  Uninstall script for Raspberry Pi 5 / Debian Trixie"
echo "============================================================"
echo ""

# Make sure we are in the project root (where install.sh lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 1. Virtual environment ----

echo "[1/4] Removing Python virtual environment ..."
if [[ -d ".venv" ]]; then
    rm -rf .venv
    info ".venv removed."
else
    warn ".venv directory not found – already removed or never created."
fi

# ---- 2. Model files ----

echo ""
echo "[2/4] Removing downloaded Hailo model files ..."
HEF_COUNT=0
while IFS= read -r -d '' hef; do
    rm -f "$hef"
    info "Removed: $hef"
    HEF_COUNT=$((HEF_COUNT + 1))
done < <(find models -maxdepth 1 -name "*.hef" -print0 2>/dev/null || true)

if [[ $HEF_COUNT -eq 0 ]]; then
    warn "No .hef model files found in models/ – nothing to remove."
fi

# Any temporary model download artefacts (cloned infra repo, compiled files)
if [[ -d "models/hailo-apps-infra" ]]; then
    rm -rf models/hailo-apps-infra
    info "Removed models/hailo-apps-infra clone."
fi

# ---- 3. Optional: apt packages ----

echo ""
echo "[3/4] Optional: remove apt packages installed by install.sh"
echo ""
echo "  The following packages were installed by install.sh:"
echo "    python3-gi  python3-gi-cairo  gir1.2-gtk-3.0  gir1.2-gdkpixbuf-2.0"
echo "    python3-opencv  python3-numpy  python3-picamera2  libcamera-apps"
echo "    gstreamer1.0-tools  gstreamer1.0-plugins-base  gstreamer1.0-plugins-good"
echo "    gstreamer1.0-plugins-bad  gstreamer1.0-rtsp  ffmpeg"
echo ""
echo -e "  ${YLW}WARNING:${NC} Some of these packages may be used by other applications."
echo "  Only remove them if you are sure no other software depends on them."
echo ""

if confirm "  Remove these apt packages now?"; then
    info "Removing apt packages ..."
    sudo apt-get remove -y \
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
    sudo apt-get autoremove -y
    info "Apt packages removed."
else
    info "Apt packages kept (skipped)."
fi

# ---- 4. Optional: project directory ----

echo ""
echo "[4/4] Optional: remove the LNPR project directory"
echo ""
echo "  Project directory: $SCRIPT_DIR"
echo ""
echo -e "  ${RED}WARNING:${NC} This will permanently delete ALL project files, including"
echo "  any configuration changes, logs, or custom models you may have added."
echo ""

if confirm "  Delete the entire project directory ($SCRIPT_DIR)?"; then
    PARENT_DIR="$(dirname "$SCRIPT_DIR")"
    info "Deleting $SCRIPT_DIR ..."
    rm -rf "$SCRIPT_DIR"
    info "Project directory removed."
    # After this point the script itself no longer exists, so exit cleanly.
    cd "$PARENT_DIR"
else
    info "Project directory kept (skipped)."
fi

echo ""
echo "============================================================"
echo "  LNPR uninstall complete."
echo ""
echo "  If you installed the Hailo Runtime SDK separately, remove it"
echo "  by following the instructions from the Hailo Developer Zone:"
echo "  https://hailo.ai/developer-zone/"
echo "============================================================"
echo ""
