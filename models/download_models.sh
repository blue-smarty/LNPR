#!/usr/bin/env bash
# download_models.sh – Download or compile Hailo HEF models for LNPR
#
# Usage:
#   chmod +x models/download_models.sh
#   ./models/download_models.sh
#
# Requires:
#   pip install hailo-model-zoo          (for compilation path)
#   Hailo Dataflow Compiler installed    (for compilation path)

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== LNPR model setup ==="
echo "Models directory: $MODELS_DIR"
echo ""

# -------------------------------------------------------------------------
# Try to use hailo-apps-infra pre-compiled HEFs first (fastest)
# -------------------------------------------------------------------------

INFRA_REPO="https://github.com/hailo-ai/hailo-apps-infra.git"
INFRA_DIR="/tmp/hailo-apps-infra"

if ! command -v git &>/dev/null; then
    echo "ERROR: git is not installed." >&2
    exit 1
fi

echo "Cloning hailo-apps-infra (shallow) ..."
git clone --depth=1 "$INFRA_REPO" "$INFRA_DIR" 2>/dev/null || true

LPD_SRC="$INFRA_DIR/resources/hefs/lpd_yolov5s.hef"
LPR_SRC="$INFRA_DIR/resources/hefs/lprnet.hef"

if [[ -f "$LPD_SRC" ]]; then
    echo "Found lpd_yolov5s.hef in hailo-apps-infra"
    cp "$LPD_SRC" "$MODELS_DIR/lpd.hef"
else
    echo "lpd_yolov5s.hef not found in hailo-apps-infra."
    echo "Attempting compilation via hailomz ..."
    if command -v hailomz &>/dev/null; then
        hailomz compile lpd_yolov5s --hw-arch hailo8 --yaml hailo_model_zoo/cfg/networks/lpd_yolov5s.yaml
        mv lpd_yolov5s.hef "$MODELS_DIR/lpd.hef"
    else
        echo "WARNING: hailomz not found. Please install hailo-model-zoo:"
        echo "  pip install hailo-model-zoo"
        echo "Then run: hailomz compile lpd_yolov5s"
    fi
fi

if [[ -f "$LPR_SRC" ]]; then
    echo "Found lprnet.hef in hailo-apps-infra"
    cp "$LPR_SRC" "$MODELS_DIR/lprnet.hef"
else
    echo "lprnet.hef not found – recognition stage will be skipped."
fi

echo ""
echo "=== Done ==="
echo "Contents of $MODELS_DIR:"
ls -lh "$MODELS_DIR"/*.hef 2>/dev/null || echo "  (no .hef files present – demo mode will be used)"
