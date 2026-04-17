# Models

This directory holds compiled Hailo Executable Format (`.hef`) model files.
These files are **not** included in the repository because they are large binary
artefacts.  Use the `download_models.sh` script (or the instructions below) to
obtain them.

---

## Required models

| File | Purpose | Source |
|------|---------|--------|
| `lpd.hef` | Licence-plate **detection** | Hailo Model Zoo |
| `lprnet.hef` | Licence-plate **recognition** (optional) | Hailo Model Zoo |

---

## Obtaining the models

### Option A – Hailo Model Zoo (recommended)

```bash
pip install hailo-model-zoo          # installs the `hailomz` CLI
hailomz compile lpd_yolov5s          # compiles for Hailo-8
hailomz compile lprnet
mv lpd_yolov5s.hef models/lpd.hef
mv lprnet.hef         models/
```

> **Note**: compilation requires the Hailo Dataflow Compiler (DFC) and an
> active Hailo Developer Zone account (<https://hailo.ai/developer-zone/>).

### Option B – Pre-compiled from Hailo App Suite

```bash
# Clone the Hailo App Suite (contains pre-compiled HEFs for common models)
git clone https://github.com/hailo-ai/hailo-apps-infra
ls hailo-apps-infra/resources/hefs/
# Copy the relevant HEFs here
cp hailo-apps-infra/resources/hefs/lpd_yolov5s.hef models/lpd.hef
```

### Option C – Hailo Developer Zone portal

Download directly from <https://hailo.ai/developer-zone/model-zoo/> after
logging in with your Developer Zone credentials.

---

## Demo / no-hardware mode

If no `.hef` files are present the application automatically runs in **demo
mode** (mock inference) so that the UI can be evaluated without Hailo hardware.
