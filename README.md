# LNPR – Licence Number Plate Recognition

A **Python / GTK3** application that detects and reads vehicle licence plates in
real time on a **Raspberry Pi 5** with a **Hailo-8** AI accelerator.

```
┌─────────────────────────────────────────────────────────────────────┐
│ Source: [Demo ▾]  [▶ Start]  [📂 Open Image]  [🗑 Clear]  [⚙ Settings] │
├──────────────────────────────────────┬──────────────────────────────┤
│                                      │ Recent Detections            │
│   Live camera preview /              │ ──────────────────────────── │
│   Still image display                │ AB12 CDE  92%  14:30:01      │
│   (annotated bounding boxes)         │ XY34 FGH  88%  14:29:45      │
└──────────────────────────────────────┴──────────────────────────────┘
│ Running in DEMO mode (no Hailo hardware detected)                    │
```

---

## Features

| Feature | Detail |
|---------|--------|
| **Camera inputs** | RTSP stream, USB / V4L2 camera, Raspberry Pi Camera (PiCamera2) |
| **Still image** | Upload any JPEG / PNG / BMP / TIFF image for instant plate recognition |
| **AI inference** | Hailo-8 via `hailo_platform` SDK (YOLOv5s LPD + LPRNet) |
| **UI** | GTK3 — source selector, Start/Stop, Open Image, live preview, detections list, settings panel |
| **Update check** | Check GitHub releases from CLI (`--check-updates`) or UI button |
| **Demo mode** | Synthetic animated frames; no hardware needed |
| **No libatlas** | Uses only `numpy` + `opencv`; no BLAS/ATLAS dependency |
| **Target OS** | Debian Trixie (bookworm-compatible), aarch64 |

---

## Project layout

```
LNPR/
├── main.py                 # Entry point
├── install.sh              # One-shot installation script
├── uninstall.sh            # Uninstall / clean-up script
├── requirements.txt        # Python dependencies
│
├── src/
│   ├── camera/
│   │   ├── base.py         # Abstract CameraBase
│   │   ├── usb_camera.py   # USB / V4L2 source
│   │   ├── rtsp_camera.py  # RTSP stream source
│   │   ├── picam.py        # Raspberry Pi camera (picamera2)
│   │   └── demo_camera.py  # Synthetic / video-file demo source
│   ├── inference/
│   │   └── hailo_inference.py  # Hailo-8 runtime wrapper
│   └── lpr/
│       └── pipeline.py     # LPD → crop → LPRNet pipeline
│
├── ui/
│   └── app.py              # GTK3 application & main window
│
├── models/
│   ├── README.md           # Model acquisition instructions
│   └── download_models.sh  # Helper script to fetch/compile HEFs
│
└── tests/
    ├── conftest.py
    ├── test_camera.py
    └── test_lpr.py
```

---

## Quick Start

### 1. Hardware requirements

| Item | Notes |
|------|-------|
| Raspberry Pi 5 | 4 GB RAM or more recommended |
| Hailo-8 M.2 HAT+ | or USB-connected Hailo-8 |
| Camera | USB webcam, CSI PiCamera2 module, or IP/RTSP camera |
| OS | Raspberry Pi OS (Debian Trixie / bookworm) – 64-bit |

### 2. Install system packages & Python dependencies

```bash
git clone https://github.com/blue-smarty/LNPR.git
cd LNPR
chmod +x install.sh
./install.sh
```

The script installs:
* `python3-gi`, `gir1.2-gtk-3.0` (GTK3 Python bindings)
* `python3-opencv`, `python3-numpy`
* `python3-picamera2`, `libcamera-apps`
* GStreamer plugins (for RTSP)
* `ffmpeg`

> **No `libatlas-dev`** is used anywhere in this project.

### 3. Hailo SDK setup

The Hailo Runtime SDK (`hailo_platform`) must be installed from the
[Hailo Developer Zone](https://hailo.ai/developer-zone/).

```bash
# After downloading the wheel from the Developer Zone:
source .venv/bin/activate
pip install /path/to/hailo_platform-*.whl
```

Alternatively, use the official Hailo App Suite installer which places the
SDK in the system Python path (the virtual environment is created with
`--system-site-packages` so it will be found automatically).

### 4. Acquire Hailo models

```bash
chmod +x models/download_models.sh
./models/download_models.sh
```

This script attempts to:
1. Clone `hailo-ai/hailo-apps-infra` and copy pre-compiled HEFs.
2. Fall back to `hailomz compile` (requires Hailo Dataflow Compiler).

See [`models/README.md`](models/README.md) for manual steps and links.

#### Models used

| Model | HEF file | Task |
|-------|----------|------|
| `lpd_yolov5s` | `models/lpd.hef` | Licence-plate **detection** |
| `lprnet` | `models/lprnet.hef` | Licence-plate **recognition** (optional) |

> If neither model is present the application starts in **demo mode**
> automatically.

### 5. Run

```bash
source .venv/bin/activate

# Demo mode (no hardware needed)
python main.py --demo

# USB camera (default device /dev/video0)
python main.py --source usb

# Raspberry Pi camera
python main.py --source picam

# RTSP stream (single)
python main.py --source rtsp --rtsp-url "rtsp://admin:secret@192.168.1.64:554/h264Preview_01_main"

# Multiple RTSP streams (comma-separated)
python main.py --source rtsp --rtsp-url "rtsp://cam1/stream1,rtsp://cam2/stream1"

# Debug logging
python main.py --demo --debug

# Check for a newer LNPR release
python main.py --check-updates
```

All options:

```
usage: lnpr [-h] [--source {demo,usb,rtsp,picam}] [--demo]
            [--rtsp-url RTSP_URL] [--usb-device USB_DEVICE]
            [--width WIDTH] [--height HEIGHT] [--fps FPS]
            [--lpd-hef LPD_HEF] [--lpr-hef LPR_HEF]
            [--conf-threshold CONF_THRESHOLD] [--debug]
```

---

## UI walkthrough

1. **Source** dropdown – select *Demo*, *USB Camera*, *PiCamera2*, or *RTSP Stream*.
2. **▶ Start** – opens the camera source and starts inference; button changes to **⏹ Stop**.
3. **📂 Open Image** – open any JPEG / PNG / BMP / TIFF file for instant still-image recognition.  Works at any time, even while a live stream is running.
4. **⚙ Settings** – configure frame size, FPS, detection confidence threshold, RTSP URL(s) (with format hints), and USB device index.
5. **Preview pane** – shows the live feed *or* the last uploaded still image, with green bounding boxes and plate text overlaid.
6. **Recent Detections** panel – timestamped list of the last 50 recognised plates (from both live and still-image sources).
7. **🗑 Clear** – resets the detections list.
8. **🔄 Check Updates** – checks GitHub for a newer LNPR release.
9. **Status bar** – shows current state (Ready / Running / file processed / error messages).

### Still image recognition

Click **📂 Open Image** at any time to open a file chooser.  Select a photo of
a vehicle (JPEG, PNG, BMP, TIFF, WebP).  The LPR pipeline runs immediately on
the loaded image: annotated bounding boxes appear in the preview pane, and any
recognised plates are added to the detections list.  The camera stream (if
active) resumes automatically.

---

## RTSP stream URL format

```
rtsp://[user:password@]<host>[:<port>]/<path>
```

| Component | Description |
|-----------|-------------|
| `user:password@` | Optional credentials.  Omit if the camera has no auth. |
| `host` | Camera IP address or hostname (e.g. `192.168.1.64`). |
| `port` | Optional; default is **554**.  Some cameras use **8554**. |
| `/path` | Stream path – varies by camera make/firmware. |

### Common examples

```bash
# Generic anonymous stream (no auth, default port)
rtsp://192.168.1.64/stream1

# With credentials
rtsp://admin:secret@192.168.1.64:554/h264Preview_01_main

# Lower-resolution sub-stream (less bandwidth)
rtsp://admin:secret@192.168.1.64:554/h264Preview_01_sub

# Alternate port
rtsp://192.168.1.64:8554/live

# Hikvision IP cameras
rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101

# Dahua IP cameras
rtsp://admin:password@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0

# Reolink cameras
rtsp://admin:password@192.168.1.64:554//h264Preview_01_main
```

### Multiple RTSP streams

You can run more than one RTSP stream at once by entering multiple URLs:

* In CLI, pass comma-separated URLs to `--rtsp-url`.
* In **⚙ Settings**, separate URLs with commas or new lines.

When multiple URLs are provided, LNPR opens all streams and creates one
inference pipeline per stream.

> **Tip**: Enter the URL in **⚙ Settings → RTSP URL** and hover over the
> field for quick examples.  Then select *RTSP Stream* in the Source dropdown
> and press **▶ Start**.

### Troubleshooting RTSP

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| "Cannot open RTSP stream" | Wrong URL or unreachable host | Check IP / port with `ping` or VLC |
| Choppy / frozen video | Network latency | Increase *latency* (code default: 200 ms) |
| "FFMPEG backend failed" | OpenCV built without FFMPEG | Install `ffmpeg`; app auto-retries via GStreamer |
| No colour | Wrong pixel format | Ensure stream is H.264; MJPEG streams may need extra configuration |

---

## Architecture & design notes

### Camera abstraction

`src/camera/base.py` defines `CameraBase` – a Python abstract class with
`open()`, `read()`, `release()` methods and context-manager support.  Each
source (`USBCamera`, `RTSPCamera`, `PiCamera2Camera`, `DemoCamera`) implements
this interface.  The UI and pipeline only ever talk to `CameraBase`, so adding
new sources requires no changes elsewhere.

### Hailo inference wrapper

`src/inference/hailo_inference.py` wraps `hailo_platform.InferVStreams` into a
simple `HailoInference` class.  When the SDK is not installed it silently enters
**mock mode**, returning empty tensors.  This allows the full UI to be exercised
on any machine.

### LPR pipeline

`src/lpr/pipeline.py` chains two Hailo models:

```
frame → LPD (YOLOv5s) → bounding boxes
         └─ crop ─→ LPRNet → CTC decode → plate text
```

Post-processing (`_parse_yolo_detections`, `_decode_lprnet_output`) is pure
NumPy; no BLAS/ATLAS dependency.

### GTK3 UI

`ui/app.py` uses only `gi.repository.Gtk` and `gi.repository.GdkPixbuf`
(both available as Debian packages).  Frames are rendered via `GdkPixbuf`
converted from OpenCV BGR arrays.  A `GLib.timeout_add` timer drives UI updates
at ≤25 fps, decoupled from the background capture thread via a small
`queue.Queue`.

---

## Uninstall

Run the interactive uninstall script from the project directory:

```bash
chmod +x uninstall.sh
./uninstall.sh
```

The script removes items in four steps, each confirmed interactively where
there is risk of removing shared resources:

| Step | What is removed | Interactive? |
|------|----------------|--------------|
| 1 | Python virtual environment (`.venv/`) | No |
| 2 | Downloaded Hailo model files (`models/*.hef`) and any cloned infra repos | No |
| 3 | apt packages installed by `install.sh` | **Yes** – skipped by default |
| 4 | The entire LNPR project directory | **Yes** – skipped by default |

> **Note**: The Hailo Runtime SDK (`hailo_platform`) must be uninstalled
> separately, following the instructions from the
> [Hailo Developer Zone](https://hailo.ai/developer-zone/).

---

## Running tests

```bash
source .venv/bin/activate
pip install pytest
pytest tests/ -v
```

Tests run entirely in **mock / demo mode** – no Hailo hardware or camera
required.

---

## Assumptions & limitations

* The application targets **Hailo-8** (PCIe/M.2 HAT+ on RPi 5).  Hailo-8L
  may work with recompiled HEFs but is untested.
* RTSP decoding relies on OpenCV's FFMPEG back-end or GStreamer; ensure the
  stream uses H.264 for best compatibility.
* Plate recognition accuracy depends on the LPRNet model and the target
  country's plate format.  Post-processing assumes left-to-right CTC decoding.
* The UI is single-monitor, single-window; no multi-display support.
* `libatlas-dev` is **not** used or required anywhere.

---

## License

MIT – see [LICENSE](LICENSE).

---

## Acknowledgements

* [Hailo AI](https://hailo.ai/) – Hailo-8 SDK and Model Zoo
* [hailo-ai/hailo-apps](https://github.com/hailo-ai/hailo-apps) – reference
  application patterns
* [OpenCV](https://opencv.org/) – camera and image processing
* [GTK](https://www.gtk.org/) – GUI toolkit
