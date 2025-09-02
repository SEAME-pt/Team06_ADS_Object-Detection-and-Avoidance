# YOLO11 TensorRT Runtime — Jetson/PC

> Real‑time object/sign detection with **TensorRT** (+ PyCUDA) and OpenCV. Includes an FPS‑aware camera loop with optional **frame‑skipping strategies** for stable performance on embedded GPUs.

Core files:

* `yolo11.py` — TensorRT runtime with YOLO‑11 style output + frame skipping/FPS utils
* `testModel.py` — alternate TensorRT runtime (YOLO‑v8 style postprocess)
* `strip.py` — utility to strip optimizer/EMA from a PyTorch checkpoint (`best.pt → otimize.pt`)
* `dataset.yaml` — dataset spec (YOLO format)

---

## Contents

1. [Quick Start](#quick-start)
2. [Building the TensorRT Engine](#building-the-tensorrt-engine)
3. [Run Inference (USB/CSI Camera)](#run-inference-usbcsi-camera)
4. [Dataset Format](#dataset-format)
5. [Class Map & Colors](#class-map--colors)
6. [Frame Skipping & FPS Utilities](#frame-skipping--fps-utilities)
7. [Tips, Troubleshooting & Performance](#tips-troubleshooting--performance)
8. [Project Structure](#project-structure)
9. [License](#license)

---

## Quick Start

### Requirements

* **Jetson** (Nano/Xavier/Orin) with JetPack (TensorRT + CUDA + cuDNN already installed) **or** x86\_64 with NVIDIA GPU and TensorRT installed
* Python 3.8–3.11

Python packages (runtime):

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install pycuda opencv-python numpy
# Optional (for GPU memory info in scripts):
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

> On Jetson, `tensorrt` is provided by JetPack; on x86, install TensorRT via NVIDIA packages. The scripts import `tensorrt` and `pycuda`.

You need a TensorRT **engine file** (e.g., `best.engine`) built from your exported ONNX.

---

## Building the TensorRT Engine

Export your trained YOLO model to **ONNX** first (via your training code/Ultralytics). Then build an engine with `trtexec`:

```bash
# Fixed input 1x3x352x352 (recommended for current scripts)
/usr/src/tensorrt/bin/trtexec \
  --onnx=model.onnx \
  --saveEngine=best.engine \
  --explicitBatch \
  --optShapes=input:1x3x352x352 \
  --minShapes=input:1x3x352x352 \
  --maxShapes=input:1x3x352x352 \
  --workspace=2048 \
  --fp16
```

Notes:

* Keep the input size consistent with the script flag `--input-size` (default **352**).
* If your ONNX input name is not `input`, replace it accordingly.
* For dynamic shapes, you can set different `min/opt/max`, but the sample code assumes a **static** 352×352.

---

## Run Inference (USB/CSI Camera)

### A) `yolo11.py` — YOLO‑11 style output + frame skipping

```bash
python yolo11.py \
  --engine best.engine \
  --input-size 352 \
  --skip-strategy fixed \
  --skip-frames 8 \
  --target-fps 60 \
  --camera-source 0        # USB cam index, or RTSP/HTTP URL
# add --use-csi to use the CSI camera (Jetson)
```

Key options:

* `--skip-strategy`: `fixed` | `adaptive` | `time_based`
* `--skip-frames`: frames to skip (for `fixed`)
* `--target-fps`: requested rate for `adaptive/time_based`

### B) `testModel.py` — YOLO‑v8 style postprocess

This script loads `model.engine` by default. Edit the filename inside or adapt to CLI.

```bash
python testModel.py
```

Both scripts draw **bounding boxes + class labels + confidences** and overlay **FPS** info on the preview window (OpenCV).

---

## Dataset Format

This repository uses the standard YOLO directory layout defined by `dataset.yaml`:

```yaml
path: /home/djoker/code/yolo2/dataset
train: images/train
val: images/val
nc: 10
names:
  0: STOP
  1: PASSAGEM
  2: VEL_50
  3: VEL_80
  4: SEMAFORO_VERMELHO
  5: SEMAFORO_VERDE
  6: SEMAFORO_LARANJA
  7: PASSADEIRA
  8: DANGER
  9: CURVA
```

* `images/<split>` contain the RGB images; labels must be in `labels/<split>` with the same stems (YOLO txt format).
* `nc` and `names` define the **training classes**; keep these in sync with your runtime class map.

---

## Class Map & Colors

The runtime maps class IDs to names and colors for drawing. **Make sure the runtime map matches your trained classes.**

* In `yolo11.py` the `self.classes` dictionary currently includes entries like `'person'`, `'car'`, `'stop sign'`, `'passadeira'`, `'speed 50'`, `'speed 80'`, etc. Adjust this mapping to reflect **exactly** the 10 classes (or more) used during training.
* Colors are defined in `self.colors` (BGR tuples). You can safely edit/add colors per class ID.

> If the engine was trained with a different class order/size, you must update both `self.classes` and any logic that depends on `num_classes` to avoid mis‑labeled boxes.

---

## Frame Skipping & FPS Utilities

`yolo11.py` includes two helpers:

* **`FPSCalculator`** — smooths FPS computation by keeping a rolling window and trimming outliers.
* **`FrameSkipper`** — strategies:

  * `fixed`: process 1 in N frames (`--skip-frames`)
  * `adaptive`: adjusts skip ratio based on measured processing time vs `--target-fps`
  * `time_based`: processes when the time since last process ≥ `1/target_fps`

These keep the UI responsive and stabilize throughput on smaller GPUs (e.g., Jetson Nano/Xavier).

---

## Tips, Troubleshooting & Performance

* **Post‑processing format**: `yolo11.py` assumes an output tensor shaped like `[4 + num_classes, num_anchors]` (or flat, reshaped accordingly). `testModel.py` expects YOLOv8‑style `[1, 4+num_classes, N]` → transposed to `(N, 4+num_classes)`. If boxes look wrong, verify your ONNX output order.
* **NMS**: both scripts use OpenCV `cv2.dnn.NMSBoxes` with `(x, y, w, h)`; ensure your conversion from center `(cx, cy, w, h)` to `(x1, y1, x2, y2)` then to `(x, y, w, h)` is correct.
* **Input size**: keep `--input-size` consistent with how you built the engine. The preprocessing pads with 114 to preserve aspect ratio.
* **GPU mem**: on Jetson, prefer `--fp16` engines. Close other GPU apps; drop resolution if needed.
* **CSI pipeline**: `setup_csi_camera()` builds a GStreamer pipeline for Jetson. If the camera doesn’t open, confirm the sensor supports the requested format/framerate.
* **Classes mismatch**: if detections show wrong labels, align `self.classes` with `dataset.yaml` and rebuild the engine if needed.

---

## Project Structure

```
.
├── yolo11.py          # TensorRT runtime (YOLO‑11 postprocess + FPS/frame‑skip)
├── testModel.py       # TensorRT runtime (YOLOv8‑style postprocess)
├── strip.py           # strip optimizer/EMA from best.pt → otimize.pt
├── dataset.yaml       # YOLO dataset spec (paths, class names)
└── README.md          # this file
```

---

 