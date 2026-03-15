# ANPR: Automatic Number Plate Recognition Pipeline

End-to-end ANPR for video footage using YOLOv8 + SORT + PaddleOCR, tuned for Indian plate formats and noisy real-world traffic scenes.

The repository provides:
- Inference pipeline with tracking, OCR stabilization, and CSV/video outputs.
- Dataset builder that merges Hugging Face + Indian Roboflow YOLO datasets.
- YOLO training script with automatic fine-tune resume from an existing checkpoint.

## Features

- Two-stage detection:
   - Vehicle detector on full frame.
   - Plate detector inside each tracked vehicle ROI.
- OCR robustness for Indian plates:
   - IND-strip noise cleanup.
   - Positional character correction templates.
   - Strict Indian regex validation.
   - Multi-variant OCR preprocessing (base + adaptive threshold variants).
- Temporal stabilization:
   - Confidence-weighted voting per track.
   - Best-read fallback memory per track.
- Throughput controls:
   - OCR frame-gap throttling per track.
   - Per-frame OCR budget cap.
   - Blur-gating before OCR.
- Output stabilization:
   - Raw CSV (frame-by-frame) + smoothed CSV via interpolation.
   - Optional final re-rendered polished video.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run pipeline:

```bash
python pipeline.py --source input.mp4 --output output.mp4 --csv results.csv
```

3. Optional polished output pass:

```bash
python pipeline.py --source input.mp4 --final final_output.mp4
```

## Runtime Behavior

- Default console mode prints progress every N frames.
- Use verbose mode for extra debug lines:

```bash
python pipeline.py --source input.mp4 --verbose
```

### CLI Arguments

- --source: input video path (default: input.mp4)
- --output: first-pass annotated video (default: output.mp4)
- --csv: smoothed CSV path (raw CSV is auto-written as *_raw.csv)
- --final: optional polished output video path (default: final_output.mp4)
- --skip-final-render: skip second-pass polished render
- --verbose: enable additional debug messages

## Configuration Knobs (pipeline.py)

Important runtime knobs are defined near the top of [pipeline.py](pipeline.py):

- Detection:
   - VEHICLE_CONF_THRESHOLD
   - PLATE_CONF_THRESHOLD
   - OCR_CONF_THRESHOLD
- OCR quality/speed:
   - PLATE_BLUR_VAR_THRESHOLD
   - OCR_MIN_FRAME_GAP
   - MAX_OCR_CALLS_PER_FRAME
   - CAR_BBOX_SMOOTH_WINDOW
   - MIN_TRACK_FRAMES_FOR_OUTPUT
- Throughput/accuracy trade-off:
   - INFERENCE_IMG_SIZE

Current defaults in code:
- INFERENCE_IMG_SIZE = 1024
- OCR_MIN_FRAME_GAP = 2
- MAX_OCR_CALLS_PER_FRAME = 4
- CAR_BBOX_SMOOTH_WINDOW = 5
- MIN_TRACK_FRAMES_FOR_OUTPUT = 0

## Why FPS Drops (and How to Fix)

FPS can plummet when scene density rises because compute scales with:
- Number of tracked vehicles (more plate ROI detections).
- Number of OCR calls (and each call runs multiple preprocessing variants).
- OCR backend running on CPU.

Practical tuning order:
1. Lower INFERENCE_IMG_SIZE (for example, 1024 -> 640 -> 512).
2. Increase OCR_MIN_FRAME_GAP (for example, 2 -> 3).
3. Lower MAX_OCR_CALLS_PER_FRAME (for example, 4 -> 2).
4. Slightly raise PLATE_CONF_THRESHOLD to skip weaker candidates.

## Flicker Control (Boxes + Text)

### 1) Box vibration smoothing

The post-processing step now supports centered moving-average smoothing of
`car_bbox` coordinates in the smoothed CSV path:

- `CAR_BBOX_SMOOTH_WINDOW = 5` (default)

This reduces jitter from detection/tracking micro-motions.

### 2) Ghost-track suppression (optional)

Short-lived IDs can be dropped at post-processing time:

- `MIN_TRACK_FRAMES_FOR_OUTPUT = 0` (default: keep all)

Set it to values like `10` to `15` if dense distant traffic creates many brief
flicker IDs.

### 3) Global-best text lock

Final polished render already uses a global-best lock per `car_id`:

- For each track, highest-confidence `license_number` is chosen once.
- That same text is rendered for all frames of that track in `final_output.mp4`.

## OCR Backend Notes (Windows)

- The pipeline attempts GPU Paddle initialization first, then falls back.
- In mixed Torch + Paddle GPU setups on Windows, runtime conflicts can occur depending on CUDA/cuDNN/DLL combinations.
- If OCR GPU is unstable in your environment, CPU OCR remains the reliable fallback.

## Dataset Builder (Merged Sources)

Use [download_dataset.py](download_dataset.py) to build a single YOLO dataset:

```bash
python download_dataset.py
```

To include Roboflow Indian dataset:

```bash
set ROBOFLOW_API_KEY=your_key_here
python download_dataset.py --roboflow-version 1
```

Output:
- data/license_plates/images/train|val|test
- data/license_plates/labels/train|val|test
- data/license_plates/data.yaml

Useful flags:
- --no-hf
- --no-roboflow
- --keep-temp

## Training

Run training via [train.py](train.py):

```bash
python train.py
```

Fine-tune behavior:
- If runs/detect/license_plate_detector/weights/best.pt exists, training auto-starts from it.
- Otherwise it starts from yolov8n.pt.

Override checkpoint explicitly:

```bash
python train.py --weights runs/detect/license_plate_detector/weights/best.pt
```

## Outputs

- output.mp4: first-pass annotated video
- results_raw.csv: raw per-frame tracking/OCR rows
- results.csv: interpolated/smoothed CSV
- final_output.mp4: optional polished render from smoothed CSV

## Project Files

- [pipeline.py](pipeline.py): main inference pipeline
- [util.py](util.py): OCR preprocessing, post-processing, helpers
- [sort.py](sort.py): tracker implementation
- [train.py](train.py): YOLO training/fine-tuning
- [download_dataset.py](download_dataset.py): dataset download and merge

## Architecture Reference

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed phase-by-phase internals and data flow.
