#!/usr/bin/env python3
"""
YOLOv11 training script (accuracy-first but T4-safe) for Google Colab.
"""

from ultralytics import YOLO
import torch

# ---------------------------
# GPU Device check/selection
# ---------------------------

# Require a GPU and require >= 15 GB VRAM (e.g., T4 16GB)
if not torch.cuda.is_available():
    raise SystemExit("GPU required. CUDA not available.")

props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / (1024**3)

if vram_gb < 15:
    raise SystemExit(f"Need >=15GB VRAM. Found {vram_gb:.2f}GB on {props.name}.")

device = "cuda:0"
print(f"Using GPU: {props.name} ({vram_gb:.2f} GB)")

# ---------------------------
# Load YOLOv11 model
# ---------------------------

# Start with "s" for better accuracy; if you still OOM, switch to "yolo11n.pt"

model = YOLO("yolo11s.pt")

print("\n" + "=" * 50)
print("Starting YOLOv11 Training (Accuracy-First, T4-Safe)")
print("=" * 50)

# ---------------------------
# Train
# ---------------------------
results = model.train(
    # Dataset config (paths + class names)
    data="yolov11train/data.yaml",

    # Train longer
    epochs=200,
    imgsz=640,
    batch=4,

    workers=2,

    # Train on GPU/CPU
    device=device,

    # Mixed precision speeds up training on GPU and usually keeps accuracy
    amp=True,

    cache=True,

    # Disable multi-scale for stability (multi-scale can spike VRAM)
    multi_scale=False,

    # Experiment tracking output folder
    project="yolov11train/runs",
    name="docs_detection_acc_t4safe",
    exist_ok=True,
    verbose=True,

    # ---------------------------
    # Optimizer / LR
    # ---------------------------
    optimizer="AdamW",
    lr0=0.0015,
    lrf=0.01,   
    weight_decay=0.0005,
    warmup_epochs=3.0,

    # ---------------------------
    # Augmentations
    # ---------------------------
    hsv_h=0.01,
    hsv_s=0.25,
    hsv_v=0.20,

    degrees=1.0,
    translate=0.05,
    scale=0.15,
    shear=0.1,
    perspective=0.0,

    flipud=0.0,
    fliplr=0.0,

    # Keep mosaic low; too high can hurt layout tasks and adds instability
    mosaic=0.2,
    mixup=0.0,
    copy_paste=0.0,

    # ---------------------------
    # Saving / plots
    # ---------------------------
    save=True,
    save_period=10,
    plots=True,
)

print("\n" + "=" * 50)
print("Training complete!")
print("=" * 50)

# ---------------------------
# Quick results summary
# ---------------------------
print("\nResults saved to: yolov11train/runs/docs_detection_acc_t4safe/")
print("\nMetrics summary:")
print(f"  - mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"  - mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
print(f"  - Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
print(f"  - Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
    