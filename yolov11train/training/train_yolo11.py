#!/usr/bin/env python3
"""
YOLOv11 training script optimized for T4 GPU (16GB VRAM) in Google Colab
Based on YOLOv8 training settings with adjustments for YOLOv11.
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml


if torch.cuda.is_available():
    device = "cuda:0"  # Use GPU if available
    print("Using GPU:", torch.cuda.get_device_name(0))  # Display the GPU name
else:
    device = "cpu"  # Use CPU if no GPU is available
    print("Warning: CUDA not available. Using CPU (training will be slow).")

# ---------------------------
# Load YOLOv11 model
# ---------------------------

model = YOLO("yolo11n.pt")

# ---------------------------
# Training parameters
# ---------------------------
print("\n" + "="*50)
print("Starting YOLOv11 Training")
print("="*50)

# Start training with the given parameters
results = model.train(
    data="yolov11train/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    workers=8,
    device=device
    amp=True,
    project="yolov11train/runs",
    name="docs_detection",
    exist_ok=True,
    verbose=True,
    
    # Augmentation settings

    hsv_h=0.015,                # Hue augmentation
    hsv_s=0.4,                  # Saturation augmentation
    hsv_v=0.3,                  # brightness augmentation
    degrees=3.0,                # Rotation augmentation
    translate=0.1,              # Horizontal/Vertical translation
    scale=0.3,                  # Image scaling
    shear=2.0,                  # Shear
    perspective=0.0001,         # Perspective transformation
    flipud=0.0,                 # No vertical flip
    fliplr=0.0,                 # No horizontal flip
    
    # Optimization parameters

    optimizer="AdamW",          # AdamW optimizer for better convergence
    lr0=0.001,                  # Initial learning rate
    lrf=0.01,                   # Learning rate final factor
    momentum=0.937,             # Momentum value for the optimizer
    weight_decay=0.0005,        # Weight decay (regularization to avoid overfitting)
    warmup_epochs=3.0,          # Number of warmup epochs to start with a lower learning rate
    
    # Saving model checkpoints and metrics

    save=True,                  # Save the model checkpoints during training
    save_period=10,             # Save checkpoints every 10 epochs
    plots=True,                 # Generate plots for training progress (loss, mAP, etc.)
)

# ---------------------------
# Completion Message
# ---------------------------

print("\n" + "="*50)
print("Training complete!")
print("="*50)

# ---------------------------
# Results summary
# ---------------------------
# Display the results and metrics after training completes

print(f"\nResults saved to: yolov11train/runs/docs_detection/")
print("\nMetrics summary:")
print(f"  - mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"  - mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
print(f"  - Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
print(f"  - Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
