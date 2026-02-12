#!/usr/bin/env python
"""
YOLOv11 Layout Detection
"""

from pathlib import Path
from ultralytics import YOLO
import yaml

cfg = yaml.safe_load(open(Path('config/config.yaml')))["detection"]

model = YOLO(cfg["model_path"])


def run_yolo(image_path):
    """Run YOLOv11 on an image, return list of detections."""

    results = model(
        image_path,
        imgsz=cfg.get("image_size", 1280),
        conf=cfg.get("confidence_threshold", 0.4),
        device=cfg.get("device", "cpu"),
        verbose=False,
    )

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "bbox": list(map(int, box.xyxy[0])),
                "confidence": float(box.conf[0]),
                "class": cls_id,
                "class_name": model.names.get(cls_id, "unknown"),
            })

    return detections
