#!/usr/bin/env python
"""
Region Cropping Module
Crops image regions from YOLO detections, preserving class information and coordinates
"""

import cv2
from pathlib import Path
from typing import List, Dict
import yaml

_cfg = yaml.safe_load(open(Path('config/config.yaml')))


def crop_regions(image_path, detections):
    """
    Crop image regions from YOLO detections

    Args:
        image_path: Path to input image
        detections: List of detection dicts with bbox, confidence, class, class_name

    Returns:
        List of crop dicts with:
            - crop: Cropped image array
            - bbox: Original bounding box [x0, y0, x1, y1]
            - confidence: Detection confidence
            - class: Class ID
            - class_name: Class name (e.g., "text", "title", "paragraph")
    """
    cfg = _cfg["cropping"]
    padding = cfg["padding"]

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = image.shape[:2]

    crops = []

    for d in detections:
        x0, y0, x1, y1 = d["bbox"]

        # Apply padding with boundary checks
        x0_padded = max(0, x0 - padding)
        y0_padded = max(0, y0 - padding)
        x1_padded = min(w, x1 + padding)
        y1_padded = min(h, y1 + padding)

        # Crop region
        crop = image[y0_padded:y1_padded, x0_padded:x1_padded]

        # Skip empty crops
        if crop.size == 0:
            continue

        crop_dict = {
            "bbox": d["bbox"],  # Original bbox (relative to original page)
            "confidence": d.get("confidence", 0.0),
            "crop": crop
        }

        # Add class information if available
        if "class" in d:
            crop_dict["class"] = d["class"]
        if "class_name" in d:
            crop_dict["class_name"] = d["class_name"]

        crops.append(crop_dict)

    return crops
