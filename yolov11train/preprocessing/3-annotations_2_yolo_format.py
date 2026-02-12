#!/usr/bin/env python3
"""
Step 3: Convert annotations to YOLO format.
Output format: class_id x_center y_center width height (normalized 0-1)
Reads settings from data.yaml.
"""

from PIL import Image
import json
from pathlib import Path
import yaml

# Load config from data.yaml
CONFIG_PATH = Path(__file__).resolve().parent.parent / "data.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMG_DIR = PROJECT_ROOT / cfg["paths"]["image_dir"]
ANN_DIR = PROJECT_ROOT / cfg["paths"]["annotation_dir"]
YOLO_LABELS = PROJECT_ROOT / cfg["paths"]["yolo_labels_dir"]

YOLO_LABELS.mkdir(exist_ok=True)

CLASS_ID = 0  # Single class: text (nc=1 in data.yaml)

print("Converting annotations to YOLO format...")
print("-" * 50)

total_boxes = 0
for img_path in sorted(IMG_DIR.glob("*.png")):
    # Parse image name to get PDF name and page index
    parts = img_path.stem.split("_page")
    if len(parts) != 2:
        print(f"  Unexpected filename format: {img_path.name}")
        continue

    pdf_name = parts[0]
    page_idx = int(parts[1]) - 1

    ann_file = ANN_DIR / f"{pdf_name}.json"
    if not ann_file.exists():
        print(f"  Annotation file missing: {ann_file.name}")
        continue

    # Load image size
    img = Image.open(img_path)
    W, H = img.size

    # Load page annotations
    with open(ann_file) as f:
        data = json.load(f)

    if page_idx >= len(data):
        print(f"  Page index {page_idx} out of range in {ann_file.name}")
        continue

    labels = []
    for item in data[page_idx]:
        x0, y0, x1, y1 = item["bbox"]

        # Normalize coordinates for YOLO (0-1 range)
        xc = ((x0 + x1) / 2) / W
        yc = ((y0 + y1) / 2) / H
        w = (x1 - x0) / W
        h = (y1 - y0) / H

        # Skip invalid boxes
        if w <= 0 or h <= 0:
            continue

        # Clamp to valid range
        xc = max(0, min(1, xc))
        yc = max(0, min(1, yc))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        labels.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    # Write YOLO .txt label file
    out_file = YOLO_LABELS / f"{img_path.stem}.txt"
    out_file.write_text("\n".join(labels))

    total_boxes += len(labels)
    print(f"  {img_path.stem}: {len(labels)} boxes")

print("-" * 50)
print(f"Total YOLO annotations: {total_boxes}")
print(f"Output directory: {YOLO_LABELS}")
