#!/usr/bin/env python3
"""
Step 4: Split YOLO dataset into train/validation sets.
Reads settings from data.yaml.
"""

from pathlib import Path
import random
import shutil
import yaml

# Load config from data.yaml
CONFIG_PATH = Path(__file__).resolve().parent.parent / "data.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / cfg["paths"]["dataset_dir"]
IMG_SRC = PROJECT_ROOT / cfg["paths"]["image_dir"]
LBL_SRC = PROJECT_ROOT / cfg["paths"]["yolo_labels_dir"]

TRAIN_RATIO = cfg["split"]["train_ratio"]
RANDOM_SEED = cfg["split"]["random_seed"]

TRAIN_IMG = DATASET_DIR / "images" / "train"
VAL_IMG = DATASET_DIR / "images" / "val"
TRAIN_LBL = DATASET_DIR / "labels" / "train"
VAL_LBL = DATASET_DIR / "labels" / "val"

# Create directories
for p in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
    p.mkdir(parents=True, exist_ok=True)

print(f"Splitting dataset (train: {TRAIN_RATIO*100:.0f}%, val: {(1-TRAIN_RATIO)*100:.0f}%)...")
print(f"Random seed: {RANDOM_SEED}")
print("-" * 50)

# Get all images with valid labels
images = sorted(IMG_SRC.glob("*.png"))
valid_images = [img for img in images if (LBL_SRC / f"{img.stem}.txt").exists()]

if not valid_images:
    print("ERROR: No images with labels found!")
    print(f"  - Image directory: {IMG_SRC}")
    print(f"  - Label directory: {LBL_SRC}")
    exit(1)

print(f"Found {len(valid_images)} images with labels")

# Shuffle and split
random.seed(RANDOM_SEED)
random.shuffle(valid_images)

split_idx = int(TRAIN_RATIO * len(valid_images))

train_count = 0
val_count = 0

for i, img_path in enumerate(valid_images):
    label_path = LBL_SRC / f"{img_path.stem}.txt"

    if i < split_idx:
        shutil.copy(img_path, TRAIN_IMG / img_path.name)
        shutil.copy(label_path, TRAIN_LBL / label_path.name)
        train_count += 1
    else:
        shutil.copy(img_path, VAL_IMG / img_path.name)
        shutil.copy(label_path, VAL_LBL / label_path.name)
        val_count += 1

print("-" * 50)
print(f"Dataset split complete!")
print(f"  Train samples: {train_count}")
print(f"  Validation samples: {val_count}")
print(f"  Dataset location: {DATASET_DIR}")
