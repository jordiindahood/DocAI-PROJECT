#!/usr/bin/env python3
"""
Step 5: Apply data augmentation to the YOLO dataset.
Reads settings from data.yaml — no command-line arguments needed.
"""

from pathlib import Path
import numpy as np
import yaml

from augmentation_utils import (
    read_image_bgr,
    write_image_png,
    load_yolo_labels,
    transform_labels,
    write_labels,
)
from augmentations import apply_augmentations


# Load config from data.yaml
CONFIG_PATH = Path(__file__).resolve().parent.parent / "data.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / cfg["paths"]["dataset_dir"]

aug_cfg = cfg["augmentation"]
COPIES = aug_cfg["copies"]
SEED = aug_cfg["seed"]
SPLITS = aug_cfg["splits"]


def iter_images(images_dir: Path):
    """Yield all image files in a directory."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for p in sorted(images_dir.rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            yield p


def default_label_path(img_path: Path, images_root: Path, labels_root: Path):
    """Map an image path to its corresponding label path."""
    rel = img_path.relative_to(images_root)
    return labels_root / rel.with_suffix(".txt")


def augment_split(split_name: str, seed: int):
    """Augment all images in a single split (train or val)."""
    src_images = DATASET_DIR / "images" / split_name
    src_labels = DATASET_DIR / "labels" / split_name

    if not src_images.exists():
        raise FileNotFoundError(f"Missing: {src_images}")
    if not src_labels.exists():
        raise FileNotFoundError(f"Missing: {src_labels}")

    rng = np.random.default_rng(seed)
    imgs = list(iter_images(src_images))

    print(f"  Found {len(imgs)} images in {split_name}")

    for img_path in imgs:
        img = read_image_bgr(img_path)
        h, w = img.shape[:2]
        label_path = default_label_path(img_path, src_images, src_labels)
        labels = load_yolo_labels(label_path)

        for k in range(COPIES):
            img_aug, H = apply_augmentations(img)
            labels_aug = transform_labels(labels, H, w, h)

            stem = img_path.stem
            aug_img_name = f"{stem}_aug{k+1:02d}.png"
            aug_lbl_name = f"{stem}_aug{k+1:02d}.txt"

            write_image_png(src_images / aug_img_name, img_aug)
            write_labels(src_labels / aug_lbl_name, labels_aug)

    print(f"  Created {len(imgs) * COPIES} augmented images for {split_name}")


def main():
    print("Applying augmentations...")
    print(f"  Copies per image: {COPIES}")
    print(f"  Seed: {SEED}")
    print(f"  Splits: {SPLITS}")
    print("-" * 50)

    for i, split in enumerate(SPLITS):
        split_seed = SEED + (i * 10000)
        print(f"\nProcessing split: {split}")
        augment_split(split, split_seed)

    print("-" * 50)
    print("Augmentation complete!")


if __name__ == "__main__":
    main()
