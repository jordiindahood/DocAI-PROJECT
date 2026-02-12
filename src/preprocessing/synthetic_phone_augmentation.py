#!/usr/bin/env python

import sys
import cv2
import random
import numpy as np
from pathlib import Path


from .augmentation_utils import parse_yolo_labels, save_yolo_labels
from .augmentations import augment_image_and_labels


def process_split(img_dir, lbl_dir, out_img_dir, out_lbl_dir, ratio, split_name):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    to_augment = set(random.sample(images, int(len(images) * ratio)))

    stats = {"processed": 0, "augmented": 0, "skipped": 0, "errors": 0}

    for idx, img_path in enumerate(images, start=1):
        print(f"\n[{split_name}] Image {idx}/{len(images)} → {img_path.name}")

        image = cv2.imread(str(img_path))
        if image is None:
            print("  ❌ Failed to read image")
            stats["errors"] += 1
            continue

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        labels = parse_yolo_labels(str(lbl_path))

        if img_path in to_augment:
            print("  ✅ Augmenting image")
            try:
                aug_img, aug_labels = augment_image_and_labels(image, labels)

                out_img = out_img_dir / f"aug_{img_path.name}"
                out_lbl = out_lbl_dir / f"aug_{img_path.stem}.txt"

                cv2.imwrite(str(out_img), aug_img)
                save_yolo_labels(aug_labels, str(out_lbl))

                print(f"  💾 Saved → {out_img.name}")
                stats["augmented"] += 1
            except Exception as e:
                print(f"  ❌ Augmentation error: {e}")
                stats["errors"] += 1
        else:
            print("  ⏭️  Skipped (kept clean)")
            stats["skipped"] += 1

        stats["processed"] += 1

    return stats


def process_dataset(input_dir, output_dir, ratio, seed):
    random.seed(seed)
    np.random.seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    img_root = input_path / "images"
    lbl_root = input_path / "labels"

    splits = ["train", "val"] if (img_root / "train").exists() else [None]
    total = {"processed": 0, "augmented": 0, "skipped": 0, "errors": 0}

    for split in splits:
        img_dir = img_root / split if split else img_root
        lbl_dir = lbl_root / split if split else lbl_root
        out_img = output_path / "images" / split if split else output_path / "images"
        out_lbl = output_path / "labels" / split if split else output_path / "labels"

        stats = process_split(img_dir, lbl_dir, out_img, out_lbl, ratio, split or "all")
        for k in total:
            total[k] += stats[k]

    print("\n✅ DONE")
    print(total)


def main():
    if len(sys.argv) < 3:
        print("Usage: python synthetic_phone_augmentation.py <input> <output> [ratio] [seed]")
        return 1

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    print("=" * 60)
    print("Synthetic Phone Augmentation")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Ratio:  {ratio * 100:.0f}%")
    print(f"Seed:   {seed}")
    print("=" * 60)

    process_dataset(input_dir, output_dir, ratio, seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
