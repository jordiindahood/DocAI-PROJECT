#!/usr/bin/env python3
"""
Automate the entire YOLOv11 training pipeline:
1. PDF -> Image conversion
2. Annotation extraction
3. YOLO format conversion
4. Dataset splitting
5. Data augmentation
6. Model training

All settings are in data.yaml — scripts require no arguments.
"""

import subprocess
import sys
from pathlib import Path
import os

# Ensure we operate from the project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def run_step(script_path, desc):
    print(f"\n{'='*50}")
    print(f"Step: {desc}")
    print(f"Script: {script_path}")
    print(f"{'='*50}\n")

    try:
        cmd = [sys.executable, str(script_path)]
        print(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        print(f"\n {desc} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n Error in {desc}. Exit code: {e.returncode}")
        print(f"Failed script: {script_path}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n Unexpected error in {desc}: {e}")
        sys.exit(1)


def main():
    os.chdir(PROJECT_ROOT)
    print(f"Working directory set to: {os.getcwd()}")

    # All scripts read their config from data.yaml — no args needed
    steps = [
        #("yolov11train/preprocessing/1-pdf2img.py", "Convert PDFs to Images"),
        ("yolov11train/preprocessing/2-extract_txt_and_bounding_boxes.py", "Extract Annotations"),
        ("yolov11train/preprocessing/3-annotations_2_yolo_format.py", "Convert to YOLO Format"),
        ("yolov11train/preprocessing/4-split_yolo_dataset.py", "Split Dataset"),
        ("yolov11train/augmentation/apply_augmentations.py", "Apply Augmentations"),
        # ("yolov11train/training/train_yolo11.py", "Train YOLOv11 Model"),
    ]

    print("Starting YOLOv11 Training Pipeline...")

    for script, desc in steps:
        if not Path(script).exists():
            print(f"Error: Script {script} not found!")
            sys.exit(1)
        run_step(script, desc)

    print(f"\n{'='*50}")
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
