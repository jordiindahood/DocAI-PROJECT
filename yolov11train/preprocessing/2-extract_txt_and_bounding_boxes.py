#!/usr/bin/env python3
"""
Step 2: Extract text and bounding boxes from PDFs.
Uses PyMuPDF to extract word-level bounding boxes.
Reads settings from data.yaml.
"""

import fitz  # PyMuPDF
import json
from pathlib import Path
from PIL import Image
import yaml

# Load config from data.yaml
CONFIG_PATH = Path(__file__).resolve().parent.parent / "data.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PDF_DIR = PROJECT_ROOT / cfg["paths"]["pdf_dir"]
IMG_DIR = PROJECT_ROOT / cfg["paths"]["image_dir"]
OUT_DIR = PROJECT_ROOT / cfg["paths"]["annotation_dir"]

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Extracting annotations from {PDF_DIR}...")
print("-" * 50)

for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
    doc = fitz.open(pdf_path)
    all_pages = []

    for page_num, page in enumerate(doc):
        words = page.get_text("words")

        # Get corresponding image size
        img_file = IMG_DIR / f"{pdf_path.stem}_page{page_num+1:04d}.png"
        if not img_file.exists():
            print(f"Image not found: {img_file.name}, skipping page")
            continue

        img = Image.open(img_file)
        img_w, img_h = img.size

        # PDF page size
        pdf_w = page.rect.width
        pdf_h = page.rect.height

        # Scale factors
        scale_x = img_w / pdf_w
        scale_y = img_h / pdf_h

        page_data = []
        for w in words:
            x0, y0, x1, y1, text = w[:5]

            # Scale to image coordinates
            x0_img = x0 * scale_x
            y0_img = y0 * scale_y
            x1_img = x1 * scale_x
            y1_img = y1 * scale_y

            page_data.append({
                "text": text,
                "bbox": [x0_img, y0_img, x1_img, y1_img],
                "page": page_num
            })

        all_pages.append(page_data)
        print(f"  {pdf_path.stem}_page{page_num+1}: {len(page_data)} words")

    # Save annotations
    out_file = OUT_DIR / f"{pdf_path.stem}.json"
    with open(out_file, "w") as f:
        json.dump(all_pages, f, indent=2)

    doc.close()
    print(f"  -> Saved: {out_file.name}")

print("-" * 50)
print(f"Annotations saved to: {OUT_DIR}")
