#!/usr/bin/env python

import os
import sys
import json
import time
import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# ------------------------------------------------------------------
# Add project root to path
# ------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ------------------------------------------------------------------
# Imports (existing modules only)
# ------------------------------------------------------------------
run_yolo = __import__('detection.run_yolo', fromlist=['run_yolo']).run_yolo
filter_boxes = __import__('detection.filter_boxes', fromlist=['filter_boxes']).filter_boxes

normalize_document = __import__('preprocessing.document_normalization', fromlist=['normalize_document']).normalize_document
crop_regions = __import__('ocr.crop_text', fromlist=['crop_regions']).crop_regions
recognize_text = __import__('ocr.run_ocr', fromlist=['recognize_text']).recognize_text

sort_reading_order = __import__('utils.sort_boxes', fromlist=['sort_reading_order']).sort_reading_order
create_pure_layout_structure = __import__('utils.merge_results', fromlist=['create_pure_layout_structure']).create_pure_layout_structure

ExportManager = __import__('export.export_manager', fromlist=['ExportManager']).ExportManager

clean_ocr_results = __import__('ocr.ocr_sanity_filter', fromlist=['clean_ocr_results']).clean_ocr_results
reconstruct_document_layout = __import__('reconstruction.layout_reconstructor', fromlist=['reconstruct_document_layout']).reconstruct_document_layout

def process_document(image_path, output_dir="outputs"):
    """
    Run full layout reconstruction pipeline on a single image.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory for output files
        
    Returns:
        Dictionary with layout_structure, image_path, and json_path
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ------------------------------------------------------------------
    # Load image info
    # ------------------------------------------------------------------
    img = Image.open(image_path)
    img_width, img_height = img.size
    image_name = os.path.basename(image_path)
    image_stem = os.path.splitext(image_name)[0]

    print("\n" + "=" * 60)
    print(f"Pure Layout Reconstruction: {image_name}")
    print(f"Image size: {img_width} x {img_height}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # STEP 1 — Document normalization
    # ------------------------------------------------------------------
    print("Step 1: Document normalization...")
    try:
        normalized_image_path = normalize_document(image_path)
    except Exception:
        normalized_image_path = image_path

    # ------------------------------------------------------------------
    # STEP 2 — YOLO layout detection
    # ------------------------------------------------------------------
    print("Step 2: YOLO layout detection...")
    detections = run_yolo(normalized_image_path)
    print(f"  Detected {len(detections)} regions")

    # ------------------------------------------------------------------
    # STEP 3 — Box filtering & deduplication
    # ------------------------------------------------------------------
    print("Step 3: Filtering & deduplicating boxes...")
    filtered_boxes = filter_boxes(detections)
    print(f"  Remaining {len(filtered_boxes)} regions")

    # ------------------------------------------------------------------
    # STEP 4 — Crop regions
    # ------------------------------------------------------------------
    print("Step 4: Cropping regions...")
    crops = crop_regions(normalized_image_path, filtered_boxes)
    print(f"  Cropped {len(crops)} regions")

    # ------------------------------------------------------------------
    # STEP 5 — OCR (region-level)
    # ------------------------------------------------------------------
    print("Step 5: OCR...")
    ocr_results = recognize_text(crops)
    ocr_results = [r for r in ocr_results if r.get("text", "").strip()]
    print(f"  OCR produced {len(ocr_results)} text blocks")

    # ------------------------------------------------------------------
    # STEP 6 — OCR sanity filtering
    # ------------------------------------------------------------------
    print("Step 6: OCR cleanup...")
    ocr_results = clean_ocr_results(ocr_results)
    print(f"  {len(ocr_results)} regions after OCR cleanup")

    # ------------------------------------------------------------------
    # STEP 7 — Reading order sorting
    # ------------------------------------------------------------------
    print("Step 7: Sorting by reading order...")
    sorted_results = sort_reading_order(ocr_results)

    # ------------------------------------------------------------------
    # STEP 7.5 — Document Layout Reconstruction
    # ------------------------------------------------------------------
    print("Step 7.5: Reconstructing document layout...")
    reconstructed_text, recon_stats = reconstruct_document_layout(
        sorted_results,
        img_width,
        img_height
    )
    print(f"  Input: {recon_stats['input_blocks']} blocks")
    print(f"  Merged: {recon_stats['merged_count']} duplicates")
    print(f"  Output: {recon_stats['output_blocks']} blocks in {recon_stats['line_count']} lines")

    # ------------------------------------------------------------------
    # STEP 8 — Create pure layout structure
    # ------------------------------------------------------------------
    print("Step 8: Creating layout structure...")
    layout_structure = create_pure_layout_structure(
        sorted_results,
        img_width,
        img_height
    )
    
    # Add reconstructed text to layout structure
    layout_structure['reconstructed_text'] = reconstructed_text

    # ------------------------------------------------------------------
    # Save JSON output
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{image_stem}_pure_layout.json")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(layout_structure, f, indent=2, ensure_ascii=False)

    print(f"✓ Layout JSON saved: {json_path}")

    return {
        "layout_structure": layout_structure,
        "image_path": image_path,
        "json_path": json_path,
    }


def process_and_export(image_path, output_format="pdf", output_dir="outputs"):
    """
    Run pipeline and export reconstructed document.
    """
    results = process_document(image_path, output_dir)

    print("\n" + "=" * 60)
    print("Exporting reconstructed document...")
    print("=" * 60)

    export_manager = ExportManager(output_dir=output_dir)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    formats = ["pdf"]

    export_paths = export_manager.export_pure_layout(
        data=results["layout_structure"],
        base_name=base_name,
        formats=formats
    )

    results["export_paths"] = export_paths
    print(f"✓ Export complete: {export_paths}")

    return results


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python layout_reconstruction_pipeline.py <image_path> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"

    process_and_export(image_path, "pdf", output_dir)
