#!/usr/bin/env python
"""
Tesseract OCR Module
Performs OCR on cropped regions with class-specific PSM modes
"""

import pytesseract
from PIL import Image
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional
import yaml

_cfg = yaml.safe_load(open(Path('config/config.yaml')))


def get_psm_mode(class_name: Optional[str] = None, default_psm: int = None) -> int:
    """
    Get PSM mode for given class name

    Args:
        class_name: Layout class name (e.g., "title", "paragraph")
        default_psm: Default PSM mode if class not found

    Returns:
        PSM mode integer
    """
    cfg = _cfg["ocr"]
    class_psm_map = cfg.get("class_psm_map", {})
    default_psm = cfg.get("default_psm", 6)

    if class_name and class_name.lower() in class_psm_map:
        return class_psm_map[class_name.lower()]
    return default_psm


def recognize_text(crops: List[Dict], class_psm_map: Optional[Dict[str, int]] = None,
                   default_psm: int = None) -> List[Dict]:
    """
    Recognize text in cropped image regions using Tesseract OCR with class-specific PSM modes

    Args:
        crops: List of dicts with keys:
            - "crop": numpy array (image region)
            - "bbox": [x0, y0, x1, y1]
            - "confidence": float (YOLO detection confidence)
            - "class_name": str (optional, layout class name)
        class_psm_map: Optional custom PSM mapping (overrides config)
        default_psm: Default PSM mode (default: from config.yaml)

    Returns:
        List of dicts with keys:
            - "bbox": [x0, y0, x1, y1]
            - "yolo_conf": float (YOLO detection confidence)
            - "text": str (recognized text)
            - "ocr_conf": float (OCR confidence, 0-1)
            - "class_name": str (class name if available)
    """
    cfg = _cfg["ocr"]
    class_psm_map = cfg.get("class_psm_map", {})
    default_psm = cfg.get("default_psm", 6)

    results = []

    for c in crops:
        img = c["crop"]
        class_name = c["class_name"]

        # Skip OCR for image regions
        if class_name.lower() == "image":
            continue

        # Convert numpy array to PIL Image if needed
        if isinstance(img, np.ndarray):
            # Handle different image formats
            if len(img.shape) == 3:
                # BGR to RGB conversion for OpenCV images
                img_rgb = img[:, :, ::-1] if img.shape[2] == 3 else img
                pil_img = Image.fromarray(img_rgb)
            else:
                # Grayscale
                pil_img = Image.fromarray(img)
        else:
            pil_img = img

        # Get PSM mode for this class
        psm_mode = get_psm_mode(class_name, default_psm)

        # Run Tesseract OCR
        text = ""
        conf = 0.0

        try:
            # Get detailed data including confidence
            ocr_config = f'--psm {psm_mode}'
            ocr_data = pytesseract.image_to_data(
                pil_img,
                output_type=pytesseract.Output.DICT,
                config=ocr_config
            )

            # Extract text and confidence
            text_parts = []
            confidences = []

            for i in range(len(ocr_data['text'])):
                text_item = ocr_data['text'][i].strip()
                conf_item = int(ocr_data['conf'][i]) if ocr_data['conf'][i] != -1 else 0

                if text_item:  # Only include non-empty text
                    text_parts.append(text_item)
                    confidences.append(conf_item)

            # Combine text parts
            text = " ".join(text_parts) if text_parts else ""

            # Calculate average confidence (convert from 0-100 to 0-1 scale)
            conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

        except Exception as e:
            print(f"Tesseract OCR error for class '{class_name}': {e}")
            text = ""
            conf = 0.0

        result = {
            "bbox": c["bbox"],
            "yolo_conf": c.get("confidence", 0.0),
            "text": text,
            "ocr_conf": conf
        }

        # Preserve class information if available
        if "class" in c:
            result["class"] = c["class"]
        if "class_name" in c:
            result["class_name"] = c["class_name"]

        results.append(result)

    return results
