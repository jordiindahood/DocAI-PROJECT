#!/usr/bin/env python
"""
Merge Results Utility
Merges YOLO and OCR results into unified structure
"""

from typing import List, Dict, Optional
import numpy as np


def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        bbox1: [x0, y0, x1, y1]
        bbox2: [x0, y0, x1, y1]
        
    Returns:
        IoU score (0-1)
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def merge_boxes(detections: List[Dict]) -> Dict:
    """
    Merge multiple overlapping YOLO detections into a single detection.
    
    Args:
        detections: List of YOLO detections with bbox, confidence, class, class_name
            Each detection should be: {
                "bbox": [x0, y0, x1, y1],
                "confidence": float,
                "class": int,
                "class_name": str
            }
    
    Returns:
        Merged detection with averaged bbox (weighted by confidence) and max confidence
    """
    if not detections:
        return None
    
    if len(detections) == 1:
        return detections[0]
    
    # Use confidence-weighted average for bbox
    total_weight = sum(d["confidence"] for d in detections)
    
    avg_bbox = [0.0, 0.0, 0.0, 0.0]
    for det in detections:
        weight = det["confidence"] / total_weight
        for i in range(4):
            avg_bbox[i] += det["bbox"][i] * weight
    
    # Find detection with highest confidence
    best_det = max(detections, key=lambda d: d["confidence"])
    
    return {
        "bbox": avg_bbox,
        "confidence": best_det["confidence"],
        "class": best_det["class"],
        "class_name": best_det["class_name"]
    }


def merge_results(ocr_results: List[Dict], layout_results: Optional[Dict] = None) -> List[Dict]:
    """
    Merge OCR results (LayoutLM removed - kept for backward compatibility)
    
    Args:
        ocr_results: List of OCR results with bbox, text, ocr_conf
        layout_results: Ignored (kept for backward compatibility)
        
    Returns:
        List of OCR results with default layout info
    """
    merged = []
    
    # Return OCR results with default layout info (no LayoutLM processing)
    for ocr_item in ocr_results:
        merged.append({
            "bbox": ocr_item.get("bbox", []),
            "text": ocr_item.get("text", ""),
            "ocr_conf": ocr_item.get("ocr_conf", 0.0),
            "yolo_conf": ocr_item.get("yolo_conf", 0.0),
            "layout_label": "other",
            "field_label": "other",
            "layout_conf": 0.0
        })
    
    return merged


def create_unified_structure(merged_results: List[Dict], fields: Optional[Dict] = None, img_width: int = None, img_height: int = None) -> Dict:
    """
    Create unified structure with all document information
    
    Args:
        merged_results: Merged OCR and layout results
        fields: Extracted structured fields (optional, can be None)
        img_width: Original image width in pixels
        img_height: Original image height in pixels
        
    Returns:
        Unified document structure
    """
    # Generate raw_text from regions if fields not provided or raw_text missing
    raw_text = None
    if fields and fields.get("raw_text"):
        raw_text = fields.get("raw_text")
    else:
        # Extract text from all regions
        text_parts = [r.get("text", "").strip() for r in merged_results if r.get("text", "").strip()]
        raw_text = " ".join(text_parts) if text_parts else ""
    
    # Create fields dict with at least raw_text
    fields_dict = fields if fields else {}
    if "raw_text" not in fields_dict:
        fields_dict = fields_dict.copy() if fields_dict else {}
        fields_dict["raw_text"] = raw_text
    
    return {
        "document_type": "document",  # Generic document type (field extraction removed)
        "fields": fields_dict,  # Only contains raw_text now
        "regions": merged_results,
        "image_dimensions": {
            "width": img_width,
            "height": img_height
        },
        "statistics": {
            "total_regions": len(merged_results),
            "header_regions": len([r for r in merged_results if r.get("layout_label") == "header"]),
            "paragraph_regions": len([r for r in merged_results if r.get("layout_label") == "paragraph"]),
            "table_regions": len([r for r in merged_results if r.get("layout_label") == "table"]),
            "avg_ocr_confidence": np.mean([r.get("ocr_conf", 0.0) for r in merged_results]) if merged_results else 0.0,
            "avg_layout_confidence": np.mean([r.get("layout_conf", 0.0) for r in merged_results]) if merged_results else 0.0
        }
    }


def create_pure_layout_structure(ocr_results: List[Dict], img_width: int, img_height: int, page_number: int = 1) -> Dict:
    """
    Create pure layout structure matching the prompt schema.
    NO semantic understanding, NO field classification, NO document type inference.
    Only text, bbox, page, and order.
    
    Args:
        ocr_results: List of OCR results with bbox and text
        img_width: Image width in pixels
        img_height: Image height in pixels
        page_number: Page number (default: 1)
        
    Returns:
        Pure layout structure matching the prompt schema:
        {
            "page_width": number,
            "page_height": number,
            "texts": [
                {
                    "text": string,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "page": number,
                    "order": number
                }
            ]
        }
    """
    texts = []

    # Sort by reading order (already sorted, but ensure order numbers)
    ocr_results = deduplicate_ocr_results(ocr_results)

    for idx, ocr_item in enumerate(ocr_results):
        text = ocr_item.get("text", "").strip()
        bbox = ocr_item.get("bbox", [])
        
        if not text or len(bbox) < 4:
            continue
        
        # Ensure bbox is in correct format [x_min, y_min, x_max, y_max]
        x0, y0, x1, y1 = bbox[:4]
        
        texts.append({
            "text": text,
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "page": page_number,
            "order": idx + 1  # 1-indexed reading order
        })
    
    return {
        "page_width": int(img_width),
        "page_height": int(img_height),
        "texts": texts
    }

def deduplicate_ocr_results(results, iou_threshold=0.95):
    """
    Remove duplicate OCR results based on text + bbox IoU
    """
    unique = []

    for r in results:
        text = r.get("text", "").strip()
        bbox = r.get("bbox", [])

        if not text or len(bbox) < 4:
            continue

        is_duplicate = False
        for u in unique:
            if (
                text == u["text"] and
                calculate_iou(bbox, u["bbox"]) > iou_threshold
            ):
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(r)

    return unique
