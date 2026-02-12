"""
OCR Sanity Filtering
STEP 6 of the layout reconstruction pipeline.

Removes OCR noise:
- very short junk
- punctuation-heavy strings
- low-confidence OCR
- repeated characters
"""

import re
from pathlib import Path
from typing import List, Dict
import yaml

_cfg = yaml.safe_load(open(Path('config/config.yaml')))


def clean_ocr_results(ocr_results: List[Dict]) -> List[Dict]:
    """
    Clean OCR results before layout reconstruction.

    Input:
        [
          {
            "text": str,
            "bbox": [x0, y0, x1, y1],
            "ocr_conf": float,
            ...
          }
        ]

    Output:
        same structure, but filtered & cleaned
    """
    cfg = _cfg["ocr_filter"]
    min_len = cfg.get("min_text_length", 2)
    min_conf = cfg.get("min_confidence", 0.35)
    min_alnum = cfg.get("min_alnum_ratio", 0.5)

    cleaned = []

    for r in ocr_results:
        text = r.get("text", "").strip()
        conf = r.get("ocr_conf", 0.0)

        # 1. Drop empty or ultra-short junk
        if len(text) < min_len:
            continue

        # 2. Drop very low confidence OCR
        if conf and conf < min_conf:
            continue

        # 3. Drop punctuation-heavy strings
        alnum_count = sum(c.isalnum() for c in text)
        if alnum_count / max(len(text), 1) < min_alnum:
            continue

        # 4. Collapse repeated characters (----, ____ , IIII)
        text = re.sub(r"(.)\1{2,}", r"\1", text)

        text = text.strip()
        if not text:
            continue

        r["text"] = text
        cleaned.append(r)

    return cleaned
