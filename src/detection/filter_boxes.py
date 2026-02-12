#!/usr/bin/env python

from pathlib import Path
import yaml

_cfg = yaml.safe_load(open(Path('config/config.yaml')))

calculate_iou = __import__('utils.collision_detector', fromlist=['calculate_iou']).calculate_iou

merge_boxes = __import__('utils.merge_results', fromlist=['merge_boxes']).merge_boxes

def filter_boxes(detections):
    cfg = _cfg["filtering"]

    min_conf = cfg["min_confidence"]
    min_area = cfg["min_area"]

    clean = []

    for d in detections:
        x0, y0, x1, y1 = d["bbox"]
        area = (x1 - x0) * (y1 - y0)

        if d["confidence"] >= min_conf and area >= min_area:
            clean.append(d)

    return clean

def deduplicate_yolo_boxes(
    detections,
    iou_threshold=0.4
):
    """
    Deduplicate overlapping YOLO detections.

    Input:
      detections: list of YOLO detections
        {
          "bbox": [x1, y1, x2, y2],
          "confidence": float,
          "class": int,
          "class_name": str
        }

    Output:
      list of merged detections (same format)
    """

    if not detections:
        return []

    used = [False] * len(detections)
    final = []

    for i, det in enumerate(detections):
        if used[i]:
            continue

        group = [det]
        used[i] = True

        for j in range(i + 1, len(detections)):
            if used[j]:
                continue

            if detections[j]["class"] != det["class"]:
                continue

            iou = calculate_iou(det["bbox"], detections[j]["bbox"])
            if iou >= iou_threshold:
                group.append(detections[j])
                used[j] = True

        final.append(merge_boxes(group))

    return final
