#!/usr/bin/env python
"""
Sort Boxes Utility
Sorts text boxes in natural reading order
"""

from typing import List, Dict


def sort_reading_order(ocr_results: List[Dict]) -> List[Dict]:
    """
    Sort OCR results in natural reading order (top-to-bottom, left-to-right).
    
    Uses a row-based approach:
    1. Group boxes into horizontal rows based on vertical overlap
    2. Sort rows from top to bottom
    3. Within each row, sort boxes from left to right
    
    Args:
        ocr_results: List of OCR results with bbox [x0, y0, x1, y1]
    
    Returns:
        Sorted list of OCR results in reading order
    """
    if not ocr_results:
        return []
    
    # Create a copy to avoid modifying the original
    results = ocr_results.copy()
    
    # Sort by y-coordinate first to group into rows
    # Use top of bounding box (y0) as primary sort key
    # Use left edge (x0) as secondary sort key for boxes at same vertical position
    results.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    
    # Group into rows based on vertical overlap
    rows = []
    current_row = [results[0]]
    
    for i in range(1, len(results)):
        current = results[i]
        prev = results[i - 1]
        
        # Calculate vertical overlap
        curr_y0, curr_y1 = current["bbox"][1], current["bbox"][3]
        prev_y0, prev_y1 = prev["bbox"][1], prev["bbox"][3]
        
        # Check if boxes are on the same row (vertical overlap)
        overlap_y_min = max(curr_y0, prev_y0)
        overlap_y_max = min(curr_y1, prev_y1)
        
        # Calculate heights
        curr_height = curr_y1 - curr_y0
        prev_height = prev_y1 - prev_y0
        avg_height = (curr_height + prev_height) / 2
        
        # If vertical overlap is significant or vertical gap is small, same row
        if overlap_y_max > overlap_y_min:
            # Has overlap, same row
            current_row.append(current)
        elif abs(curr_y0 - prev_y0) < avg_height * 0.5:
            # Small vertical distance, likely same row
            current_row.append(current)
        else:
            # New row
            rows.append(current_row)
            current_row = [current]
    
    # Add the last row
    if current_row:
        rows.append(current_row)
    
    # Sort each row by x-coordinate (left to right)
    for row in rows:
        row.sort(key=lambda r: r["bbox"][0])
    
    # Flatten rows back into a single list
    sorted_results = []
    for row in rows:
        sorted_results.extend(row)
    
    return sorted_results
