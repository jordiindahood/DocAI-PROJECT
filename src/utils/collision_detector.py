#!/usr/bin/env python
"""
Collision Detection System
Detects and resolves overlapping text blocks in document reconstruction
"""

from typing import List, Tuple, Dict
from .spatial_mapping import TextBlock
import math


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes
    
    Args:
        bbox1: [x0, y0, x1, y1]
        bbox2: [x0, y0, x1, y1]
        
    Returns:
        IoU value (0.0 to 1.0)
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2
    
    # Calculate intersection
    x0_inter = max(x0_1, x0_2)
    y0_inter = max(y0_1, y0_2)
    x1_inter = min(x1_1, x1_2)
    y1_inter = min(y1_1, y1_2)
    
    if x1_inter <= x0_inter or y1_inter <= y0_inter:
        return 0.0
    
    inter_area = (x1_inter - x0_inter) * (y1_inter - y0_inter)
    
    # Calculate union
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def calculate_overlap(bbox1: List[float], bbox2: List[float]) -> Tuple[float, float]:
    """
    Calculate overlap amounts (horizontal and vertical)
    
    Args:
        bbox1: [x0, y0, x1, y1]
        bbox2: [x0, y0, x1, y1]
        
    Returns:
        (horizontal_overlap, vertical_overlap) in target space units
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2
    
    # Horizontal overlap
    x_overlap = max(0, min(x1_1, x1_2) - max(x0_1, x0_2))
    
    # Vertical overlap
    y_overlap = max(0, min(y1_1, y1_2) - max(y0_1, y0_2))
    
    return x_overlap, y_overlap


class CollisionDetector:
    """
    Detects and resolves overlapping text blocks
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.1,
        min_spacing: float = 2.0,
        adjustment_strategy: str = "vertical_first"
    ):
        """
        Initialize collision detector
        
        Args:
            iou_threshold: IoU threshold for considering blocks as overlapping (0.0-1.0)
            min_spacing: Minimum spacing between blocks in target space units
            adjustment_strategy: "vertical_first", "horizontal_first", or "priority"
        """
        self.iou_threshold = iou_threshold
        self.min_spacing = min_spacing
        self.adjustment_strategy = adjustment_strategy
    
    def detect_overlaps(self, blocks: List[TextBlock]) -> List[Tuple[int, int, float]]:
        """
        Detect overlapping text blocks
        
        Args:
            blocks: List of TextBlock instances
            
        Returns:
            List of (index1, index2, iou) tuples for overlapping blocks
        """
        overlaps = []
        
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                iou = calculate_iou(blocks[i].bbox, blocks[j].bbox)
                if iou > self.iou_threshold:
                    overlaps.append((i, j, iou))
        
        # Sort by IoU (highest first)
        overlaps.sort(key=lambda x: x[2], reverse=True)
        
        return overlaps
    
    def adjust_position_vertical(
        self,
        block: TextBlock,
        reference_block: TextBlock,
        min_spacing: float
    ) -> TextBlock:
        """
        Adjust block position vertically to avoid overlap
        
        Args:
            block: Block to adjust
            reference_block: Reference block (already placed)
            min_spacing: Minimum spacing required
            
        Returns:
            Adjusted TextBlock
        """
        x0, y0, x1, y1 = block.bbox
        ref_x0, ref_y0, ref_x1, ref_y1 = reference_block.bbox
        
        # Calculate block height
        block_height = y1 - y0
        
        # Move block down below reference block
        new_y0 = ref_y1 + min_spacing
        new_y1 = new_y0 + block_height
        
        # Create new bbox
        new_bbox = [x0, new_y0, x1, new_y1]
        
        # Update baseline
        new_baseline = new_y0 + block.font_size * 0.75  # Approximate baseline
        
        # Create adjusted block
        adjusted = TextBlock(
            text=block.text,
            bbox=new_bbox,
            font_size=block.font_size,
            baseline_y=new_baseline,
            actual_width=block.actual_width,
            actual_height=block.actual_height,
            confidence=block.confidence,
            order=block.order
        )
        
        return adjusted
    
    def adjust_position_horizontal(
        self,
        block: TextBlock,
        reference_block: TextBlock,
        min_spacing: float
    ) -> TextBlock:
        """
        Adjust block position horizontally to avoid overlap
        
        Args:
            block: Block to adjust
            reference_block: Reference block (already placed)
            min_spacing: Minimum spacing required
            
        Returns:
            Adjusted TextBlock
        """
        x0, y0, x1, y1 = block.bbox
        ref_x0, ref_y0, ref_x1, ref_y1 = reference_block.bbox
        
        # Calculate block width
        block_width = x1 - x0
        
        # Move block to the right of reference block
        new_x0 = ref_x1 + min_spacing
        new_x1 = new_x0 + block_width
        
        # Create new bbox
        new_bbox = [new_x0, y0, new_x1, y1]
        
        # Create adjusted block
        adjusted = TextBlock(
            text=block.text,
            bbox=new_bbox,
            font_size=block.font_size,
            baseline_y=block.baseline_y,
            actual_width=block.actual_width,
            actual_height=block.actual_height,
            confidence=block.confidence,
            order=block.order
        )
        
        return adjusted
    
    def resolve_collisions(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Resolve all collisions by adjusting positions
        
        Args:
            blocks: List of TextBlock instances (should be sorted by reading order)
            
        Returns:
            List of adjusted TextBlock instances
        """
        if not blocks:
            return blocks
        
        # Sort blocks by Y coordinate (top to bottom), then X (left to right)
        sorted_blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
        
        resolved_blocks = []
        
        for i, block in enumerate(sorted_blocks):
            adjusted_block = block
            
            # Check against all previously placed blocks
            for ref_block in resolved_blocks:
                iou = calculate_iou(adjusted_block.bbox, ref_block.bbox)
                
                if iou > self.iou_threshold:
                    # Overlap detected, adjust position
                    if self.adjustment_strategy == "vertical_first":
                        # Try vertical adjustment first
                        adjusted_block = self.adjust_position_vertical(
                            adjusted_block,
                            ref_block,
                            self.min_spacing
                        )
                        
                        # Check if vertical adjustment still causes overlap
                        if calculate_iou(adjusted_block.bbox, ref_block.bbox) > self.iou_threshold:
                            # Fall back to horizontal adjustment
                            adjusted_block = self.adjust_position_horizontal(
                                adjusted_block,
                                ref_block,
                                self.min_spacing
                            )
                    
                    elif self.adjustment_strategy == "horizontal_first":
                        # Try horizontal adjustment first
                        adjusted_block = self.adjust_position_horizontal(
                            adjusted_block,
                            ref_block,
                            self.min_spacing
                        )
                        
                        # Check if horizontal adjustment still causes overlap
                        if calculate_iou(adjusted_block.bbox, ref_block.bbox) > self.iou_threshold:
                            # Fall back to vertical adjustment
                            adjusted_block = self.adjust_position_vertical(
                                adjusted_block,
                                ref_block,
                                self.min_spacing
                            )
                    
                    elif self.adjustment_strategy == "priority":
                        # Higher confidence/order wins, adjust lower priority
                        if block.confidence < ref_block.confidence or block.order > ref_block.order:
                            adjusted_block = self.adjust_position_vertical(
                                adjusted_block,
                                ref_block,
                                self.min_spacing
                            )
                        # If same priority, use vertical adjustment
                        else:
                            adjusted_block = self.adjust_position_vertical(
                                adjusted_block,
                                ref_block,
                                self.min_spacing
                            )
            
            resolved_blocks.append(adjusted_block)
        
        return resolved_blocks
    
    def get_collision_statistics(self, blocks: List[TextBlock]) -> Dict:
        """
        Get statistics about collisions in the block set
        
        Args:
            blocks: List of TextBlock instances
            
        Returns:
            Dictionary with collision statistics
        """
        overlaps = self.detect_overlaps(blocks)
        
        total_blocks = len(blocks)
        overlapping_blocks = set()
        for i, j, _ in overlaps:
            overlapping_blocks.add(i)
            overlapping_blocks.add(j)
        
        return {
            "total_blocks": total_blocks,
            "overlapping_pairs": len(overlaps),
            "overlapping_blocks": len(overlapping_blocks),
            "overlap_rate": len(overlapping_blocks) / total_blocks if total_blocks > 0 else 0.0,
            "max_iou": max([iou for _, _, iou in overlaps], default=0.0)
        }
