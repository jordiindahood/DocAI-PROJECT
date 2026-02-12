#!/usr/bin/env python
"""
Text Block Processor
Manages text blocks, calculates font sizes, groups by lines, and sorts by reading order
"""

from typing import List, Dict, Tuple
from .spatial_mapping import SpatialMapping, TextBlock, create_spatial_mapper


class TextBlockProcessor:
    """
    Processes text blocks for document reconstruction
    """
    
    def __init__(
        self,
        img_width: int,
        img_height: int,
        format_type: str = "pdf",
        line_tolerance: float = 10.0
    ):
        """
        Initialize text block processor
        
        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels
            format_type: "pdf"
            line_tolerance: Y-coordinate tolerance for grouping into lines (in pixels)
        """
        self.img_width = img_width
        self.img_height = img_height
        self.format_type = format_type
        self.line_tolerance = line_tolerance
        
        # Create spatial mapper
        self.spatial_mapper = create_spatial_mapper(
            img_width,
            img_height,
            format_type=format_type
        )
    
    def create_text_blocks(
        self,
        ocr_results: List[Dict],
        layout_results: List[Dict] = None
    ) -> List[TextBlock]:
        """
        Convert OCR results to TextBlock instances
        
        Args:
            ocr_results: List of OCR results with bbox, text, ocr_conf, yolo_conf
            layout_results: Optional layout predictions with layout_label, field_label
            
        Returns:
            List of TextBlock instances
        """
        text_blocks = []
        
        # Create mapping from OCR results to layout results by bbox
        layout_map = {}
        if layout_results:
            for layout_item in layout_results:
                layout_bbox = layout_item.get("bbox", [])
                if layout_bbox:
                    # Use bbox as key (rounded for matching)
                    key = tuple(int(coord) for coord in layout_bbox)
                    layout_map[key] = layout_item
        
        for idx, ocr_item in enumerate(ocr_results):
            text = ocr_item.get("text", "").strip()
            if not text:
                continue
            
            bbox = ocr_item.get("bbox", [])
            if len(bbox) < 4:
                continue
            
            # Get confidence scores
            ocr_conf = ocr_item.get("ocr_conf", 0.0)
            yolo_conf = ocr_item.get("yolo_conf", 0.0)
            confidence = (ocr_conf + yolo_conf) / 2.0 if (ocr_conf > 0 or yolo_conf > 0) else 1.0
            
            # Try to find matching layout result
            layout_item = None
            bbox_key = tuple(int(coord) for coord in bbox)
            if bbox_key in layout_map:
                layout_item = layout_map[bbox_key]
            
            # Create text block using spatial mapper
            text_block = self.spatial_mapper.create_text_block(
                text=text,
                bbox=bbox,
                confidence=confidence,
                order=idx
            )
            
            # Add layout information if available
            # Note: We'll access layout info from layout_results directly when needed
            # For now, store as attributes (TextBlock is a dataclass, so we can add attributes dynamically)
            if layout_item:
                setattr(text_block, 'layout_label', layout_item.get("layout_label", "other"))
                setattr(text_block, 'field_label', layout_item.get("field_label", "other"))
            
            text_blocks.append(text_block)
        
        return text_blocks
    
    def calculate_font_size(self, bbox: List[float], text: str = "") -> float:
        """
        Calculate font size from bounding box
        
        Args:
            bbox: Bounding box [x0, y0, x1, y1]
            text: Optional text content
            
        Returns:
            Font size in points
        """
        return self.spatial_mapper.calculate_font_size(bbox, text)
    
    def calculate_baseline(self, bbox: List[float], font_size: float) -> float:
        """
        Calculate text baseline position
        
        Args:
            bbox: Bounding box [x0, y0, x1, y1]
            font_size: Font size in points
            
        Returns:
            Baseline Y coordinate
        """
        return self.spatial_mapper.calculate_baseline(bbox, font_size)
    
    def group_by_lines(self, blocks: List[TextBlock]) -> List[List[TextBlock]]:
        """
        Group text blocks that belong to the same line
        
        Args:
            blocks: List of TextBlock instances
            
        Returns:
            List of line groups, each containing TextBlocks on the same line
        """
        if not blocks:
            return []
        
        # Sort by Y coordinate (top to bottom)
        sorted_blocks = sorted(blocks, key=lambda b: b.bbox[1])
        
        lines = []
        current_line = [sorted_blocks[0]]
        current_y = sorted_blocks[0].bbox[1]
        
        for block in sorted_blocks[1:]:
            block_y = block.bbox[1]
            
            # Check if block is on the same line (within tolerance)
            if abs(block_y - current_y) <= self.line_tolerance:
                current_line.append(block)
            else:
                # New line detected
                # Sort current line by X coordinate (left to right)
                current_line.sort(key=lambda b: b.bbox[0])
                lines.append(current_line)
                
                # Start new line
                current_line = [block]
                current_y = block_y
        
        # Add last line
        if current_line:
            current_line.sort(key=lambda b: b.bbox[0])
            lines.append(current_line)
        
        return lines
    
    def sort_by_reading_order(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Sort text blocks by reading order (top-to-bottom, left-to-right)
        
        Args:
            blocks: List of TextBlock instances
            
        Returns:
            Sorted list of TextBlock instances
        """
        if not blocks:
            return blocks
        
        # Sort by Y coordinate first (top to bottom), then X (left to right)
        # Use a small tolerance for Y to group similar rows
        sorted_blocks = sorted(blocks, key=lambda b: (
            round(b.bbox[1] / self.line_tolerance) * self.line_tolerance,  # Group by approximate line
            b.bbox[0]  # Then by X coordinate
        ))
        
        return sorted_blocks
    
    def merge_adjacent_blocks(self, blocks: List[TextBlock], max_gap: float = 5.0) -> List[TextBlock]:
        """
        Merge adjacent text blocks that likely belong together
        
        Args:
            blocks: List of TextBlock instances
            max_gap: Maximum gap between blocks to consider merging (in target space units)
            
        Returns:
            List of merged TextBlock instances
        """
        if not blocks:
            return blocks
        
        sorted_blocks = self.sort_by_reading_order(blocks)
        merged = []
        current_block = sorted_blocks[0]
        
        for next_block in sorted_blocks[1:]:
            # Check if blocks are on the same line and close together
            current_bbox = current_block.bbox
            next_bbox = next_block.bbox
            
            # Check Y overlap (same line)
            y_overlap = min(current_bbox[3], next_bbox[3]) - max(current_bbox[1], next_bbox[1])
            same_line = y_overlap > 0
            
            # Check horizontal gap
            horizontal_gap = next_bbox[0] - current_bbox[2]
            
            if same_line and 0 <= horizontal_gap <= max_gap:
                # Merge blocks
                merged_text = current_block.text + " " + next_block.text
                merged_bbox = [
                    min(current_bbox[0], next_bbox[0]),  # x0
                    min(current_bbox[1], next_bbox[1]),  # y0
                    max(current_bbox[2], next_bbox[2]),  # x1
                    max(current_bbox[3], next_bbox[3])   # y1
                ]
                
                # Recalculate font size and baseline
                font_size = self.calculate_font_size(merged_bbox, merged_text)
                baseline = self.calculate_baseline(merged_bbox, font_size)
                
                current_block = TextBlock(
                    text=merged_text,
                    bbox=merged_bbox,
                    font_size=font_size,
                    baseline_y=baseline,
                    actual_width=current_block.actual_width + next_block.actual_width + horizontal_gap,
                    actual_height=max(current_block.actual_height, next_block.actual_height),
                    confidence=min(current_block.confidence, next_block.confidence),
                    order=min(current_block.order, next_block.order)
                )
            else:
                # Can't merge, save current and move to next
                merged.append(current_block)
                current_block = next_block
        
        # Add last block
        merged.append(current_block)
        
        return merged
    
    def filter_low_confidence(self, blocks: List[TextBlock], min_confidence: float = 0.3) -> List[TextBlock]:
        """
        Filter out low-confidence text blocks
        
        Args:
            blocks: List of TextBlock instances
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of TextBlock instances
        """
        return [b for b in blocks if b.confidence >= min_confidence]
    
    def get_statistics(self, blocks: List[TextBlock]) -> Dict:
        """
        Get statistics about text blocks
        
        Args:
            blocks: List of TextBlock instances
            
        Returns:
            Dictionary with statistics
        """
        if not blocks:
            return {
                "total_blocks": 0,
                "total_text_length": 0,
                "avg_font_size": 0.0,
                "avg_confidence": 0.0
            }
        
        total_text_length = sum(len(b.text) for b in blocks)
        avg_font_size = sum(b.font_size for b in blocks) / len(blocks)
        avg_confidence = sum(b.confidence for b in blocks) / len(blocks)
        
        return {
            "total_blocks": len(blocks),
            "total_text_length": total_text_length,
            "avg_font_size": avg_font_size,
            "avg_confidence": avg_confidence,
            "min_font_size": min(b.font_size for b in blocks),
            "max_font_size": max(b.font_size for b in blocks)
        }
