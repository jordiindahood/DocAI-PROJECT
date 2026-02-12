#!/usr/bin/env python
"""
Spatial Mapping Utility
Handles coordinate normalization and spatial correction for document reconstruction
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass
import math


@dataclass
class TextBlock:
    """Represents a text block with spatial information"""
    text: str
    bbox: List[float]  # [x0, y0, x1, y1] in image coordinates
    font_size: float = 0.0
    baseline_y: float = 0.0
    actual_width: float = 0.0
    actual_height: float = 0.0
    confidence: float = 1.0
    order: int = 0


class SpatialMapping:
    """
    Handles coordinate normalization and spatial correction for document reconstruction
    """
    
    def __init__(
        self,
        img_width: int,
        img_height: int,
        target_width: float = 612.0,  # Letter size width in points
        target_height: float = 792.0,  # Letter size height in points
        dpi: int = 300,
        font_metric_factor: float = 0.7,
        descent_ratio: float = 0.25
    ):
        """
        Initialize spatial mapping
        
        Args:
            img_width: Source image width in pixels
            img_height: Source image height in pixels
            target_width: Target page width (PDF points)
            target_height: Target page height (PDF points)
            dpi: Assumed DPI for scaling calculations
            font_metric_factor: Factor for font size calculation (0.6-0.8 typical)
            descent_ratio: Ratio of font size for baseline descent (0.2-0.3 typical)
        """
        self.img_width = img_width
        self.img_height = img_height
        self.target_width = target_width
        self.target_height = target_height
        self.dpi = dpi
        self.font_metric_factor = font_metric_factor
        self.descent_ratio = descent_ratio
        
        # Calculate scale factors
        scale_x = target_width / img_width if img_width > 0 else 1.0
        scale_y = target_height / img_height if img_height > 0 else 1.0
        self.scale = min(scale_x, scale_y)  # Maintain aspect ratio
    
    def normalize_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """
        Normalize image coordinates to target space
        
        Args:
            x: X coordinate in image space
            y: Y coordinate in image space (top-left origin)
            
        Returns:
            (x_norm, y_norm) in target space (PDF: bottom-left origin)
        """
        x_norm = x * self.scale
        # For PDF: flip Y axis (bottom-left origin)
        y_norm = (self.img_height - y) * self.scale  # PDF format (flipped)
        
        return x_norm, y_norm
    
    def normalize_bbox(self, bbox: List[float]) -> List[float]:
        """
        Normalize bounding box coordinates
        
        Args:
            bbox: [x0, y0, x1, y1] in image coordinates
            
        Returns:
            [x0, y0, x1, y1] in target space
        """
        x0, y0, x1, y1 = bbox
        
        # Normalize coordinates
        x0_norm, y1_norm = self.normalize_coordinates(x0, y0)  # Top-left
        x1_norm, y0_norm = self.normalize_coordinates(x1, y1)  # Bottom-right
        
        # Ensure valid order
        if x0_norm > x1_norm:
            x0_norm, x1_norm = x1_norm, x0_norm
        if y0_norm > y1_norm:
            y0_norm, y1_norm = y1_norm, y0_norm
        
        return [x0_norm, y0_norm, x1_norm, y1_norm]
    
    def calculate_font_size(self, bbox: List[float], text: str = "") -> float:
        """
        Calculate font size from bounding box height
        
        Args:
            bbox: Bounding box [x0, y0, x1, y1] in image coordinates
            text: Optional text content for more accurate estimation
            
        Returns:
            Font size in points
        """
        x0, y0, x1, y1 = bbox
        box_height = y1 - y0
        
        # Scale box height to target space
        scaled_height = box_height * self.scale
        
        # Calculate font size: box_height * scale_factor * font_metric_factor
        # Typical font size is 60-80% of box height
        font_size = scaled_height * self.font_metric_factor
        
        # Clamp to reasonable range (6-24 points)
        font_size = max(6.0, min(24.0, font_size))
        
        return font_size
    
    def calculate_baseline(self, bbox: List[float], font_size: float) -> float:
        """
        Calculate text baseline position
        
        Args:
            bbox: Bounding box [x0, y0, x1, y1] in image coordinates
            font_size: Font size in points
            
        Returns:
            Baseline Y coordinate in target space
        """
        x0, y0, x1, y1 = bbox
        
        # Normalize top Y coordinate (y0 is top of bbox)
        _, pdf_y_top = self.normalize_coordinates(x0, y0)
        
        # Baseline is below the top by descent amount
        # Descent is typically 20-30% of font size
        baseline_y = pdf_y_top - (font_size * self.descent_ratio)
        
        return baseline_y
    
    def calculate_text_dimensions(self, text: str, font_size: float, font_name: str = "Helvetica") -> Tuple[float, float]:
        """
        Estimate text dimensions (simplified - actual measurement requires font metrics)
        
        Args:
            text: Text content
            font_size: Font size in points
            font_name: Font name (for future font metric lookup)
            
        Returns:
            (width, height) in target space units
        """
        # Simplified estimation: ~0.6 * font_size per character (average)
        # This is approximate and varies by font
        char_width = font_size * 0.6
        text_width = len(text) * char_width
        
        # Height is approximately font_size
        text_height = font_size
        
        return text_width, text_height
    
    def create_text_block(
        self,
        text: str,
        bbox: List[float],
        confidence: float = 1.0,
        order: int = 0
    ) -> TextBlock:
        """
        Create a TextBlock with all spatial information calculated
        
        Args:
            text: Text content
            bbox: Bounding box [x0, y0, x1, y1] in image coordinates
            confidence: Confidence score
            order: Reading order
            
        Returns:
            TextBlock instance
        """
        # Calculate font size
        font_size = self.calculate_font_size(bbox, text)
        
        # Calculate baseline
        baseline_y = self.calculate_baseline(bbox, font_size)
        
        # Calculate text dimensions
        actual_width, actual_height = self.calculate_text_dimensions(text, font_size)
        
        # Normalize bbox
        normalized_bbox = self.normalize_bbox(bbox)
        
        return TextBlock(
            text=text,
            bbox=normalized_bbox,
            font_size=font_size,
            baseline_y=baseline_y,
            actual_width=actual_width,
            actual_height=actual_height,
            confidence=confidence,
            order=order
        )
    



def create_spatial_mapper(
    img_width: int,
    img_height: int,
    target_width: float = 612.0,
    target_height: float = 792.0,
    format_type: str = "pdf"
) -> SpatialMapping:
    """
    Factory function to create SpatialMapping for different formats
    
    Args:
        img_width: Source image width
        img_height: Source image height
        target_width: Target page width
        target_height: Target page height
        format_type: "pdf"
        
    Returns:
        SpatialMapping instance
    """
    
    return SpatialMapping(
        img_width=img_width,
        img_height=img_height,
        target_width=target_width,
        target_height=target_height
    )
