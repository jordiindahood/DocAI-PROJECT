#!/usr/bin/env python
"""
Export Manager
Exports pure layout structure to PDF format with bbox-based font sizing
"""

import os
from typing import Dict, List
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import yaml

_cfg = yaml.safe_load(open(Path('config/config.yaml')))


class ExportManager:
    """
    Manages export of pure layout structure to PDF.
    Uses bounding box dimensions to determine font size.
    """

    def __init__(self, output_dir=None):
        """
        Initialize export manager.

        Args:
            output_dir: Directory to save exported files (default: from config.yaml)
        """
        if output_dir is None:
            output_dir = _cfg["export"].get("output_dir", "outputs")
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def export_pure_layout(self, data, base_name = "document", formats = None):
        """
        Export pure layout structure to specified formats.
        
        Args:
            data: Pure layout structure containing:
                {
                    "page_width": int,
                    "page_height": int,
                    "texts": [
                        {
                            "text": str,
                            "bbox": [x_min, y_min, x_max, y_max],
                            "page": int,
                            "order": int
                        }
                    ]
                }
            base_name: Base name for output files
            formats: List of formats to export ["pdf"]
        
        Returns:
            Dict mapping format to output file path
        """
        if formats is None:
            formats = ["pdf"]
        
        export_paths = {}
        
        if "pdf" in formats:
            pdf_path = self._export_pdf(data, base_name)
            export_paths["pdf"] = pdf_path
        
        return export_paths
    
    def export_pdf(self, data, base_name):
        """
        Export to PDF format using bbox dimensions for font sizing.
        Each text block is positioned and sized according to its bounding box.
        
        Args:
            data: Pure layout structure
            base_name: Base name for output file
        
        Returns:
            Path to generated PDF
        """
        output_path = os.path.join(self.output_dir, f"{base_name}.pdf")
        
        # Get page dimensions
        cfg_export = _cfg["export"]
        page_width = data.get("page_width", cfg_export.get("page_width_default", 595))
        page_height = data.get("page_height", cfg_export.get("page_height_default", 842))
        
        # Create PDF canvas
        c = canvas.Canvas(output_path, pagesize=(page_width, page_height))
        
        # Get texts sorted by reading order
        texts = sorted(data.get("texts", []), key=lambda t: t.get("order", 0))
        
        # Draw text on PDF
        for text_item in texts:
            text = text_item.get("text", "").strip()
            if not text:
                continue
            
            bbox = text_item.get("bbox", [])
            if len(bbox) < 4:
                continue
            
            x0, y0, x1, y1 = bbox
            
            # Calculate font size based on bbox height
            box_height = y1 - y0
            box_width = x1 - x0
            
            # Font size is ~85% of box height (accounts for line spacing)
            font_size = box_height * 0.85
            
            # Minimum readable font size
            font_size = max(6, font_size)
            
            # Set font
            c.setFont("Helvetica", font_size)
            
            # Check if text fits in box width, scale down if needed
            text_width = c.stringWidth(text, "Helvetica", font_size)
            if text_width > box_width and box_width > 0:
                # Scale font to fit width
                scale_factor = box_width / text_width * 0.95
                font_size = font_size * scale_factor
                font_size = max(4, font_size)  # Absolute minimum
                c.setFont("Helvetica", font_size)
            
            # PDF coordinates: origin at bottom-left, we need to flip Y
            # Position text at bottom of bbox (baseline)
            pdf_y = page_height - y1 + (font_size * 0.15)  # Slight baseline adjustment
            
            # Draw text at bbox position
            c.drawString(x0, pdf_y, text)
        
        # Save PDF
        c.save()
        
        print(f"PDF exported: {output_path}")
        return output_path
