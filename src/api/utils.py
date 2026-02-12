#!/usr/bin/env python
"""
DocAI API - Utility Functions
Helper functions for file handling, response construction, and error handling.
"""

import os
import uuid
import shutil
from datetime import datetime
from typing import Optional
from pathlib import Path
import yaml

_project_root = Path(__file__).resolve().parents[2]
_cfg = yaml.safe_load(open(_project_root / 'config' / 'config.yaml'))

_api_cfg = _cfg.get("api", {})

# Temporary directory for uploaded files and outputs
TEMP_DIR = _project_root / _api_cfg.get("temp_dir", "temp")
OUTPUT_DIR = _project_root / _api_cfg.get("output_dir", "outputs")


def ensure_directories():
    """Ensure temp and output directories exist."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_unique_filename(original_filename: str) -> str:
    """
    Generate a unique filename to avoid collisions.
    
    Args:
        original_filename: Original uploaded file name
        
    Returns:
        Unique filename with UUID prefix
    """
    ext = Path(original_filename).suffix
    unique_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{unique_id}{ext}"


def save_uploaded_file(file_content: bytes, original_filename: str) -> str:
    """
    Save uploaded file to temporary directory.
    
    Args:
        file_content: File content as bytes
        original_filename: Original file name
        
    Returns:
        Path to saved file (as string for compatibility)
    """
    ensure_directories()
    unique_name = generate_unique_filename(original_filename)
    file_path = TEMP_DIR / unique_name
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    return str(file_path)


def cleanup_temp_file(file_path: str) -> bool:
    """
    Remove a temporary file.
    
    Args:
        file_path: Path to file to remove
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            return True
        return False
    except Exception:
        return False


def cleanup_old_temp_files(max_age_hours: int = 24):
    """
    Remove temporary files older than specified age.
    
    Args:
        max_age_hours: Maximum age in hours before cleanup
    """
    ensure_directories()
    cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
    
    for file_path in TEMP_DIR.iterdir():
        try:
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
        except Exception:
            pass


def get_file_extension(filename: str) -> str:
    """
    Get file extension in lowercase without dot.
    
    Args:
        filename: File name
        
    Returns:
        File extension without dot, lowercase
    """
    return Path(filename).suffix.lower().lstrip(".")


def is_valid_image(filename: str) -> bool:
    """
    Check if file has a valid image extension.
    
    Args:
        filename: File name to check
        
    Returns:
        True if valid image extension
    """
    valid_extensions = {"png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp"}
    return get_file_extension(filename) in valid_extensions


def get_output_pdf_path(input_filename: str) -> str:
    """
    Generate output PDF path based on input filename.
    
    Args:
        input_filename: Original input file name
        
    Returns:
        Full path to expected output PDF (as string)
    """
    base_name = Path(input_filename).stem
    return str(OUTPUT_DIR / f"{base_name}.pdf")
