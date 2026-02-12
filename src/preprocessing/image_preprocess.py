#!/usr/bin/env python
"""
Image Preprocessing Module
Preprocesses scanned document images for better OCR accuracy
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale
    
    Args:
        image: Input image (BGR or RGB numpy array)
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            # Convert BGR to grayscale (OpenCV format)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # Already grayscale or has alpha channel
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY) if image.shape[2] == 4 else image
    else:
        gray = image
    
    return gray


def apply_adaptive_threshold(image: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    """
    Apply adaptive thresholding to binarize image
    
    Args:
        image: Grayscale image
        block_size: Size of pixel neighborhood (must be odd)
        C: Constant subtracted from mean
        
    Returns:
        Binary image
    """
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)
    
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )
    
    return binary


def deskew_image(image: np.ndarray, max_angle: float = 15.0) -> Tuple[np.ndarray, float]:
    """
    Correct image skew using projection profile method
    
    Args:
        image: Grayscale or binary image
        max_angle: Maximum angle to try (degrees)
        
    Returns:
        Tuple of (deskewed_image, detected_angle)
    """
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)
    
    # Convert to binary if needed
    if image.dtype != np.uint8 or np.max(image) > 1:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary = image
    
    # Calculate projection profile for different angles
    best_angle = 0.0
    best_variance = 0.0
    
    angles = np.arange(-max_angle, max_angle + 0.5, 0.5)
    
    for angle in angles:
        # Rotate image
        h, w = binary.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Calculate horizontal projection
        projection = np.sum(rotated, axis=1)
        
        # Calculate variance (higher variance = better alignment)
        variance = np.var(projection)
        
        if variance > best_variance:
            best_variance = variance
            best_angle = angle
    
    # Apply best rotation to original image
    if abs(best_angle) > 0.1:  # Only rotate if angle is significant
        h, w = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        deskewed = image.copy()
    
    return deskewed, best_angle


def normalize_resolution(image: np.ndarray, target_dpi: int = 300, current_dpi: Optional[int] = None) -> np.ndarray:
    """
    Normalize image resolution to target DPI
    
    Args:
        image: Input image
        target_dpi: Target DPI (default: 300)
        current_dpi: Current DPI (if known, otherwise estimated)
        
    Returns:
        Resized image
    """
    if current_dpi is None:
        # Estimate current DPI based on image size
        # Assume typical scanned document is 8.5x11 inches at original DPI
        # This is a rough estimate - in practice, DPI metadata should be used
        height, width = image.shape[:2]
        # Estimate: assume document is ~11 inches tall
        estimated_dpi = int(height / 11.0)
        current_dpi = estimated_dpi if estimated_dpi > 0 else 300
    
    if current_dpi == target_dpi:
        return image
    
    # Calculate scaling factor
    scale_factor = target_dpi / current_dpi
    
    if scale_factor == 1.0:
        return image
    
    # Resize image
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return resized


def preprocess_image(image_path: Union[str, Path], target_dpi: int = 300, 
                     apply_grayscale: bool = True, apply_threshold: bool = True,
                     apply_deskew: bool = True, apply_resize: bool = True) -> Image.Image:
    """
    Main preprocessing function for scanned document images
    
    Args:
        image_path: Path to input image
        target_dpi: Target DPI for resolution normalization (default: 300)
        apply_grayscale: Whether to convert to grayscale
        apply_threshold: Whether to apply adaptive thresholding
        apply_deskew: Whether to deskew the image
        apply_resize: Whether to normalize resolution
        
    Returns:
        Preprocessed PIL Image
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image using OpenCV (BGR format)
    cv_image = cv2.imread(str(image_path))
    if cv_image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Step 1: Convert to grayscale
    if apply_grayscale:
        processed = convert_to_grayscale(cv_image)
    else:
        processed = cv_image.copy()
        if len(processed.shape) == 3:
            processed = convert_to_grayscale(processed)
    
    # Step 2: Apply adaptive thresholding
    if apply_threshold:
        processed = apply_adaptive_threshold(processed)
    
    # Step 3: Deskew
    if apply_deskew:
        processed, angle = deskew_image(processed)
        if abs(angle) > 0.1:
            print(f"  Deskewed image by {angle:.2f} degrees")
    
    # Step 4: Normalize resolution
    if apply_resize:
        processed = normalize_resolution(processed, target_dpi=target_dpi)
    
    # Convert back to PIL Image (RGB format)
    # OpenCV uses BGR, PIL uses RGB
    if len(processed.shape) == 2:
        # Grayscale - convert to RGB
        pil_image = Image.fromarray(processed, mode='L').convert('RGB')
    else:
        # Already RGB/BGR
        rgb_image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB) if len(processed.shape) == 3 else processed
        pil_image = Image.fromarray(rgb_image)
    
    return pil_image


if __name__ == "__main__":
    # Test the preprocessing module
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python image_preprocess.py <image_path> [output_path]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "preprocessed_output.png"
    
    print(f"Preprocessing image: {input_path}")
    preprocessed = preprocess_image(input_path)
    preprocessed.save(output_path)
    print(f"Preprocessed image saved to: {output_path}")
