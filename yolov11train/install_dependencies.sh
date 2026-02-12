#!/bin/bash

# Install dependencies for YOLOv11 training pipeline
# Run as: ./install_dependencies.sh

set -e

echo "============================================="
echo "Installing Dependencies for YOLOv11 Pipeline"
echo "============================================="

# 1. System Dependencies
echo "Checking system dependencies..."

if command -v apt-get &> /dev/null; then
    if ! command -v pdftocairo &> /dev/null; then
        echo "Installing poppler-utils (requires sudo)..."
        if sudo -n true 2>/dev/null; then
             sudo apt-get update && sudo apt-get install -y poppler-utils
        else
             echo "Requesting sudo permission to install 'poppler-utils'..."
             sudo apt-get update && sudo apt-get install -y poppler-utils
        fi
    else
        echo "✓ poppler-utils is already installed."
    fi
else
    echo "Warning: apt-get not found. Skipping system package installation."
    echo "Ensure 'poppler-utils' is installed manually if you encounter errors."
fi

# 2. Python Dependencies
echo "Installing Python libraries..."

# List of required packages
PACKAGES=(
    "ultralytics"         # YOLOv11 framework (includes torch, torchvision)
    "pdf2image"           # PDF to Image conversion
    "pymupdf"             # PDF parsing (fitz)
    "opencv-python-headless" # Image processing (cv2) - using headless for servers/scripts
    "Pillow"              # Image handling
    "numpy"               # Numerical operations
)

# Install packages
pip install --upgrade pip
pip install "${PACKAGES[@]}"

echo "============================================="
echo "All dependencies installed successfully!"
echo "============================================="
