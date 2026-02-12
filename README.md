# DocAI — Intelligent Document Layout Reconstruction

DocAI is an advanced document processing pipeline that reconstructs the layout of scanned documents or images. It detects structural elements (text blocks, titles, figures) using **YOLOv11**, extracts text with **Tesseract OCR**, and re-generates a searchable PDF that preserves the original visual layout.

## 🚀 Features

-   **Layout Detection**: Identifies document regions (paragraphs, headers, tables) using a custom-trained YOLOv11 model.
-   **Smart OCR**: Uses Tesseract with layout-aware Page Segmentation Modes (PSM) for high accuracy.
-   **Document Normalization**: Automatically detects document corners and applies perspective correction (de-skewing) using OpenCV.
-   **High-Fidelity Reconstruction**: Re-aligns text blocks on horizontal axes to fix OCR jitter and merges duplicate detections.
-   **PDF Export**: Generates a clean, searchable PDF where text is positioned exactly as in the original image.
-   **REST API**: FastAPI backend for easy integration.
-   **Web Interface**: Simple drag-and-drop UI for testing.

## 🛠️ Tech Stack

-   **Core**: Python 3.10+
-   **Deep Learning**: PyTorch, Ultralytics YOLOv11
-   **OCR**: Tesseract (pytesseract), PaddleOCR (optional)
-   **Computer Vision**: OpenCV, NumPy, Pillow
-   **Backend**: FastAPI, Uvicorn
-   **PDF Generation**: ReportLab
-   **Data Versioning**: DVC

## 📂 Project Structure

```
DocAI-PROJECT/
├── config/             # Configuration (model paths, thresholds)
├── models/             # Trained YOLO weights
├── src/
│   ├── api/            # FastAPI application
│   ├── detection/      # YOLOv11 inference
│   ├── ocr/            # Tesseract & Text cropping
│   ├── reconstruction/ # Layout analysis & text realignment
│   ├── export/         # PDF generation logic
│   └── layout_reconstruction_pipeline.py  # Main pipeline orchestrator
├── train/              # Legacy YOLOv8 training scripts
├── yolov11train/       # Current YOLOv11 training pipeline
└── data/               # Datasets & raw documents
```

## ⚡ Quick Start

### 1. Installation

Ensure you have **Tesseract OCR** installed on your system:
```bash
sudo apt-get install tesseract-ocr
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the API & Web App

Start the FastAPI server:
```bash
python src/api/main.py
```
-   **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
-   **Web Interface**: [http://localhost:8000/app/](http://localhost:8000/app/)

### 3. Run Pipeline via CLI

Process a single image and generate a PDF:
```bash
python src/layout_reconstruction_pipeline.py path/to/image.jpg output_folder/
```

## 🧠 Training Pipeline

The project includes a complete training pipeline in `yolov11train/` that automates:
1.  **PDF → Image**: Converts training documents.
2.  **Annotation**: Extracts ground truth from born-digital PDFs.
3.  **Augmentation**: Applies perspective warps, blur, and noise to simulate real-world scans.
4.  **Training**: Fine-tunes YOLOv11 on the augmented dataset.

See `yolov11train/README.md` (if available) or the configuration in `yolov11train/data.yaml` for details.
