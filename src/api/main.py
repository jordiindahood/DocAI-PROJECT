#!/usr/bin/env python
"""
DocAI API - Main Application
FastAPI application for document processing pipeline.

Usage:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
    
Or from project root:
    python -m uvicorn src.api.main:app --reload
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# Add src to path so pipeline modules (detection, ocr, etc.) are importable
_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from .routes import router
from .utils import ensure_directories, cleanup_old_temp_files, OUTPUT_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Handles startup and shutdown events.
    """
    # Startup
    print("Starting DocAI API...")
    ensure_directories()
    cleanup_old_temp_files(max_age_hours=24)
    print("✓ Directories initialized")
    print("✓ Old temp files cleaned up")
    
    yield
    
    # Shutdown
    print("Shutting down DocAI API...")


# Create FastAPI application
app = FastAPI(
    title="DocAI API",
    description="""
    Document Processing API for layout reconstruction.
    
    ## Features
    
    - **Image to PDF conversion**: Upload an image and receive a reconstructed PDF
    - **Layout extraction**: Get structured layout data in JSON format
    
    ## Supported Image Formats
    
    - PNG, JPG, JPEG, TIFF, BMP, WEBP
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for web app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Mount static files for output access (optional)
if os.path.exists(OUTPUT_DIR):
    app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Mount the web app (src/app/) as static files at /app
APP_DIR = Path(__file__).resolve().parent.parent / "app"
if APP_DIR.is_dir():
    app.mount("/app", StaticFiles(directory=str(APP_DIR), html=True), name="webapp")


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "service": "DocAI API",
        "version": "1.0.0",
        "description": "Document Processing API for layout reconstruction",
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "process": "POST /api/v1/process - Upload image, get PDF",
            "process_json": "POST /api/v1/process/json - Upload image, get JSON layout",
            "health": "GET /api/v1/health - Health check",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
