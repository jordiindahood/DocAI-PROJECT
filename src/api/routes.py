#!/usr/bin/env python
"""
DocAI API - Routes
API endpoints for document processing.
"""

import os
import sys
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse


# Add src to path for pipeline imports
_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from .utils import (
    save_uploaded_file,
    cleanup_temp_file,
    is_valid_image,
    OUTPUT_DIR,
)

# Import the pipeline
from layout_reconstruction_pipeline import process_and_export


router = APIRouter(prefix="/api/v1", tags=["Document Processing"])


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status of the API and its components
    """
    return {
        "status": "healthy",
        "service": "DocAI API",
        "version": "1.0.0",
    }


@router.post("/process")
async def process_document(image: UploadFile = File(...)):
    """
    Process an image and return the reconstructed PDF.
    
    This endpoint accepts an image file, runs the full layout reconstruction
    pipeline, and returns the generated PDF document.
    
    Args:
        image: Uploaded image file (PNG, JPG, JPEG, TIFF, BMP, WEBP)
        
    Returns:
        PDF file as a downloadable response
        
    Raises:
        400: Invalid file type
        500: Processing error
    """
    # Validate file type
    if not image.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not is_valid_image(image.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported: PNG, JPG, JPEG, TIFF, BMP, WEBP"
        )
    
    temp_image_path = None
    
    try:
        # Save uploaded file
        content = await image.read()
        temp_image_path = save_uploaded_file(content, image.filename)
        
        # Run the pipeline
        results = process_and_export(
            image_path=temp_image_path,
            output_format="pdf",
            output_dir=OUTPUT_DIR
        )
        
        # Get the PDF path
        pdf_path = results.get("export_paths", {}).get("pdf")
        
        if not pdf_path or not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=500,
                detail="PDF generation failed"
            )
        
        # Return the PDF file
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=os.path.basename(pdf_path),
            headers={
                "Content-Disposition": f'attachment; filename="{os.path.basename(pdf_path)}"'
            }
        )
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Cleanup temp file
        if temp_image_path:
            cleanup_temp_file(temp_image_path)


@router.post("/process/json")
async def process_document_json(image: UploadFile = File(...)):
    """
    Process an image and return the layout structure as JSON.
    
    This endpoint accepts an image file, runs the layout reconstruction
    pipeline, and returns the structured layout data without generating a PDF.
    
    Args:
        image: Uploaded image file (PNG, JPG, JPEG, TIFF, BMP, WEBP)
        
    Returns:
        JSON response with layout structure
        
    Raises:
        400: Invalid file type
        500: Processing error
    """
    # Validate file type
    if not image.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not is_valid_image(image.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported: PNG, JPG, JPEG, TIFF, BMP, WEBP"
        )
    
    temp_image_path = None
    
    try:
        # Save uploaded file
        content = await image.read()
        temp_image_path = save_uploaded_file(content, image.filename)
        
        # Import process_document for JSON-only response
        from layout_reconstruction_pipeline import process_document as pipeline_process
        
        results = pipeline_process(
            image_path=temp_image_path,
            output_dir=OUTPUT_DIR
        )
        
        # Return the layout structure
        return JSONResponse(content={
            "status": "success",
            "layout": results.get("layout_structure", {}),
            "json_path": results.get("json_path", "")
        })
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Cleanup temp file
        if temp_image_path:
            cleanup_temp_file(temp_image_path)
