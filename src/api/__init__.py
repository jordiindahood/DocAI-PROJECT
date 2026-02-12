"""
DocAI API Module
REST API for document processing pipeline.
"""

from .main import app
from .routes import router

__all__ = ["app", "router"]
