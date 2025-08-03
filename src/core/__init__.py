"""
Core functionality namespace with optimized imports and lazy loading
"""
# Import shortcuts for frequently used classes
from .camera import CameraManager
from .hand_tracker import GestureDetector, GestureType
from .color_manager import ColorManager
from .canvas import DrawingEngine
from .exceptions import VirtualPainterException

# Lazy imports for heavy dependencies (available but not loaded immediately)
from .lazy_imports import cv2, numpy

# Optional/heavy components loaded on demand
_brush_engine = None
_canvas_tools = None

def get_brush_engine():
    """Lazy loader for brush engine"""
    global _brush_engine
    if _brush_engine is None:
        from .brush_engine import BrushEngine
        _brush_engine = BrushEngine
    return _brush_engine

def get_canvas_tools():
    """Lazy loader for canvas tools"""
    global _canvas_tools
    if _canvas_tools is None:
        from .canvas_tools import CanvasTools
        _canvas_tools = CanvasTools
    return _canvas_tools

# Export all public classes and functions
__all__ = [
    'CameraManager', 'GestureDetector', 'GestureType',
    'ColorManager', 'DrawingEngine', 'VirtualPainterException',
    'cv2', 'numpy', 'get_brush_engine', 'get_canvas_tools'
]

# Performance tracking
__optimized__ = True
__lazy_components__ = ['brush_engine', 'canvas_tools']
