"""
UI components namespace with optimized loading and lazy utilities
"""
# Core UI components (always needed)
from .ui_renderer import UIRenderer
from .info_panel import InfoPanelManager

# Optional UI utilities (loaded on demand)
_drawing_utils = None
_color_wheel = None

def get_drawing_utils():
    """Lazy loader for drawing utilities"""
    global _drawing_utils
    if _drawing_utils is None:
        from .drawing_utils import DrawingUtils
        _drawing_utils = DrawingUtils
    return _drawing_utils

def get_color_wheel():
    """Lazy loader for color wheel component"""
    global _color_wheel
    if _color_wheel is None:
        from .color_wheel import ColorWheel
        _color_wheel = ColorWheel
    return _color_wheel

# Check if component is already loaded
def is_drawing_utils_loaded() -> bool:
    return _drawing_utils is not None

def is_color_wheel_loaded() -> bool:
    return _color_wheel is not None

# Export all public classes and utilities
__all__ = [
    'UIRenderer', 'InfoPanelManager',
    'get_drawing_utils', 'get_color_wheel',
    'is_drawing_utils_loaded', 'is_color_wheel_loaded'
]

# Performance metadata
__optimized__ = True
__lazy_components__ = ['drawing_utils', 'color_wheel']
