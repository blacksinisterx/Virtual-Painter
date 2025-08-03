"""
Optimized configuration package with namespace shortcuts and fast access
"""
from .settings import (
    AppConfig,
    DrawingConfig,
    CameraConfig,
    UIConfig,
    InfoPanelConfig
)

# Fast access shortcuts for common usage
from .settings import AppConfig as Config

# Export all public classes
__all__ = [
    'AppConfig', 'Config', 'DrawingConfig',
    'CameraConfig', 'UIConfig', 'InfoPanelConfig'
]

# Package metadata for debugging and optimization tracking
__version__ = '1.0.0'
__optimized__ = True
__performance_level__ = 'high'
