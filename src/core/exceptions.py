class VirtualPainterException(Exception):
    """Base exception for Virtual Painter application"""
    pass

class CameraException(VirtualPainterException):
    """Camera-related exceptions"""
    pass

class GestureDetectionException(VirtualPainterException):
    """Gesture detection exceptions"""
    pass

class UIRenderingException(VirtualPainterException):
    """UI rendering exceptions"""
    pass

class ColorManagementException(VirtualPainterException):
    """Color management exceptions"""
    pass
