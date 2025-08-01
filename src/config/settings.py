from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class UIConfig:
    """UI layout and styling configuration"""
    # Palette Layout
    PALETTE_COLS: int = 1
    PALETTE_ROWS: int = 10
    PALETTE_ITEM_SIZE: int = 45
    PALETTE_SPACING: int = 55
    PALETTE_START_X: int = 30
    PALETTE_START_Y: int = 150
    
    # Custom Color Section
    CUSTOM_SQUARE_POS: Tuple[int, int] = (30, 30)
    CUSTOM_SQUARE_SIZE: int = 50
    CUSTOM_BUTTON_POS: Tuple[int, int] = (90, 30)
    CUSTOM_BUTTON_SIZE: Tuple[int, int] = (80, 50)
    
    # Color Wheel
    WHEEL_CENTER: Tuple[int, int] = (200, 200)
    WHEEL_RADIUS: int = 80
    
    # Visual Effects
    BORDER_RADIUS: int = 15
    SHADOW_OFFSET: int = 4
    GLASS_ALPHA: float = 0.2
    
    def get_resolution_aware_positions(self, resolution: Tuple[int, int]) -> dict:
        """Get resolution-aware UI positions while maintaining current defaults for compatibility"""
        width, height = resolution
        
        # Calculate scale factor based on resolution (720p as baseline)
        scale_factor = min(width / 1280, height / 720, 2.0)  # Cap at 2x for very large screens
        
        return {
            'palette_start_x': max(20, int(width * 0.02)),  # 2% from left edge, min 20px
            'palette_start_y': max(100, int(height * 0.15)),  # 15% from top, min 100px
            'palette_item_size': max(30, int(self.PALETTE_ITEM_SIZE * scale_factor)),
            'palette_spacing': max(35, int(self.PALETTE_SPACING * scale_factor)),
            
            'custom_square_pos': (max(20, int(width * 0.02)), max(20, int(height * 0.05))),
            'custom_square_size': max(30, int(self.CUSTOM_SQUARE_SIZE * scale_factor)),
            'custom_button_pos': (max(70, int(width * 0.065)), max(20, int(height * 0.05))),
            'custom_button_size': (max(60, int(80 * scale_factor)), max(30, int(50 * scale_factor))),
            
            'wheel_center': (max(150, int(width * 0.15)), max(150, int(height * 0.25))),
            'wheel_radius': max(50, int(self.WHEEL_RADIUS * scale_factor)),
            
            'border_radius': max(8, int(self.BORDER_RADIUS * scale_factor)),
            'shadow_offset': max(2, int(self.SHADOW_OFFSET * scale_factor))
        }

@dataclass
class DrawingConfig:
    """Drawing and gesture configuration"""
    # Static thickness ranges and defaults
    BRUSH_THICKNESS: int = 10  # Default brush size
    ERASER_THICKNESS: int = 50  # Default eraser size
    
    # Dynamic sizing configuration
    MIN_BRUSH_SIZE: int = 1
    MAX_BRUSH_SIZE: int = 30
    MIN_ERASER_SIZE: int = 10
    MAX_ERASER_SIZE: int = 100
    
    SELECTION_DELAY: float = 0.3
    
    # Hand Detection
    MAX_NUM_HANDS: int = 2
    MIN_DETECTION_CONFIDENCE: float = 0.7
    
    # Gesture Processing
    MIN_GESTURE_DURATION: float = 0.05  # Minimum time to hold gesture before recognition
    
    def __post_init__(self):
        """Validate drawing configuration values"""
        if self.MIN_BRUSH_SIZE >= self.MAX_BRUSH_SIZE:
            raise ValueError(f"MIN_BRUSH_SIZE ({self.MIN_BRUSH_SIZE}) must be < MAX_BRUSH_SIZE ({self.MAX_BRUSH_SIZE})")
        
        if self.MIN_ERASER_SIZE >= self.MAX_ERASER_SIZE:
            raise ValueError(f"MIN_ERASER_SIZE ({self.MIN_ERASER_SIZE}) must be < MAX_ERASER_SIZE ({self.MAX_ERASER_SIZE})")
        
        if not (1 <= self.BRUSH_THICKNESS <= 100):
            raise ValueError(f"BRUSH_THICKNESS ({self.BRUSH_THICKNESS}) must be between 1 and 100")
        
        if not (1 <= self.ERASER_THICKNESS <= 200):
            raise ValueError(f"ERASER_THICKNESS ({self.ERASER_THICKNESS}) must be between 1 and 200")
        
        if not (0.0 <= self.MIN_DETECTION_CONFIDENCE <= 1.0):
            raise ValueError(f"MIN_DETECTION_CONFIDENCE ({self.MIN_DETECTION_CONFIDENCE}) must be between 0.0 and 1.0")
        
        if not (1 <= self.MAX_NUM_HANDS <= 4):
            raise ValueError(f"MAX_NUM_HANDS ({self.MAX_NUM_HANDS}) must be between 1 and 4")
        
        if not (0.01 <= self.MIN_GESTURE_DURATION <= 1.0):
            raise ValueError(f"MIN_GESTURE_DURATION ({self.MIN_GESTURE_DURATION}) must be between 0.01 and 1.0 seconds")
        
        # Ensure default sizes are within ranges
        if not (self.MIN_BRUSH_SIZE <= self.BRUSH_THICKNESS <= self.MAX_BRUSH_SIZE):
            self.BRUSH_THICKNESS = max(self.MIN_BRUSH_SIZE, min(self.MAX_BRUSH_SIZE, self.BRUSH_THICKNESS))
        
        if not (self.MIN_ERASER_SIZE <= self.ERASER_THICKNESS <= self.MAX_ERASER_SIZE):
            self.ERASER_THICKNESS = max(self.MIN_ERASER_SIZE, min(self.MAX_ERASER_SIZE, self.ERASER_THICKNESS))

@dataclass
class InfoPanelConfig:
    """Info panel behavior configuration"""
    DISPLAY_DURATION: float = 5.0
    COOLDOWN_DURATION: float = 2.0
    FADE_START_TIME: float = 2.0

@dataclass
class CameraConfig:
    """Camera and resolution configuration"""
    RESOLUTIONS: List[Tuple[int, int]] = None
    DEFAULT_CAMERA_INDEX: int = 0
    
    def __post_init__(self):
        if self.RESOLUTIONS is None:
            self.RESOLUTIONS = [
                (1920, 1080),
                (1280, 720),
                (1024, 576),
                (800, 600),
                (640, 480)
            ]
        
        # Validate camera index
        if not (0 <= self.DEFAULT_CAMERA_INDEX <= 10):
            raise ValueError(f"DEFAULT_CAMERA_INDEX ({self.DEFAULT_CAMERA_INDEX}) must be between 0 and 10")
        
        # Validate resolutions
        for i, (width, height) in enumerate(self.RESOLUTIONS):
            if not (320 <= width <= 4096) or not (240 <= height <= 2160):
                raise ValueError(f"Resolution {i} ({width}x{height}) is outside valid range (320x240 to 4096x2160)")
    
    def get_preferred_resolution(self) -> Tuple[int, int]:
        """Get the first (preferred) resolution"""
        return self.RESOLUTIONS[0] if self.RESOLUTIONS else (1280, 720)

@dataclass
class AppConfig:
    """Main application configuration"""
    ui: UIConfig = None
    drawing: DrawingConfig = None
    info_panel: InfoPanelConfig = None
    camera: CameraConfig = None
    
    # Color Palette
    COLOR_PALETTE: List[Tuple[Tuple[int, int, int], str, Tuple[int, int, int]]] = None
    
    def __post_init__(self):
        # Initialize nested configs if None
        try:
            if self.ui is None:
                self.ui = UIConfig()
            if self.drawing is None:
                self.drawing = DrawingConfig()
            if self.info_panel is None:
                self.info_panel = InfoPanelConfig()
            if self.camera is None:
                self.camera = CameraConfig()
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")
            
        if self.COLOR_PALETTE is None:
            self.COLOR_PALETTE = [
                ((0, 0, 255), 'Red', (50, 50, 255)),
                ((0, 128, 255), 'Orange', (100, 180, 255)),
                ((0, 255, 0), 'Green', (100, 255, 100)),
                ((255, 255, 0), 'Yellow', (255, 255, 100)),
                ((255, 0, 0), 'Blue', (255, 100, 100)),
                ((255, 0, 255), 'Magenta', (255, 100, 255)),
                ((128, 0, 255), 'Purple', (180, 100, 255)),
                ((255, 128, 0), 'Gold', (255, 180, 100)),
                ((128, 128, 128), 'Gray', (180, 180, 180)),
                ((0, 0, 0), 'Black', (50, 50, 50))
            ]
        
        # Validate color palette
        self._validate_color_palette()
    
    def _validate_color_palette(self):
        """Validate color palette format and values"""
        if not isinstance(self.COLOR_PALETTE, list) or len(self.COLOR_PALETTE) == 0:
            raise ValueError("COLOR_PALETTE must be a non-empty list")
        
        for i, item in enumerate(self.COLOR_PALETTE):
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError(f"COLOR_PALETTE item {i} must be a tuple of 3 elements (color, name, gradient)")
            
            color, name, gradient = item
            
            # Validate color format
            if not isinstance(color, tuple) or len(color) != 3:
                raise ValueError(f"COLOR_PALETTE item {i}: color must be a tuple of 3 integers (B, G, R)")
            
            if not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
                raise ValueError(f"COLOR_PALETTE item {i}: color values must be integers between 0 and 255")
            
            # Validate name
            if not isinstance(name, str) or len(name.strip()) == 0:
                raise ValueError(f"COLOR_PALETTE item {i}: name must be a non-empty string")
            
            # Validate gradient
            if not isinstance(gradient, tuple) or len(gradient) != 3:
                raise ValueError(f"COLOR_PALETTE item {i}: gradient must be a tuple of 3 integers (B, G, R)")
            
            if not all(isinstance(c, int) and 0 <= c <= 255 for c in gradient):
                raise ValueError(f"COLOR_PALETTE item {i}: gradient values must be integers between 0 and 255")