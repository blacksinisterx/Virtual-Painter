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
    """Drawing and gesture configuration with optimized validation"""
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

    # MediaPipe Optimization
    MIN_TRACKING_CONFIDENCE: float = 0.5  # Lower for better tracking continuity
    MODEL_COMPLEXITY: int = 1  # 0=lite, 1=full (balanced performance)
    WARMUP_FRAMES: int = 3  # Number of frames to warm up MediaPipe

    # Frame Rate Control
    TARGET_FPS: int = 30  # Target frames per second
    FRAME_BUFFER_SIZE: int = 3  # Number of frames to buffer
    ENABLE_VSYNC: bool = True  # Enable vertical sync for smooth display

    # Gesture Processing
    MIN_GESTURE_DURATION: float = 0.05  # Minimum time to hold gesture before recognition

    # Dual-Stream MediaPipe Processing (Phase 2.3)
    ENABLE_DUAL_STREAM: bool = True  # Enable dual-stream architecture
    DISPLAY_WIDTH: int = 1280  # High-quality display resolution
    DISPLAY_HEIGHT: int = 720
    PROCESS_WIDTH: int = 480   # Optimized MediaPipe processing resolution
    PROCESS_HEIGHT: int = 270

    # Adaptive Preprocessing
    ENABLE_GAUSSIAN_BLUR: bool = False  # Conditional blur for noise reduction
    GAUSSIAN_KERNEL_SIZE: int = 3
    ENABLE_HISTOGRAM_EQ: bool = False   # Conditional histogram equalization
    HISTOGRAM_EQ_THRESHOLD: float = 0.3 # Apply if contrast below this

    # Pre-validated defaults for fast validation bypass
    _DEFAULT_VALUES = {
        'BRUSH_THICKNESS': 10,
        'ERASER_THICKNESS': 50,
        'MIN_BRUSH_SIZE': 1,
        'MAX_BRUSH_SIZE': 30,
        'MIN_ERASER_SIZE': 10,
        'MAX_ERASER_SIZE': 100,
        'SELECTION_DELAY': 0.3,
        'MAX_NUM_HANDS': 2,
        'MIN_DETECTION_CONFIDENCE': 0.7,
        'MIN_TRACKING_CONFIDENCE': 0.5,
        'MODEL_COMPLEXITY': 1,
        'WARMUP_FRAMES': 3,
        'TARGET_FPS': 30,
        'FRAME_BUFFER_SIZE': 3,
        'ENABLE_VSYNC': True,
        'MIN_GESTURE_DURATION': 0.05,
        'ENABLE_DUAL_STREAM': True,
        'DISPLAY_WIDTH': 1280,
        'DISPLAY_HEIGHT': 720,
        'PROCESS_WIDTH': 480,
        'PROCESS_HEIGHT': 270,
        'ENABLE_GAUSSIAN_BLUR': False,
        'GAUSSIAN_KERNEL_SIZE': 3,
        'ENABLE_HISTOGRAM_EQ': False,
        'HISTOGRAM_EQ_THRESHOLD': 0.3
    }

    def __post_init__(self):
        """Optimized validation with defaults bypass"""
        # Fast path: Skip validation if using all default values
        if self._is_using_defaults():
            return

        # Only validate changed values
        self._validate_configuration()

    def _is_using_defaults(self) -> bool:
        """Check if all values match defaults for fast bypass"""
        for key, default_value in self._DEFAULT_VALUES.items():
            if getattr(self, key) != default_value:
                return False
        return True

    def _validate_configuration(self):
        """Streamlined validation for non-default configurations"""
        # Size range validations (grouped for efficiency)
        if self.MIN_BRUSH_SIZE >= self.MAX_BRUSH_SIZE:
            raise ValueError(f"MIN_BRUSH_SIZE ({self.MIN_BRUSH_SIZE}) must be < MAX_BRUSH_SIZE ({self.MAX_BRUSH_SIZE})")

        if self.MIN_ERASER_SIZE >= self.MAX_ERASER_SIZE:
            raise ValueError(f"MIN_ERASER_SIZE ({self.MIN_ERASER_SIZE}) must be < MAX_ERASER_SIZE ({self.MAX_ERASER_SIZE})")

        # Range validations with early exit
        self._validate_range(self.BRUSH_THICKNESS, 1, 100, "BRUSH_THICKNESS")
        self._validate_range(self.ERASER_THICKNESS, 1, 200, "ERASER_THICKNESS")
        self._validate_range(self.MIN_DETECTION_CONFIDENCE, 0.0, 1.0, "MIN_DETECTION_CONFIDENCE")
        self._validate_range(self.MAX_NUM_HANDS, 1, 4, "MAX_NUM_HANDS")
        self._validate_range(self.MIN_GESTURE_DURATION, 0.01, 1.0, "MIN_GESTURE_DURATION")

        # Auto-correct default sizes to valid ranges
        self.BRUSH_THICKNESS = max(self.MIN_BRUSH_SIZE, min(self.MAX_BRUSH_SIZE, self.BRUSH_THICKNESS))
        self.ERASER_THICKNESS = max(self.MIN_ERASER_SIZE, min(self.MAX_ERASER_SIZE, self.ERASER_THICKNESS))

    def _validate_range(self, value: float, min_val: float, max_val: float, field: str):
        """Optimized range validation"""
        if not (min_val <= value <= max_val):
            raise ValueError(f"{field} ({value}) must be between {min_val} and {max_val}")

@dataclass
class InfoPanelConfig:
    """Info panel behavior configuration"""
    DISPLAY_DURATION: float = 5.0
    COOLDOWN_DURATION: float = 2.0
    FADE_START_TIME: float = 2.0

@dataclass
class CameraConfig:
    """Camera and resolution configuration with quality profiles"""
    RESOLUTIONS: List[Tuple[int, int]] = None

    # Pre-validated resolution presets for fast initialization
    VALIDATED_PRESETS = {
        'HD': [(1920, 1080), (1280, 720), (800, 600), (640, 480)],
        'PERFORMANCE': [(1280, 720), (800, 600), (640, 480)],
        'COMPATIBILITY': [(800, 600), (640, 480)],
        'DEFAULT': [(1920, 1080), (1280, 720), (1024, 576), (800, 600), (640, 480)]
    }

    # Quality profiles for performance tuning
    QUALITY_PROFILES = {
        'HIGH': {
            'resolution_preset': 'HD',
            'frame_buffer_size': 3,
            'enable_post_processing': True,
            'compression_quality': 95
        },
        'MEDIUM': {
            'resolution_preset': 'PERFORMANCE',
            'frame_buffer_size': 2,
            'enable_post_processing': True,
            'compression_quality': 85
        },
        'LOW': {
            'resolution_preset': 'COMPATIBILITY',
            'frame_buffer_size': 1,
            'enable_post_processing': False,
            'compression_quality': 75
        }
    }

    def __init__(self, preset: str = 'DEFAULT', custom_resolutions: List[Tuple[int, int]] = None, quality_profile: str = 'MEDIUM'):
        self.quality_profile = quality_profile

        if custom_resolutions:
            self.RESOLUTIONS = self._validate_custom_resolutions(custom_resolutions)
        else:
            # Use quality profile to determine resolution preset
            if quality_profile in self.QUALITY_PROFILES:
                profile = self.QUALITY_PROFILES[quality_profile]
                preset = profile['resolution_preset']

            # Use pre-validated presets - no validation needed
            self.RESOLUTIONS = self.VALIDATED_PRESETS[preset].copy()

    def get_quality_settings(self) -> dict:
        """Get current quality profile settings"""
        return self.QUALITY_PROFILES.get(self.quality_profile, self.QUALITY_PROFILES['MEDIUM'])

    def __post_init__(self):
        # Only validate if RESOLUTIONS was set directly (not via constructor)
        if self.RESOLUTIONS is None:
            self.RESOLUTIONS = self.VALIDATED_PRESETS['DEFAULT'].copy()
        elif not hasattr(self, '_validated'):
            # Validate only if not already validated via constructor
            self._validate_resolutions()

    def _validate_custom_resolutions(self, resolutions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Validate custom resolution list"""
        validated = []
        for i, (width, height) in enumerate(resolutions):
            if not (320 <= width <= 4096) or not (240 <= height <= 2160):
                raise ValueError(f"Resolution {i} ({width}x{height}) is outside valid range (320x240 to 4096x2160)")
            validated.append((width, height))
        self._validated = True
        return validated

    def _validate_resolutions(self):
        """Validate resolution list for direct assignment"""
        for i, (width, height) in enumerate(self.RESOLUTIONS):
            if not (320 <= width <= 4096) or not (240 <= height <= 2160):
                raise ValueError(f"Resolution {i} ({width}x{height}) is outside valid range (320x240 to 4096x2160)")
        self._validated = True

    def get_preferred_resolution(self) -> Tuple[int, int]:
        """Get the first (preferred) resolution"""
        return self.RESOLUTIONS[0] if self.RESOLUTIONS else (1280, 720)

@dataclass
class AppConfig:
    """Streamlined application configuration with property-based lazy loading"""
    # Core configs (always loaded)
    camera: CameraConfig = None
    drawing: DrawingConfig = None

    # Lazy-loaded configs (loaded on first access)
    _ui: UIConfig = None
    _info_panel: InfoPanelConfig = None

    # Color Palette with optimized validation
    COLOR_PALETTE: List[Tuple[Tuple[int, int, int], str, Tuple[int, int, int]]] = None
    _palette_validated: bool = False

    def __post_init__(self):
        # Initialize core configs immediately with optimized constructors
        if self.camera is None:
            self.camera = CameraConfig('PERFORMANCE')  # Use performance preset
        if self.drawing is None:
            self.drawing = DrawingConfig()  # Uses cached defaults

        # Defer palette initialization until needed
        if self.COLOR_PALETTE is None:
            self._init_default_palette()

    @property
    def ui(self) -> UIConfig:
        """Lazy-loaded UI configuration"""
        if self._ui is None:
            self._ui = UIConfig()
        return self._ui

    @ui.setter
    def ui(self, value: UIConfig):
        self._ui = value

    @property
    def info_panel(self) -> InfoPanelConfig:
        """Lazy-loaded info panel configuration"""
        if self._info_panel is None:
            self._info_panel = InfoPanelConfig()
        return self._info_panel

    @info_panel.setter
    def info_panel(self, value: InfoPanelConfig):
        self._info_panel = value

    def _init_default_palette(self):
        """Initialize default color palette without validation"""
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
        self._palette_validated = True  # Default palette is pre-validated

    def ensure_palette_valid(self):
        """Validate color palette only when needed"""
        if not self._palette_validated:
            self._validate_color_palette()
            self._palette_validated = True

    def get_critical_config(self):
        """Get critical configuration subset"""
        return {
            'camera': self.camera,
            'drawing': self.drawing
        }

    def _validate_color_palette(self):
        """Optimized color palette validation with early exit"""
        if not isinstance(self.COLOR_PALETTE, list) or len(self.COLOR_PALETTE) == 0:
            raise ValueError("COLOR_PALETTE must be a non-empty list")

        # Pre-check structure before detailed validation
        for i, item in enumerate(self.COLOR_PALETTE):
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError(f"COLOR_PALETTE item {i} must be a tuple of 3 elements (color, name, gradient)")

        # Batch validation for better performance
        for i, (color, name, gradient) in enumerate(self.COLOR_PALETTE):
            # Validate color format (optimized checks)
            if not (isinstance(color, tuple) and len(color) == 3 and
                    all(isinstance(c, int) and 0 <= c <= 255 for c in color)):
                raise ValueError(f"COLOR_PALETTE item {i}: color must be a tuple of 3 integers (0-255)")

            # Validate name (streamlined check)
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"COLOR_PALETTE item {i}: name must be a non-empty string")

            # Validate gradient (reuse color validation logic)
            if not (isinstance(gradient, tuple) and len(gradient) == 3 and
                    all(isinstance(c, int) and 0 <= c <= 255 for c in gradient)):
                raise ValueError(f"COLOR_PALETTE item {i}: gradient must be a tuple of 3 integers (0-255)")
