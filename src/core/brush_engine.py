import cv2
import numpy as np
import math
import time
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from core.exceptions import VirtualPainterException

class BrushType(Enum):
    """Different brush types available"""
    ROUND = "round"
    SQUARE = "square"
    TEXTURE = "texture"
    WATERCOLOR = "watercolor"
    OIL_PAINT = "oil_paint"
    CHARCOAL = "charcoal"
    MARKER = "marker"
    AIRBRUSH = "airbrush"

class BlendMode(Enum):
    """Blending modes for brush effects"""
    NORMAL = "normal"
    MULTIPLY = "multiply"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft_light"
    ADDITIVE = "additive"

@dataclass
class BrushSettings:
    """Configuration for a brush"""
    brush_type: BrushType
    size: int
    opacity: float  # 0.0 - 1.0
    hardness: float  # 0.0 - 1.0 (soft to hard edges)
    texture_strength: float  # 0.0 - 1.0
    speed_sensitivity: float  # How much speed affects the brush
    pressure_sensitivity: float  # Simulated pressure sensitivity
    blend_mode: BlendMode
    color_variation: float  # Random color variation
    flow: float  # Paint flow rate
    spacing: float  # Distance between brush stamps
    jitter: float  # Random position variation

    @classmethod
    def default_round(cls):
        return cls(
            brush_type=BrushType.ROUND,
            size=10,
            opacity=1.0,
            hardness=0.8,
            texture_strength=0.0,
            speed_sensitivity=0.3,
            pressure_sensitivity=0.5,
            blend_mode=BlendMode.NORMAL,
            color_variation=0.0,
            flow=1.0,
            spacing=0.1,
            jitter=0.0
        )

    @classmethod
    def watercolor_brush(cls):
        return cls(
            brush_type=BrushType.WATERCOLOR,
            size=15,
            opacity=0.3,
            hardness=0.2,
            texture_strength=0.4,
            speed_sensitivity=0.6,
            pressure_sensitivity=0.8,
            blend_mode=BlendMode.MULTIPLY,
            color_variation=0.2,
            flow=0.7,
            spacing=0.05,
            jitter=0.1
        )

@dataclass
class BrushStroke:
    """Represents a single brush stroke point"""
    x: int
    y: int
    pressure: float
    speed: float
    timestamp: float
    size: int
    opacity: float

class TextureManager:
    """Manages brush textures and patterns"""

    def __init__(self):
        self.textures: Dict[str, np.ndarray] = {}
        self._generate_default_textures()

    def _generate_default_textures(self) -> None:
        """Generate default procedural textures"""

        # Paper texture
        paper_texture = self._generate_paper_texture(64)
        self.textures["paper"] = paper_texture

        # Canvas texture
        canvas_texture = self._generate_canvas_texture(32)
        self.textures["canvas"] = canvas_texture

        # Noise texture for watercolor
        noise_texture = self._generate_noise_texture(128)
        self.textures["noise"] = noise_texture

        # Brush stroke texture
        stroke_texture = self._generate_stroke_texture(64)
        self.textures["stroke"] = stroke_texture

    def _generate_paper_texture(self, size: int) -> np.ndarray:
        """Generate paper-like texture"""
        texture = np.random.randint(240, 255, (size, size), dtype=np.uint8)

        # Add some fiber patterns
        for _ in range(20):
            x, y = np.random.randint(0, size, 2)
            length = np.random.randint(5, 15)
            angle = np.random.random() * 2 * np.pi

            end_x = int(x + length * np.cos(angle))
            end_y = int(y + length * np.sin(angle))

            if 0 <= end_x < size and 0 <= end_y < size:
                cv2.line(texture, (x, y), (end_x, end_y),
                        int(np.random.randint(230, 250)), 1)

        return texture

    def _generate_canvas_texture(self, size: int) -> np.ndarray:
        """Generate canvas-like texture"""
        texture = np.ones((size, size), dtype=np.uint8) * 245

        # Add weave pattern
        for i in range(0, size, 4):
            texture[i:i+2, :] = 250
            texture[:, i:i+2] = 250

        # Add some random variation
        noise = np.random.randint(-10, 10, (size, size))
        texture = np.clip(texture.astype(int) + noise, 0, 255).astype(np.uint8)

        return texture

    def _generate_noise_texture(self, size: int) -> np.ndarray:
        """Generate noise texture for organic effects"""
        # Generate Perlin-like noise
        texture = np.random.randint(0, 255, (size, size), dtype=np.uint8)

        # Smooth the noise
        kernel = np.ones((3, 3), np.float32) / 9
        texture = cv2.filter2D(texture, -1, kernel)

        return texture

    def _generate_stroke_texture(self, size: int) -> np.ndarray:
        """Generate brush stroke texture"""
        texture = np.zeros((size, size), dtype=np.uint8)
        center = size // 2

        # Create soft circular gradient
        for y in range(size):
            for x in range(size):
                distance = np.sqrt((x - center)**2 + (y - center)**2)
                if distance < center:
                    value = int(255 * (1 - distance / center)**2)
                    texture[y, x] = value

        return texture

    def get_texture(self, name: str, size: Tuple[int, int] = None) -> np.ndarray:
        """Get texture, optionally resized"""
        if name not in self.textures:
            return self.textures["paper"]

        texture = self.textures[name]

        if size and size != texture.shape[:2]:
            texture = cv2.resize(texture, size)

        return texture

class BrushEngine:
    """Advanced brush rendering engine with effects and textures"""

    def __init__(self):
        self.texture_manager = TextureManager()
        self.current_settings = BrushSettings.default_round()
        self.brush_presets = self._initialize_presets()

        # Stroke tracking for advanced effects
        self.current_stroke: List[BrushStroke] = []
        self.last_render_time = 0
        self.accumulated_paint = {}  # For wet paint effects

    def _initialize_presets(self) -> Dict[str, BrushSettings]:
        """Initialize brush presets"""
        return {
            "Watercolor": BrushSettings.watercolor_brush()
        }

    def set_brush_preset(self, preset_name: str) -> bool:
        """Set brush to a preset configuration"""
        if preset_name in self.brush_presets:
            self.current_settings = self.brush_presets[preset_name]
            return True
        return False

    def calculate_dynamic_properties(self, speed: float, base_size: int) -> Tuple[int, float]:
        """Calculate dynamic brush properties based on movement speed"""
        current_time = time.time()

        # Speed-based pressure simulation
        pressure = 1.0 - min(speed * self.current_settings.speed_sensitivity * 0.1, 0.8)
        pressure = max(0.2, pressure)  # Minimum pressure

        # Dynamic size based on pressure
        size_variation = 1.0 + (pressure - 0.5) * self.current_settings.pressure_sensitivity
        dynamic_size = int(base_size * size_variation)
        dynamic_size = max(1, min(dynamic_size, base_size * 2))

        # Dynamic opacity
        opacity_variation = 0.3 + (pressure * 0.7)
        dynamic_opacity = self.current_settings.opacity * opacity_variation

        return dynamic_size, dynamic_opacity

    def render_brush_stroke(self, canvas: np.ndarray, x: int, y: int,
                          color: Tuple[int, int, int], base_size: int,
                          speed: float = 0.0) -> np.ndarray:
        """Render advanced brush stroke with effects"""
        try:
            # Calculate dynamic properties
            dynamic_size, dynamic_opacity = self.calculate_dynamic_properties(speed, base_size)

            # Add current point to stroke
            current_time = time.time()
            stroke_point = BrushStroke(
                x=x, y=y,
                pressure=1.0 - min(speed * 0.1, 0.8),
                speed=speed,
                timestamp=current_time,
                size=dynamic_size,
                opacity=dynamic_opacity
            )
            self.current_stroke.append(stroke_point)

            # Render based on brush type
            if self.current_settings.brush_type == BrushType.WATERCOLOR:
                return self._render_watercolor_brush(canvas, x, y, color, dynamic_size, dynamic_opacity)
            else:
                return self._render_round_brush(canvas, x, y, color, dynamic_size, dynamic_opacity)

        except Exception as e:
            # Fallback to simple circle
            cv2.circle(canvas, (x, y), base_size // 2, color, -1)
            return canvas

    def _render_round_brush(self, canvas: np.ndarray, x: int, y: int,
                          color: Tuple[int, int, int], size: int, opacity: float) -> np.ndarray:
        """Render round brush with soft edges"""
        radius = size // 2

        # Create brush mask with soft edges based on hardness
        mask_size = radius * 2 + 10
        mask = np.zeros((mask_size, mask_size), dtype=np.float32)
        center = mask_size // 2

        for dy in range(mask_size):
            for dx in range(mask_size):
                distance = np.sqrt((dx - center)**2 + (dy - center)**2)
                if distance <= radius:
                    # Soft edge calculation
                    if distance <= radius * self.current_settings.hardness:
                        alpha = 1.0
                    else:
                        alpha = 1.0 - ((distance - radius * self.current_settings.hardness) /
                                     (radius * (1 - self.current_settings.hardness)))
                    mask[dy, dx] = max(0, alpha * opacity)

        # Apply mask to canvas
        start_x = max(0, x - center)
        start_y = max(0, y - center)
        end_x = min(canvas.shape[1], x + center)
        end_y = min(canvas.shape[0], y + center)

        mask_start_x = max(0, center - x)
        mask_start_y = max(0, center - y)
        mask_end_x = mask_start_x + (end_x - start_x)
        mask_end_y = mask_start_y + (end_y - start_y)

        if start_x < end_x and start_y < end_y:
            region = canvas[start_y:end_y, start_x:end_x]
            brush_mask = mask[mask_start_y:mask_end_y, mask_start_x:mask_end_x]

            for c in range(3):
                region[:, :, c] = (region[:, :, c] * (1 - brush_mask) +
                                 color[c] * brush_mask).astype(np.uint8)

        return canvas

    def _render_watercolor_brush(self, canvas: np.ndarray, x: int, y: int,
                               color: Tuple[int, int, int], size: int, opacity: float) -> np.ndarray:
        """Render watercolor brush with bleeding effects"""
        # Base round brush
        canvas = self._render_round_brush(canvas, x, y, color, size, opacity * 0.7)

        # Add bleeding effect with multiple smaller circles
        num_bleeds = 3
        for i in range(num_bleeds):
            offset_x = np.random.randint(-size//2, size//2)
            offset_y = np.random.randint(-size//2, size//2)
            bleed_x = x + offset_x
            bleed_y = y + offset_y
            bleed_size = size // (2 + i)
            bleed_opacity = opacity * 0.3 / (i + 1)

            if (0 <= bleed_x < canvas.shape[1] and 0 <= bleed_y < canvas.shape[0]):
                canvas = self._render_round_brush(canvas, bleed_x, bleed_y, color,
                                                bleed_size, bleed_opacity)

        return canvas

    def start_stroke(self) -> None:
        """Start a new brush stroke"""
        self.current_stroke.clear()

    def end_stroke(self) -> None:
        """End current brush stroke"""
        self.current_stroke.clear()

    def get_current_brush_info(self) -> Dict:
        """Get information about current brush settings"""
        return {
            "type": self.current_settings.brush_type.value,
            "size": self.current_settings.size,
            "opacity": self.current_settings.opacity,
            "hardness": self.current_settings.hardness,
            "texture_strength": self.current_settings.texture_strength,
            "speed_sensitivity": self.current_settings.speed_sensitivity
        }

    def get_available_presets(self) -> List[str]:
        """Get list of available brush presets"""
        return list(self.brush_presets.keys())
