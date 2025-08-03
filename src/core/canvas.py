import cv2
import numpy as np
from typing import Tuple, Optional, List, Deque, Dict
from dataclasses import dataclass
from collections import deque
import math
import time

from config.settings import DrawingConfig
from core.hand_tracker import GestureType
from core.exceptions import VirtualPainterException
from core.brush_engine import BrushEngine, BrushSettings
from core.canvas_tools import CanvasTools, GridType

@dataclass
class DrawingState:
    """Enhanced drawing state with movement tracking"""
    prev_x: int = 0
    prev_y: int = 0
    is_drawing: bool = False
    last_movement_time: float = 0.0
    stroke_started: bool = False  # Track if we just started a new stroke
    movement_speed: float = 0.0  # Track movement speed for brush effects

@dataclass
class CanvasAction:
    """Represents a single action for undo/redo functionality"""
    action_type: str  # 'draw', 'erase', 'clear'
    canvas_state: np.ndarray
    timestamp: float
    metadata: dict = None

class DrawingEngine:
    """Enhanced drawing engine with advanced brush dynamics and state management"""

    def __init__(self, config: DrawingConfig, canvas_size: Tuple[int, int]):
        self.config = config
        self.canvas = np.zeros((canvas_size[1], canvas_size[0], 3), np.uint8)
        self.canvas_size = canvas_size
        self.state = DrawingState()

        # Enhanced features
        self.anti_aliasing = True

        # Advanced brush system
        self.brush_engine = BrushEngine()
        self.use_advanced_brushes = True  # Toggle for advanced brush effects

        # Phase 4.1: Canvas Tools System
        self.canvas_tools = CanvasTools(canvas_size)

        # Undo/Redo system
        self.history: Deque[CanvasAction] = deque(maxlen=50)  # 50 action limit
        self.history_index = -1

        # Performance tracking
        self.stroke_quality_metrics = {
            'total_strokes': 0
        }

        # Save initial state
        self._save_canvas_state('init')

    def process_gesture(self, gesture_type: GestureType, position: Tuple[int, int],
                       color: Tuple[int, int, int], thickness: Optional[int] = None) -> Optional[str]:
        """Enhanced gesture processing with improved drawing logic and snap-to-grid support"""
        try:
            # Apply snap-to-grid if enabled
            x, y = self.canvas_tools.snap_point_to_grid(position)
            current_time = time.time()

            if gesture_type == GestureType.DRAWING:
                brush_thickness = thickness if thickness is not None else self.config.BRUSH_THICKNESS
                return self._handle_enhanced_drawing(x, y, color, brush_thickness, current_time)
            elif gesture_type == GestureType.ERASING:
                eraser_thickness = thickness if thickness is not None else self.config.ERASER_THICKNESS
                return self._handle_enhanced_erasing(x, y, eraser_thickness, current_time)
            elif gesture_type == GestureType.PEN_LIFTED:
                return self._handle_pen_lifted()
            else:
                # Important: Reset drawing state when no drawing gesture is detected
                if self.state.is_drawing:
                    self._reset_drawing_state()
                return None

        except Exception as e:
            return None

    def _handle_enhanced_drawing(self, x: int, y: int, color: Tuple[int, int, int],
                               thickness: int, current_time: float) -> str:
        """Enhanced drawing with advanced brush effects and grid constraint"""

        # Apply grid direction constraint if snap-to-grid is enabled
        if self.canvas_tools.state.snap_to_grid_enabled and self.state.is_drawing:
            x, y = self.canvas_tools.constrain_to_grid_direction((self.state.prev_x, self.state.prev_y), (x, y))

        # Handle pixel art mode
        if self.canvas_tools.state.drawing_mode.value == "pixel":
            return self._handle_pixel_art_drawing(x, y, color, current_time)

        # Check if we need to start a new stroke
        if not self.state.is_drawing:
            # Start completely new stroke - save canvas state for undo
            self._save_canvas_state('draw')
            self.state.prev_x, self.state.prev_y = x, y
            self.state.is_drawing = True
            self.state.stroke_started = True
            self.state.last_movement_time = current_time
            self.state.movement_speed = 0.0

            # Start brush stroke
            self.brush_engine.start_stroke()

            # Draw initial point to avoid line jumping
            if self.use_advanced_brushes:
                self.brush_engine.render_brush_stroke(self.canvas, x, y, color, thickness, 0.0)
            else:
                cv2.circle(self.canvas, (x, y), max(1, thickness // 2), color, -1)
            return "Drawing started"

        # Calculate movement speed for brush effects
        if self.state.prev_x != 0 and self.state.prev_y != 0:
            distance = math.sqrt((x - self.state.prev_x)**2 + (y - self.state.prev_y)**2)
            time_delta = current_time - self.state.last_movement_time
            self.state.movement_speed = distance / max(time_delta, 0.001)  # pixels per second

        # Continue existing stroke
        if self.use_advanced_brushes:
            # Render continuous stroke by interpolating between previous and current position
            self._render_continuous_brush_stroke(x, y, color, thickness, self.state.movement_speed)
        else:
            self._draw_basic_line(x, y, color, thickness)

        self.state.prev_x, self.state.prev_y = x, y
        self.state.last_movement_time = current_time
        self.state.stroke_started = False  # No longer starting

        return "Drawing"

    def _handle_pixel_art_drawing(self, x: int, y: int, color: Tuple[int, int, int], current_time: float) -> str:
        """Handle pixel art drawing mode - fills entire grid squares"""

        # Get the grid square bounds for the current position
        x1, y1, x2, y2 = self.canvas_tools.get_grid_square_bounds((x, y))

        # Check if we need to start a new stroke
        if not self.state.is_drawing:
            self._save_canvas_state('pixel_draw')
            self.state.is_drawing = True
            self.state.stroke_started = True
            self.state.last_movement_time = current_time

            # Store the current grid square as our starting square
            self.state.prev_x, self.state.prev_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw the pixel
            self._draw_pixel_square(x1, y1, x2, y2, color)
            return "Pixel art started"

        # Continue pixel art - only fill new grid squares
        current_center_x, current_center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Check if we've moved to a different grid square
        if current_center_x != self.state.prev_x or current_center_y != self.state.prev_y:
            # Draw the new pixel
            self._draw_pixel_square(x1, y1, x2, y2, color)

            # Update previous position to current grid square center
            self.state.prev_x, self.state.prev_y = current_center_x, current_center_y

        self.state.last_movement_time = current_time
        self.state.stroke_started = False

        return "Pixel drawing"

    def _draw_pixel_square(self, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]) -> None:
        """Draw a pixel square with optional border"""
        if self.canvas_tools.state.grid_type == GridType.NORMAL:
            # Normal rectangular pixel
            if self.canvas_tools.state.pixel_borders:
                # Fill the square with color
                cv2.rectangle(self.canvas, (x1, y1), (x2 - 1, y2 - 1), color, -1)
                # Add black border (1 pixel thick)
                cv2.rectangle(self.canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 0), 1)
            else:
                # Just fill the square
                cv2.rectangle(self.canvas, (x1, y1), (x2 - 1, y2 - 1), color, -1)
        else:
            # Slanted grid: draw diamond shape
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            half_size = (x2 - x1) // 2

            # Diamond points (rotated 45°)
            pts = np.array([
                [center_x, center_y - half_size],  # Top
                [center_x + half_size, center_y],  # Right
                [center_x, center_y + half_size],  # Bottom
                [center_x - half_size, center_y]   # Left
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))

            if self.canvas_tools.state.pixel_borders:
                # Fill diamond with color
                cv2.fillPoly(self.canvas, [pts], color)
                # Add black border
                cv2.polylines(self.canvas, [pts], True, (0, 0, 0), 1)
            else:
                # Just fill the diamond
                cv2.fillPoly(self.canvas, [pts], color)

    def _draw_basic_line(self, x: int, y: int, color: Tuple[int, int, int], thickness: int) -> None:
        """Draw basic line between current and previous position"""
        line_type = cv2.LINE_AA if self.anti_aliasing else cv2.LINE_8
        cv2.line(self.canvas, (self.state.prev_x, self.state.prev_y),
                (x, y), color, thickness, line_type)

    def _render_continuous_brush_stroke(self, x: int, y: int, color: Tuple[int, int, int],
                                      thickness: int, speed: float) -> None:
        """Render continuous brush stroke by interpolating between previous and current position"""
        # If we're just starting or have no previous position, just render current point
        if self.state.stroke_started or self.state.prev_x == 0 or self.state.prev_y == 0:
            self.brush_engine.render_brush_stroke(self.canvas, x, y, color, thickness, speed)
            return

        # Calculate distance between previous and current position
        distance = math.sqrt((x - self.state.prev_x)**2 + (y - self.state.prev_y)**2)

        # If distance is very small, just render current point
        if distance <= 2:
            self.brush_engine.render_brush_stroke(self.canvas, x, y, color, thickness, speed)
            return

        # Calculate optimal spacing for smooth strokes
        # Use brush spacing setting with a maximum density limit
        brush_spacing = self.brush_engine.current_settings.spacing
        optimal_spacing = max(1.0, min(thickness * brush_spacing, thickness * 0.3))  # Never more than 30% of brush size

        # Number of interpolation points based on distance and optimal spacing
        num_points = max(2, int(distance / optimal_spacing))

        for i in range(1, num_points + 1):  # Start from 1 to avoid duplicate at previous position
            # Linear interpolation between previous and current position
            t = i / num_points
            interp_x = int(self.state.prev_x + t * (x - self.state.prev_x))
            interp_y = int(self.state.prev_y + t * (y - self.state.prev_y))

            # Calculate interpolated speed
            interp_speed = speed * (0.8 + 0.4 * t)  # Gradually increase speed towards current

            # Render brush stroke at interpolated position
            self.brush_engine.render_brush_stroke(self.canvas, interp_x, interp_y, color, thickness, interp_speed)

    def _handle_enhanced_erasing(self, x: int, y: int, thickness: int, current_time: float) -> str:
        """Enhanced erasing with no line jumping"""

        # Check if we need to start new erasing stroke
        if not self.state.is_drawing:
            # Start completely new erase stroke
            self._save_canvas_state('erase')
            self.state.prev_x, self.state.prev_y = x, y
            self.state.is_drawing = True
            self.state.stroke_started = True
            self.state.last_movement_time = current_time

            # Draw initial erase circle to avoid jumping
            cv2.circle(self.canvas, (x, y), thickness // 2, (0, 0, 0), -1)
            return "Erasing started"

        # Continue erasing - draw circle at current position
        cv2.circle(self.canvas, (x, y), thickness // 2, (0, 0, 0), -1)

        # Connect with previous position for continuous erasing only if we're not just starting
        if not self.state.stroke_started and self.state.prev_x != 0 and self.state.prev_y != 0:
            cv2.line(self.canvas, (self.state.prev_x, self.state.prev_y),
                    (x, y), (0, 0, 0), thickness)

        self.state.prev_x, self.state.prev_y = x, y
        self.state.last_movement_time = current_time
        self.state.stroke_started = False  # No longer starting
        return "Erasing"

    def _handle_pen_lifted(self) -> str:
        """Enhanced pen lifted gesture with stroke completion"""
        if self.state.is_drawing:
            # End brush stroke
            self.brush_engine.end_stroke()
            # Finalize stroke and update metrics
            self._finalize_stroke()
        self._reset_drawing_state()
        return "Pen lifted"

    def _finalize_stroke(self) -> None:
        """Finalize current stroke and update quality metrics"""
        # Update metrics
        self.stroke_quality_metrics['total_strokes'] += 1

    def _save_canvas_state(self, action_type: str, metadata: dict = None) -> None:
        """Save current canvas state for undo/redo functionality"""
        # Remove any future history if we're not at the end
        if self.history_index < len(self.history) - 1:
            # Keep only history up to current index
            self.history = deque(list(self.history)[:self.history_index + 1], maxlen=50)

        # Save current state
        action = CanvasAction(
            action_type=action_type,
            canvas_state=self.canvas.copy(),
            timestamp=time.time(),
            metadata=metadata or {}
        )

        self.history.append(action)
        self.history_index = len(self.history) - 1

    def undo(self) -> bool:
        """Undo last action"""
        if self.history_index > 0:
            self.history_index -= 1
            self.canvas = self.history[self.history_index].canvas_state.copy()
            self._reset_drawing_state()
            return True
        return False

    def redo(self) -> bool:
        """Redo next action"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.canvas = self.history[self.history_index].canvas_state.copy()
            self._reset_drawing_state()
            return True
        return False

    def can_undo(self) -> bool:
        """Check if undo is available"""
        return self.history_index > 0

    def can_redo(self) -> bool:
        """Check if redo is available"""
        return self.history_index < len(self.history) - 1

    def get_canvas_info(self) -> dict:
        """Get comprehensive canvas information"""
        return {
            'canvas_size': self.canvas_size,
            'total_strokes': self.stroke_quality_metrics['total_strokes'],
            'history_size': len(self.history),
            'can_undo': self.can_undo(),
            'can_redo': self.can_redo(),
            'is_drawing': self.state.is_drawing
        }

    def _reset_drawing_state(self) -> None:
        """Enhanced drawing state reset to prevent line jumping"""
        self.state.prev_x, self.state.prev_y = 0, 0
        self.state.is_drawing = False
        self.state.last_movement_time = 0.0
        self.state.stroke_started = False
        self.state.movement_speed = 0.0

    def merge_with_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced canvas merging with canvas tools overlay"""
        try:
            # Render canvas tools on frame (grid only, no rulers)
            self.canvas_tools.render_grid(frame)

            # Then merge drawing canvas
            gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)

            return cv2.add(frame_bg, canvas_fg)

        except Exception as e:
            return frame

    def save_canvas(self, filename: str) -> bool:
        """Enhanced canvas saving with metadata"""
        try:
            # Create metadata overlay (optional)
            canvas_with_info = self.canvas.copy()

            success = cv2.imwrite(filename, canvas_with_info)
            if success:
                # Save canvas state
                self._save_canvas_state('save', {'filename': filename})
            return success
        except Exception as e:
            return False

    def clear_canvas(self) -> None:
        """Enhanced canvas clearing with state preservation"""
        self._save_canvas_state('clear')
        self.canvas.fill(0)
        self._reset_drawing_state()

        # Reset metrics
        self.stroke_quality_metrics = {
            'total_strokes': 0
        }

    def get_canvas(self) -> np.ndarray:
        """Get current canvas with optional processing"""
        return self.canvas.copy()

    def toggle_anti_aliasing(self) -> bool:
        """Toggle anti-aliasing feature"""
        self.anti_aliasing = not self.anti_aliasing
        return self.anti_aliasing

    # Advanced brush system methods
    def toggle_advanced_brushes(self) -> bool:
        """Toggle between normal mode and watercolor mode"""
        self.use_advanced_brushes = not self.use_advanced_brushes
        if self.use_advanced_brushes:
            # Switch to watercolor mode
            self.brush_engine.set_brush_preset("Watercolor")
        return self.use_advanced_brushes

    def set_brush_preset(self, preset_name: str) -> bool:
        """Set brush to a preset configuration"""
        return self.brush_engine.set_brush_preset(preset_name)

    def get_brush_presets(self) -> List[str]:
        """Get list of available brush presets"""
        return self.brush_engine.get_available_presets()

    def get_current_brush_info(self) -> Dict:
        """Get information about current brush"""
        info = self.brush_engine.get_current_brush_info() if self.use_advanced_brushes else {}
        info['mode'] = "Watercolor" if self.use_advanced_brushes else "Normal"
        info['advanced_brushes_enabled'] = self.use_advanced_brushes
        return info

    def adjust_brush_opacity(self, delta: float) -> float:
        """Adjust brush opacity"""
        current = self.brush_engine.current_settings.opacity
        new_opacity = max(0.1, min(1.0, current + delta))
        self.brush_engine.current_settings.opacity = new_opacity
        return new_opacity

    def adjust_brush_hardness(self, delta: float) -> float:
        """Adjust brush hardness"""
        current = self.brush_engine.current_settings.hardness
        new_hardness = max(0.0, min(1.0, current + delta))
        self.brush_engine.current_settings.hardness = new_hardness
        return new_hardness

    # Phase 4.1: Canvas Tools Interface
    def toggle_grid(self) -> bool:
        """Toggle grid overlay"""
        return self.canvas_tools.toggle_grid()

    def toggle_grid_type(self) -> str:
        """Toggle between normal and slanted grid"""
        grid_type = self.canvas_tools.toggle_grid_type()
        return grid_type.value

    def toggle_snap_to_grid(self) -> bool:
        """Toggle snap-to-grid functionality"""
        return self.canvas_tools.toggle_snap_to_grid()

    def toggle_drawing_mode(self):
        """Toggle between normal and pixel art drawing mode"""
        return self.canvas_tools.toggle_drawing_mode()

    def toggle_pixel_borders(self) -> bool:
        """Toggle black borders around pixels in pixel art mode"""
        return self.canvas_tools.toggle_pixel_borders()

    def cycle_grid_spacing(self) -> str:
        """Cycle through grid spacing options"""
        spacing = self.canvas_tools.cycle_grid_spacing()
        return f"{spacing.value}px"

    def get_canvas_tools_info(self) -> dict:
        """Get canvas tools state information"""
        return self.canvas_tools.get_state_info()

    def render_snap_indicator(self, frame: np.ndarray, point: Tuple[int, int]) -> None:
        """Render snap-to-grid indicator"""
        self.canvas_tools.render_snap_indicator(frame, point)
