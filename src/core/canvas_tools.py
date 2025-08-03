"""
Canvas Tools System - Professional drawing aids for precision work
Provides grid overlay, rulers, and snap-to-grid functionality
"""

import cv2
import numpy as np
import math
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

class GridSpacing(Enum):
    """Grid spacing options"""
    SMALL = 20
    MEDIUM = 40
    LARGE = 60
    XLARGE = 80

class GridType(Enum):
    """Grid type options"""
    NORMAL = "normal"      # 0° and 90° lines
    SLANTED = "slanted"    # 45° and -45° lines

class DrawingMode(Enum):
    """Drawing mode options"""
    NORMAL = "normal"      # Regular brush drawing
    PIXEL_ART = "pixel"    # Pixel art mode (fill grid squares)

@dataclass
class CanvasToolsState:
    """State management for canvas tools"""
    grid_enabled: bool = False
    snap_to_grid_enabled: bool = False
    grid_spacing: GridSpacing = GridSpacing.MEDIUM
    grid_type: GridType = GridType.NORMAL
    grid_opacity: float = 0.3
    drawing_mode: DrawingMode = DrawingMode.NORMAL
    pixel_borders: bool = True  # Black borders around pixels in pixel art mode

class CanvasTools:
    """Professional canvas tools for enhanced drawing precision"""

    def __init__(self, resolution: Tuple[int, int]):
        """
        Initialize canvas tools system

        Args:
            resolution: (width, height) of the canvas
        """
        self.width, self.height = resolution
        self.state = CanvasToolsState()

        # Visual styling
        self.grid_color = (80, 80, 80)  # Subtle gray

    def toggle_grid(self) -> bool:
        """Toggle grid overlay"""
        self.state.grid_enabled = not self.state.grid_enabled
        return self.state.grid_enabled

    def toggle_grid_type(self) -> GridType:
        """Toggle between normal and slanted grid"""
        if self.state.grid_type == GridType.NORMAL:
            self.state.grid_type = GridType.SLANTED
        else:
            self.state.grid_type = GridType.NORMAL
        return self.state.grid_type

    def toggle_snap_to_grid(self) -> bool:
        """Toggle snap-to-grid functionality"""
        self.state.snap_to_grid_enabled = not self.state.snap_to_grid_enabled
        return self.state.snap_to_grid_enabled

    def toggle_drawing_mode(self) -> DrawingMode:
        """Toggle between normal and pixel art drawing mode"""
        if self.state.drawing_mode == DrawingMode.NORMAL:
            self.state.drawing_mode = DrawingMode.PIXEL_ART
        else:
            self.state.drawing_mode = DrawingMode.NORMAL
        return self.state.drawing_mode

    def toggle_pixel_borders(self) -> bool:
        """Toggle black borders around pixels in pixel art mode"""
        self.state.pixel_borders = not self.state.pixel_borders
        return self.state.pixel_borders

    def get_grid_square_bounds(self, point: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Get the bounds of the grid square containing the given point

        Args:
            point: (x, y) coordinate

        Returns:
            (x1, y1, x2, y2) bounds of the grid square
        """
        x, y = point
        spacing = self.state.grid_spacing.value

        if self.state.grid_type == GridType.NORMAL:
            # Normal grid: rectangular squares
            grid_x = (x // spacing) * spacing
            grid_y = (y // spacing) * spacing
            return (grid_x, grid_y, grid_x + spacing, grid_y + spacing)
        else:
            # Slanted grid: diamond-shaped cells
            # Use the SAME logic as snap_point_to_grid for consistency

            # Find the nearest normal grid intersection (same as snap-to-grid does)
            grid_x = round(x / spacing) * spacing
            grid_y = round(y / spacing) * spacing

            # Get all diamond intersection candidates (same as snap-to-grid)
            candidates = [
                (grid_x + spacing//2, grid_y + spacing//2),  # Bottom-right diamond point
                (grid_x - spacing//2, grid_y + spacing//2),  # Bottom-left diamond point
                (grid_x + spacing//2, grid_y - spacing//2),  # Top-right diamond point
                (grid_x - spacing//2, grid_y - spacing//2),  # Top-left diamond point
                (grid_x, grid_y)  # Center grid point
            ]

            # Find the closest candidate (same as snap-to-grid)
            min_distance = float('inf')
            best_point = (grid_x, grid_y)

            for candidate in candidates:
                distance = math.sqrt((x - candidate[0])**2 + (y - candidate[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    best_point = candidate

            # Use the closest snap point as the diamond center
            center_x, center_y = best_point

            # Diamond size - same as normal grid squares
            half_size = spacing // 2

            return (
                center_x - half_size,
                center_y - half_size,
                center_x + half_size,
                center_y + half_size
            )

    def cycle_grid_spacing(self) -> GridSpacing:
        """Cycle through grid spacing options"""
        spacings = list(GridSpacing)
        current_index = spacings.index(self.state.grid_spacing)
        next_index = (current_index + 1) % len(spacings)
        self.state.grid_spacing = spacings[next_index]
        return self.state.grid_spacing

    def snap_point_to_grid(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """
        Snap a point to the nearest grid intersection

        Args:
            point: (x, y) coordinate to snap

        Returns:
            Snapped (x, y) coordinate
        """
        if not self.state.snap_to_grid_enabled:
            return point

        x, y = point
        spacing = self.state.grid_spacing.value

        if self.state.grid_type == GridType.NORMAL:
            # Normal grid: snap to horizontal/vertical intersections
            snapped_x = round(x / spacing) * spacing
            snapped_y = round(y / spacing) * spacing
        else:
            # Slanted grid: snap to diamond intersections aligned with normal grid
            # The slanted grid creates diamond shapes where intersections occur at
            # the midpoints between normal grid intersections

            # Find the nearest normal grid intersection
            grid_x = round(x / spacing) * spacing
            grid_y = round(y / spacing) * spacing

            # Check all four possible diamond intersection points around this grid point
            candidates = [
                (grid_x + spacing//2, grid_y + spacing//2),  # Bottom-right diamond point
                (grid_x - spacing//2, grid_y + spacing//2),  # Bottom-left diamond point
                (grid_x + spacing//2, grid_y - spacing//2),  # Top-right diamond point
                (grid_x - spacing//2, grid_y - spacing//2),  # Top-left diamond point
                (grid_x, grid_y)  # Center grid point
            ]

            # Find the closest candidate
            min_distance = float('inf')
            best_point = (grid_x, grid_y)

            for candidate in candidates:
                distance = math.sqrt((x - candidate[0])**2 + (y - candidate[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    best_point = candidate

            snapped_x, snapped_y = best_point

        return (int(snapped_x), int(snapped_y))

    def constrain_to_grid_direction(self, start_point: Tuple[int, int], current_point: Tuple[int, int]) -> Tuple[int, int]:
        """
        Constrain drawing to grid direction (horizontal/vertical for normal, diagonal for slanted)

        Args:
            start_point: Starting point of the line
            current_point: Current drawing point

        Returns:
            Constrained point that aligns with grid direction
        """
        if not self.state.snap_to_grid_enabled:
            return current_point

        start_x, start_y = start_point
        curr_x, curr_y = current_point

        dx = curr_x - start_x
        dy = curr_y - start_y

        if self.state.grid_type == GridType.NORMAL:
            # Normal grid: constrain to horizontal or vertical
            if abs(dx) > abs(dy):
                # More horizontal movement - constrain to horizontal
                return (curr_x, start_y)
            else:
                # More vertical movement - constrain to vertical
                return (start_x, curr_y)
        else:
            # Slanted grid: constrain to 45° diagonals (both directions)
            # Determine which diagonal direction is closer
            abs_dx = abs(dx)
            abs_dy = abs(dy)

            # Calculate the average distance for perfect diagonal
            avg_distance = (abs_dx + abs_dy) / 2

            # Determine signs for the diagonal direction
            sign_x = 1 if dx >= 0 else -1
            sign_y = 1 if dy >= 0 else -1

            # Choose the diagonal direction that's closer to the current movement
            if dx * dy >= 0:
                # Same signs - use 45° diagonal (both same direction)
                return (start_x + int(avg_distance * sign_x), start_y + int(avg_distance * sign_y))
            else:
                # Different signs - use -45° diagonal (opposite directions)
                return (start_x + int(avg_distance * sign_x), start_y - int(avg_distance * sign_x))

    def render_grid(self, frame: np.ndarray) -> None:
        """
        Render grid overlay on frame (normal or slanted)

        Args:
            frame: OpenCV frame to draw on
        """
        if not self.state.grid_enabled:
            return

        spacing = self.state.grid_spacing.value
        color = tuple(int(c * self.state.grid_opacity) for c in self.grid_color)

        if self.state.grid_type == GridType.NORMAL:
            self._render_normal_grid(frame, spacing, color)
        else:
            self._render_slanted_grid(frame, spacing, color)

    def _render_normal_grid(self, frame: np.ndarray, spacing: int, color: tuple) -> None:
        """Render normal horizontal/vertical grid"""
        # Draw vertical lines
        x = 0
        while x < self.width:
            cv2.line(frame, (x, 0), (x, self.height), color, 1)
            x += spacing

        # Draw horizontal lines
        y = 0
        while y < self.height:
            cv2.line(frame, (0, y), (self.width, y), color, 1)
            y += spacing

    def _render_slanted_grid(self, frame: np.ndarray, spacing: int, color: tuple) -> None:
        """Render slanted 45° grid aligned with normal grid intersections"""

        # Draw 45° diagonal lines that pass through normal grid intersections
        # These go from top-left to bottom-right
        for i in range(-self.height // spacing - 1, (self.width // spacing) + 2):
            # Start point on top or left edge
            start_x = i * spacing
            start_y = 0

            # End point on bottom or right edge
            end_x = start_x + self.height
            end_y = self.height

            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 1)

        # Draw perpendicular 45° lines (top-right to bottom-left)
        # These also pass through the same grid intersections
        for i in range(0, (self.width // spacing) + (self.height // spacing) + 2):
            # Start point on top or right edge
            start_x = i * spacing
            start_y = 0

            # End point on bottom or left edge
            end_x = start_x - self.height
            end_y = self.height

            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 1)

    def render_snap_indicator(self, frame: np.ndarray, point: Tuple[int, int]) -> None:
        """
        Render visual indicator when snap-to-grid is active

        Args:
            frame: OpenCV frame to draw on
            point: Current point position
        """
        if not self.state.snap_to_grid_enabled:
            return

        snapped_point = self.snap_point_to_grid(point)

        # Only show indicator if point would be snapped
        if snapped_point != point:
            # Draw small crosshair at snap position
            x, y = snapped_point
            size = 8

            # Different colors for different grid types
            if self.state.grid_type == GridType.NORMAL:
                color = (0, 255, 255)  # Cyan for normal grid
            else:
                color = (255, 0, 255)  # Magenta for slanted grid

            cv2.line(frame, (x - size, y), (x + size, y), color, 2)
            cv2.line(frame, (x, y - size), (x, y + size), color, 2)
            cv2.circle(frame, (x, y), 3, color, 2)

    def get_state_info(self) -> dict:
        """Get current state information"""
        return {
            'grid_enabled': self.state.grid_enabled,
            'snap_to_grid_enabled': self.state.snap_to_grid_enabled,
            'grid_spacing': self.state.grid_spacing.value,
            'grid_type': self.state.grid_type.value,
            'grid_opacity': self.state.grid_opacity,
            'drawing_mode': self.state.drawing_mode.value,
            'pixel_borders': self.state.pixel_borders
        }

    def get_help_text(self) -> List[str]:
        """Get help text for canvas tools"""
        return [
            "Canvas Tools:",
            "G - Toggle Grid",
            "F - Toggle Grid Type (Normal/Slanted)",
            "Ctrl+T - Toggle Snap-to-Grid",
            "P - Toggle Drawing Mode (Normal/Pixel Art)",
            "X - Toggle Pixel Borders (in Pixel Art mode)",
            "Shift+G - Cycle Grid Spacing (20px/40px/60px/80px)"
        ]
