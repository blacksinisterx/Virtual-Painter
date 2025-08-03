import cv2
import numpy as np
import math
from typing import Tuple, Optional, List

from config.settings import AppConfig
from core.color_manager import ColorState, ColorSuggestion
from ui.drawing_utils import draw_rounded_rectangle, draw_glass_effect, draw_modern_shadow
from ui.color_wheel import overlay_color_wheel, create_color_wheel_image
from core.exceptions import UIRenderingException

class UIRenderer:
    """Modern UI renderer with glass morphism effects"""

    def __init__(self, config: AppConfig, resolution: Tuple[int, int]):
        self.config = config
        self.resolution = resolution
        self.color_wheel_img = create_color_wheel_image(config.ui.WHEEL_RADIUS)

    def render_palette(self, frame: np.ndarray, hover_color: Optional[Tuple[int, int, int]],
                      color_state: ColorState, suggestions: Optional[List[ColorSuggestion]] = None,
                      recent_colors: Optional[List[Tuple[int, int, int]]] = None) -> None:
        """Render modern color palette with glass effects and intelligent features"""
        try:
            self._draw_palette_background(frame)
            self._draw_color_items(frame, hover_color, color_state)
            self._draw_custom_color_section(frame, color_state)

            # New intelligent features - only show if enabled
            if color_state.show_smart_features:
                if suggestions:
                    self._draw_smart_suggestions(frame, suggestions)
                if recent_colors:
                    self._draw_recent_colors(frame, recent_colors)
            else:
                # Show a small indicator that smart features are disabled
                self._draw_smart_features_disabled_indicator(frame)

            if color_state.show_color_wheel:
                self._draw_color_wheel(frame)

        except Exception as e:
            raise UIRenderingException(f"Palette rendering failed: {e}")

    def _draw_palette_background(self, frame: np.ndarray) -> None:
        """Draw modern palette background with glass effect"""
        palette_width = 120
        palette_height = self.config.ui.PALETTE_ROWS * self.config.ui.PALETTE_SPACING + 50

        # Modern shadow
        draw_modern_shadow(frame,
                          (self.config.ui.PALETTE_START_X - 10, self.config.ui.PALETTE_START_Y - 30),
                          (self.config.ui.PALETTE_START_X + palette_width,
                           self.config.ui.PALETTE_START_Y + palette_height))

        # Glass effect background
        draw_glass_effect(frame,
                         (self.config.ui.PALETTE_START_X - 10, self.config.ui.PALETTE_START_Y - 30),
                         (self.config.ui.PALETTE_START_X + palette_width,
                          self.config.ui.PALETTE_START_Y + palette_height))

    def _draw_color_items(self, frame: np.ndarray, hover_color: Optional[Tuple[int, int, int]],
                         color_state: ColorState) -> None:
        """Draw color palette items with selection indicators"""
        for i, (color, name, gradient_color) in enumerate(self.config.COLOR_PALETTE):
            x = self.config.ui.PALETTE_START_X
            y = self.config.ui.PALETTE_START_Y + i * self.config.ui.PALETTE_SPACING

            # Selection indicator
            if color == color_state.current_color and not color_state.using_custom_color:
                self._draw_selection_indicator(frame, x, y)

            # Hover effect
            if hover_color == color:
                self._draw_hover_effect(frame, x, y)

            # Color square with gradient
            self._draw_gradient_square(frame, (x, y), color, gradient_color)

    def _draw_gradient_square(self, frame: np.ndarray, pos: Tuple[int, int],
                            color: Tuple[int, int, int], gradient_color: Tuple[int, int, int]) -> None:
        """Draw color square with gradient effect and rounded corners"""
        x, y = pos
        size = self.config.ui.PALETTE_ITEM_SIZE
        border_radius = 8

        # Create a temporary overlay for the rounded gradient
        overlay = np.zeros((size, size, 3), dtype=np.uint8)

        # Create gradient
        for i in range(size):
            alpha = i / size
            blended_color = tuple(int(c1 * (1 - alpha) + c2 * alpha)
                                for c1, c2 in zip(color, gradient_color))
            cv2.line(overlay, (i, 0), (i, size), blended_color, 1)

        # Create a mask for rounded corners
        mask = np.zeros((size, size), dtype=np.uint8)
        draw_rounded_rectangle(mask, (0, 0), (size, size), 255, -1, border_radius)

        # Apply the mask to the gradient
        rounded_gradient = cv2.bitwise_and(overlay, overlay, mask=mask)

        # Create the inverse mask for the frame area
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame[y:y+size, x:x+size], frame[y:y+size, x:x+size], mask=mask_inv)

        # Combine the rounded gradient with the frame background
        result = cv2.add(frame_bg, rounded_gradient)
        frame[y:y+size, x:x+size] = result

        # Draw border on top
        draw_rounded_rectangle(frame, (x, y), (x + size, y + size), (255, 255, 255), 2, border_radius)

    def _draw_selection_indicator(self, frame: np.ndarray, x: int, y: int) -> None:
        """Draw selection indicator with pulse effect"""
        size = self.config.ui.PALETTE_ITEM_SIZE
        draw_rounded_rectangle(frame, (x - 3, y - 3), (x + size + 3, y + size + 3),
                             (0, 255, 0), 3, 10)

    def _draw_hover_effect(self, frame: np.ndarray, x: int, y: int) -> None:
        """Draw hover effect"""
        size = self.config.ui.PALETTE_ITEM_SIZE
        draw_rounded_rectangle(frame, (x - 2, y - 2), (x + size + 2, y + size + 2),
                             (255, 255, 0), 2, 8)

    def _draw_custom_color_section(self, frame: np.ndarray, color_state: ColorState) -> None:
        """Draw custom color section"""
        # Custom color square
        square_pos = self.config.ui.CUSTOM_SQUARE_POS
        size = self.config.ui.CUSTOM_SQUARE_SIZE
        border_radius = 8

        if color_state.using_custom_color:
            self._draw_selection_indicator(frame, square_pos[0], square_pos[1])

        # Draw custom color with rounded corners
        draw_rounded_rectangle(frame, square_pos,
                             (square_pos[0] + size, square_pos[1] + size),
                             color_state.custom_color, -1, border_radius)

        # Draw border on top
        draw_rounded_rectangle(frame, square_pos,
                             (square_pos[0] + size, square_pos[1] + size),
                             (255, 255, 255), 2, border_radius)

        # Custom button
        button_pos = self.config.ui.CUSTOM_BUTTON_POS
        button_size = self.config.ui.CUSTOM_BUTTON_SIZE

        draw_glass_effect(frame, button_pos,
                         (button_pos[0] + button_size[0], button_pos[1] + button_size[1]))
        draw_rounded_rectangle(frame, button_pos,
                             (button_pos[0] + button_size[0], button_pos[1] + button_size[1]),
                             (255, 255, 255), 2, 8)

        cv2.putText(frame, "Custom", (button_pos[0] + 10, button_pos[1] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_color_wheel(self, frame: np.ndarray) -> None:
        """Draw color wheel overlay"""
        center = self.config.ui.WHEEL_CENTER
        radius = self.config.ui.WHEEL_RADIUS

        overlay_color_wheel(frame, center[0], center[1], radius, self.color_wheel_img)

        # Border
        cv2.circle(frame, center, radius + 2, (255, 255, 255), 2)

    def _draw_smart_suggestions(self, frame: np.ndarray, suggestions: List[ColorSuggestion]) -> None:
        """Draw intelligent color suggestions"""
        if not suggestions:
            return

        # Position suggestions to the right of the palette
        start_x = self.config.ui.PALETTE_START_X + 140
        start_y = self.config.ui.PALETTE_START_Y

        # Background for suggestions
        bg_width = 200
        bg_height = len(suggestions) * 50 + 40

        draw_modern_shadow(frame, (start_x - 10, start_y - 10),
                          (start_x + bg_width, start_y + bg_height))
        draw_glass_effect(frame, (start_x - 10, start_y - 10),
                         (start_x + bg_width, start_y + bg_height))
        draw_rounded_rectangle(frame, (start_x - 10, start_y - 10),
                             (start_x + bg_width, start_y + bg_height),
                             (255, 255, 255), 1, 10)

        # Title
        cv2.putText(frame, "Smart Suggestions", (start_x, start_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw each suggestion
        for i, suggestion in enumerate(suggestions):
            y_pos = start_y + 40 + i * 50

            # Color swatch
            swatch_size = 30
            draw_rounded_rectangle(frame, (start_x, y_pos),
                                 (start_x + swatch_size, y_pos + swatch_size),
                                 suggestion.color, -1, 5)
            draw_rounded_rectangle(frame, (start_x, y_pos),
                                 (start_x + swatch_size, y_pos + swatch_size),
                                 (255, 255, 255), 1, 5)

            # Suggestion reason and confidence
            reason_text = suggestion.reason
            if len(reason_text) > 20:
                reason_text = reason_text[:17] + "..."

            cv2.putText(frame, reason_text, (start_x + 40, y_pos + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Confidence bar
            confidence_width = int(suggestion.confidence * 80)
            cv2.rectangle(frame, (start_x + 40, y_pos + 20),
                         (start_x + 40 + confidence_width, y_pos + 25),
                         (100, 255, 100), -1)

    def _draw_recent_colors(self, frame: np.ndarray, recent_colors: List[Tuple[int, int, int]]) -> None:
        """Draw recently used colors"""
        if not recent_colors:
            return

        # Position below suggestions or main palette
        start_x = self.config.ui.PALETTE_START_X + 140
        start_y = self.config.ui.PALETTE_START_Y + 300  # Below suggestions

        # Background for recent colors
        bg_width = 200
        bg_height = 80

        draw_modern_shadow(frame, (start_x - 10, start_y - 10),
                          (start_x + bg_width, start_y + bg_height))
        draw_glass_effect(frame, (start_x - 10, start_y - 10),
                         (start_x + bg_width, start_y + bg_height))
        draw_rounded_rectangle(frame, (start_x - 10, start_y - 10),
                             (start_x + bg_width, start_y + bg_height),
                             (255, 255, 255), 1, 10)

        # Title
        cv2.putText(frame, "Recent Colors", (start_x, start_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw recent colors in a grid
        colors_per_row = 6
        swatch_size = 25
        padding = 5

        for i, color in enumerate(recent_colors[:12]):  # Max 12 recent colors
            row = i // colors_per_row
            col = i % colors_per_row

            x_pos = start_x + col * (swatch_size + padding)
            y_pos = start_y + 30 + row * (swatch_size + padding)

            draw_rounded_rectangle(frame, (x_pos, y_pos),
                                 (x_pos + swatch_size, y_pos + swatch_size),
                                 color, -1, 3)
            draw_rounded_rectangle(frame, (x_pos, y_pos),
                                 (x_pos + swatch_size, y_pos + swatch_size),
                                 (255, 255, 255), 1, 3)

    def _draw_smart_features_disabled_indicator(self, frame: np.ndarray) -> None:
        """Draw a small indicator when smart features are disabled"""
        # Position to the right of the palette
        start_x = self.config.ui.PALETTE_START_X + 140
        start_y = self.config.ui.PALETTE_START_Y

        # Small indicator box
        bg_width = 200
        bg_height = 60

        draw_glass_effect(frame, (start_x - 10, start_y - 10),
                         (start_x + bg_width, start_y + bg_height))
        draw_rounded_rectangle(frame, (start_x - 10, start_y - 10),
                             (start_x + bg_width, start_y + bg_height),
                             (100, 100, 100), 1, 10)

        # Message
        cv2.putText(frame, "Smart Features OFF", (start_x, start_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        cv2.putText(frame, "Press 'T' to enable", (start_x, start_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
