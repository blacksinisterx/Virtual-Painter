import cv2
import time
from typing import Tuple
from dataclasses import dataclass

from config.settings import InfoPanelConfig
from core.hand_tracker import GestureType
from ui.drawing_utils import draw_rounded_rectangle, draw_glass_effect, draw_modern_shadow
from core.exceptions import UIRenderingException

@dataclass
class InfoPanelState:
    """State management for info panel"""
    show_panel: bool = False
    activated_time: float = 0
    cooldown_until: float = 0

class InfoPanelManager:
    """Manages info panel display and interactions"""
    
    def __init__(self, config: InfoPanelConfig, resolution: Tuple[int, int]):
        self.config = config
        self.resolution = resolution
        self.state = InfoPanelState()
    
    def handle_interaction(self, x: int, y: int, gesture_type: GestureType) -> None:
        """Handle info panel interaction"""
        try:
            import time
            current_time = time.time()
            
            # Check if we're in cooldown period
            if current_time < self.state.cooldown_until:
                return
            
            # Check if clicking on info button
            button_size = 40
            margin = 20
            button_x = max(10, self.resolution[0] - button_size - margin)
            button_y = margin
            
            if (button_x < x < button_x + button_size and 
                button_y < y < button_y + button_size and 
                gesture_type == GestureType.COLOR_SELECTION):
                
                self.state.show_panel = True
                self.state.activated_time = current_time
                    
        except Exception as e:
            pass  # Silently handle interaction errors
    
    def update_state(self) -> None:
        """Update info panel state (auto-hide logic)"""
        try:
            import time
            current_time = time.time()
            
            if self.state.show_panel:
                # Auto-hide after duration
                if current_time - self.state.activated_time > self.config.DISPLAY_DURATION:
                    self.state.show_panel = False
                    self.state.cooldown_until = current_time + self.config.COOLDOWN_DURATION
                
        except Exception as e:
            pass  # Silently handle update errors
    
    def render(self, frame) -> None:
        """Render info panel and info button"""
        try:
            self.update_state()
            
            # Always draw the info button
            self._draw_info_button(frame)
            
            # Draw panel if active
            if self.state.show_panel:
                self._draw_info_panel(frame)
                
        except Exception as e:
            pass  # Silently handle rendering errors
    
    def _draw_info_panel(self, frame) -> None:
        """Draw the main info panel with instructions in multi-column layout"""
        import time
        
        # Panel dimensions and positioning - increased width for multi-column
        panel_width = 650  # Increased from 400
        panel_height = 320  # Slightly increased height
        panel_x = (self.resolution[0] - panel_width) // 2
        panel_y = (self.resolution[1] - panel_height) // 2
        
        # Fade effect based on time
        current_time = time.time()
        time_shown = current_time - self.state.activated_time
        
        if time_shown > self.config.FADE_START_TIME:
            fade_progress = min(1.0, (time_shown - self.config.FADE_START_TIME) / 
                               (self.config.DISPLAY_DURATION - self.config.FADE_START_TIME))
            alpha = 0.9 * (1 - fade_progress)
        else:
            alpha = 0.9
        
        # Draw panel background
        draw_modern_shadow(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height))
        draw_glass_effect(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), alpha)
        draw_rounded_rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                             (255, 255, 255), 2, 15)
        
        # Multi-column layout configuration
        column_width = 200
        column_spacing = 25
        column1_x = panel_x + 20
        column2_x = column1_x + column_width + column_spacing
        column3_x = column2_x + column_width + column_spacing
        
        # Title
        cv2.putText(frame, "Virtual Painter Controls", (panel_x + 20, panel_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Column 1: Basic Controls
        column1_content = [
            "Hand Gestures:",
            "• Index only: Draw",
            "• All fingers: Erase", 
            "• Index+Thumb: Color",
            "• Index+Middle: Pen up",
            "• Hold+Draw: Size control",
            "",
            "Basic Controls:",
            "• 'S' - Save drawing",
            "• 'C' - Clear canvas",
            "• 'Z' - Undo action",
            "• 'Y' - Redo action"
        ]
        
        # Column 2: Brush & Canvas Tools
        column2_content = [
            "Brush Controls:",
            "• 'B' - Normal/Watercolor",
            "• 'O/L' - Opacity +/-",
            "• 'K/J' - Hardness +/-",
            "• 'M' - Brush info",
            "",
            "Canvas Tools:",
            "• 'G' - Toggle grid",
            "• 'U' - Toggle rulers",
            "• Ctrl+T - Snap-to-grid",
            "• Ctrl+G - Grid spacing",
            "• 'V' - Tools info"
        ]
        
        # Column 3: Color Intelligence
        column3_content = [
            "Color Intelligence:",
            "• 'T' - Smart features",
            "• 'H' - Color harmony",
            "• 'R' - Recent colors",
            "• 'P' - Popular colors",
            "• 'N' - Color stats",
            "",
            "Application:",
            "• 'A' - Anti-aliasing",
            "• 'I' - Canvas info",
            "• 'Q' - Quit",
            "",
            "Tips:",
            "• Draw smoothly",
            "• Use grid for precision"
        ]
        
        # Render columns
        self._render_column(frame, column1_content, column1_x, panel_y + 60)
        self._render_column(frame, column2_content, column2_x, panel_y + 60)
        self._render_column(frame, column3_content, column3_x, panel_y + 60)
    
    def _render_column(self, frame, content, x, start_y):
        """Render a single column of content"""
        y_offset = start_y
        for line in content:
            if line == "":
                y_offset += 12
                continue
            
            if line.endswith(":"):
                # Section header
                cv2.putText(frame, line, (x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                y_offset += 20
            else:
                # Regular content
                cv2.putText(frame, line, (x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 16
    
    def _draw_info_button(self, frame) -> None:
        """Draw modern info button with responsive positioning"""
        # Make sure button stays on screen
        button_size = 40
        margin = 20
        button_x = max(10, self.resolution[0] - button_size - margin)
        button_y = margin
        
        # Shadow and glass effect
        draw_modern_shadow(frame, (button_x, button_y), (button_x + button_size, button_y + button_size))
        draw_glass_effect(frame, (button_x, button_y), (button_x + button_size, button_y + button_size))
        
        # Border
        draw_rounded_rectangle(frame, (button_x, button_y), (button_x + button_size, button_y + button_size), 
                             (255, 255, 255), 2, 10)
        
        # Info icon (i)
        cv2.putText(frame, "i", (button_x + 15, button_y + 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def reset(self) -> None:
        """Reset info panel state"""
        self.state = InfoPanelState()