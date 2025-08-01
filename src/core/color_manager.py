import cv2
import numpy as np
import math
import time
import json
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Set
from dataclasses import dataclass, field
from collections import deque, Counter

from config.settings import AppConfig, UIConfig
from core.exceptions import ColorManagementException

@dataclass
class ColorState:
    """Represents the current color selection state"""
    current_color: Tuple[int, int, int]
    custom_color: Tuple[int, int, int]
    using_custom_color: bool
    show_color_wheel: bool
    show_smart_features: bool = True  # New toggle for intelligent features

@dataclass
class ColorHistoryEntry:
    """Represents a color history entry with metadata"""
    color: Tuple[int, int, int]
    timestamp: float
    usage_count: int = 1
    context: str = "manual"  # manual, suggested, harmony

@dataclass
class ColorSuggestion:
    """Represents a color suggestion with reasoning"""
    color: Tuple[int, int, int]
    reason: str
    confidence: float
    harmony_type: str = ""

class ColorIntelligence:
    """Advanced color intelligence and harmony system"""
    
    def __init__(self):
        # Color history management
        self.color_history: deque = deque(maxlen=50)
        self.color_usage: Counter = Counter()
        self.color_combinations: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = {}
        
        # Preferences file
        self.preferences_file = Path("assets/color_preferences.json")
        self._load_preferences()
    
    def _load_preferences(self) -> None:
        """Load color preferences from file"""
        try:
            if self.preferences_file.exists():
                with open(self.preferences_file, 'r') as f:
                    data = json.load(f)
                    
                # Restore history
                for entry_data in data.get('history', []):
                    entry = ColorHistoryEntry(
                        color=tuple(entry_data['color']),
                        timestamp=entry_data['timestamp'],
                        usage_count=entry_data.get('usage_count', 1),
                        context=entry_data.get('context', 'manual')
                    )
                    self.color_history.append(entry)
                
                # Restore usage counts
                for color_str, count in data.get('usage', {}).items():
                    color = tuple(map(int, color_str.split(',')))
                    self.color_usage[color] = count
                    
        except Exception as e:
            print(f"Could not load color preferences: {e}")
    
    def _save_preferences(self) -> None:
        """Save color preferences to file"""
        try:
            # Ensure directory exists
            self.preferences_file.parent.mkdir(exist_ok=True)
            
            data = {
                'history': [
                    {
                        'color': list(entry.color),
                        'timestamp': entry.timestamp,
                        'usage_count': entry.usage_count,
                        'context': entry.context
                    } for entry in self.color_history
                ],
                'usage': {
                    f"{color[0]},{color[1]},{color[2]}": count
                    for color, count in self.color_usage.items()
                }
            }
            
            with open(self.preferences_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Could not save color preferences: {e}")
    
    def add_color_usage(self, color: Tuple[int, int, int], context: str = "manual") -> None:
        """Add color usage to history and statistics"""
        # Update usage counter
        self.color_usage[color] += 1
        
        # Add or update history entry
        current_time = time.time()
        
        # Check if color is already in recent history
        for entry in reversed(self.color_history):
            if entry.color == color and current_time - entry.timestamp < 300:  # 5 minutes
                entry.usage_count += 1
                entry.timestamp = current_time
                return
        
        # Add new history entry
        entry = ColorHistoryEntry(
            color=color,
            timestamp=current_time,
            context=context
        )
        self.color_history.append(entry)
        
        # Save preferences periodically
        if len(self.color_history) % 10 == 0:
            self._save_preferences()
    
    def get_color_harmony(self, base_color: Tuple[int, int, int], harmony_type: str) -> List[Tuple[int, int, int]]:
        """Generate color harmony based on color theory"""
        # Convert BGR to HSV
        hsv = cv2.cvtColor(np.array([[base_color]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        
        harmonies = []
        
        if harmony_type == "complementary":
            # Complementary color (opposite on color wheel)
            comp_h = (h + 90) % 180  # HSV hue is 0-179
            harmonies.append(self._hsv_to_bgr(comp_h, s, v))
            
        elif harmony_type == "analogous":
            # Analogous colors (adjacent on color wheel)
            for offset in [-30, -15, 15, 30]:
                analog_h = (h + offset) % 180
                harmonies.append(self._hsv_to_bgr(analog_h, s, v))
                
        elif harmony_type == "triadic":
            # Triadic colors (120 degrees apart)
            for offset in [60, 120]:
                triadic_h = (h + offset) % 180
                harmonies.append(self._hsv_to_bgr(triadic_h, s, v))
                
        elif harmony_type == "split_complementary":
            # Split complementary (complement +/- 30 degrees)
            comp_h = (h + 90) % 180
            for offset in [-30, 30]:
                split_h = (comp_h + offset) % 180
                harmonies.append(self._hsv_to_bgr(split_h, s, v))
                
        elif harmony_type == "monochromatic":
            # Monochromatic (same hue, different saturation/value)
            for s_offset, v_offset in [(40, 20), (20, 40), (-20, 20), (-40, -20)]:
                mono_s = max(0, min(255, s + s_offset))
                mono_v = max(0, min(255, v + v_offset))
                harmonies.append(self._hsv_to_bgr(h, mono_s, mono_v))
        
        return harmonies
    
    def _hsv_to_bgr(self, h: int, s: int, v: int) -> Tuple[int, int, int]:
        """Convert HSV to BGR color"""
        try:
            hsv = np.array([[[h, s, v]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return tuple(map(int, bgr[0][0]))
        except Exception as e:
            return (0, 0, 0)
    
    def get_smart_suggestions(self, current_color: Tuple[int, int, int], max_suggestions: int = 4) -> List[ColorSuggestion]:
        """Get intelligent color suggestions based on history and harmony"""
        suggestions = []
        
        # 1. Most used colors from history
        if self.color_usage:
            most_used = self.color_usage.most_common(2)
            for color, count in most_used[:2]:
                if color != current_color:
                    confidence = min(1.0, count / 20)  # Normalize to 0-1
                    suggestions.append(ColorSuggestion(
                        color=color,
                        reason=f"Frequently used ({count}x)",
                        confidence=confidence
                    ))
        
        # 2. Recent colors from history
        recent_colors = []
        current_time = time.time()
        for entry in reversed(self.color_history):
            if current_time - entry.timestamp < 3600 and entry.color != current_color:  # Last hour
                if entry.color not in recent_colors:
                    recent_colors.append(entry.color)
                if len(recent_colors) >= 1:
                    break
        
        for color in recent_colors:
            suggestions.append(ColorSuggestion(
                color=color,
                reason="Recently used",
                confidence=0.7
            ))
        
        # 3. Complementary color
        complementary = self.get_color_harmony(current_color, "complementary")
        if complementary:
            suggestions.append(ColorSuggestion(
                color=complementary[0],
                reason="Complementary",
                confidence=0.8,
                harmony_type="complementary"
            ))
        
        # 4. Analogous color
        analogous = self.get_color_harmony(current_color, "analogous")
        if analogous:
            suggestions.append(ColorSuggestion(
                color=analogous[1],  # Pick middle analogous color
                reason="Analogous harmony",
                confidence=0.6,
                harmony_type="analogous"
            ))
        
        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return suggestions[:max_suggestions]
    
    def get_recent_colors(self, max_colors: int = 8) -> List[Tuple[int, int, int]]:
        """Get recently used colors"""
        recent = []
        current_time = time.time()
        
        for entry in reversed(self.color_history):
            if entry.color not in recent and current_time - entry.timestamp < 7200:  # Last 2 hours
                recent.append(entry.color)
                if len(recent) >= max_colors:
                    break
        
        return recent
    
    def get_popular_colors(self, max_colors: int = 6) -> List[Tuple[int, int, int]]:
        """Get most popular colors from usage statistics"""
        if not self.color_usage:
            return []
        
        return [color for color, _ in self.color_usage.most_common(max_colors)]

class ColorManager:
    """Enhanced color manager with intelligent features and history"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.state = ColorState(
            current_color=config.COLOR_PALETTE[0][0],
            custom_color=(0, 0, 0),
            using_custom_color=False,
            show_color_wheel=False,
            show_smart_features=True
        )
        
        # Timing for color selection
        self.wheel_hover_start_time = 0
        self.palette_hover_start_time = 0
        self.palette_last_hovered_color: Optional[Tuple[int, int, int]] = None
        
        # Intelligence system
        self.intelligence = ColorIntelligence()
        
        # Track current color for suggestions
        self.last_color_change_time = 0
        
        # Pre-compute color wheel for performance
        self.color_wheel_img = self._create_color_wheel_image()
        
        # Add initial color to history
        self.intelligence.add_color_usage(self.state.current_color, "initial")
    
    def _hsv_to_bgr(self, h: int, s: int, v: int) -> Tuple[int, int, int]:
        """Convert HSV to BGR color"""
        try:
            hsv = np.array([[[h, s, v]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return tuple(map(int, bgr[0][0]))
        except Exception as e:
            return (0, 0, 0)
    
    def _create_color_wheel_image(self) -> np.ndarray:
        """Create enhanced color wheel with smooth gradients"""
        try:
            radius = self.config.ui.WHEEL_RADIUS
            size = radius * 2 + 1
            wheel_img = np.zeros((size, size, 3), dtype=np.uint8)
            
            for y in range(size):
                for x in range(size):
                    dx = x - radius
                    dy = y - radius
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance <= radius:
                        angle = math.atan2(dy, dx)
                        hue = int((angle + math.pi) * 180 / (2 * math.pi)) % 180
                        saturation = int((distance / radius) * 255)
                        value = 255
                        color = self._hsv_to_bgr(hue, saturation, value)
                        wheel_img[y, x] = color
            return wheel_img
            
        except Exception as e:
            raise ColorManagementException(f"Color wheel creation failed: {e}")
    
    def get_color_from_wheel_position(self, x: int, y: int) -> Optional[Tuple[int, int, int]]:
        """Get color based on position in color wheel"""
        try:
            center_x, center_y = self.config.ui.WHEEL_CENTER
            radius = self.config.ui.WHEEL_RADIUS
            
            dx = x - center_x
            dy = y - center_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > radius:
                return None
            
            angle = math.atan2(dy, dx)
            hue = int((angle + math.pi) * 180 / (2 * math.pi)) % 180
            saturation = int((distance / radius) * 255)
            value = 255
            
            return self._hsv_to_bgr(hue, saturation, value)
            
        except Exception as e:
            return None
    
    def handle_color_selection(self, x: int, y: int) -> Optional[str]:
        """Handle color selection logic with timing"""
        try:
            current_time = time.time()
            
            if self.state.show_color_wheel:
                return self._handle_wheel_selection(x, y, current_time)
            else:
                return self._handle_palette_selection(x, y, current_time)
                
        except Exception as e:
            return None
    
    def _handle_wheel_selection(self, x: int, y: int, current_time: float) -> Optional[str]:
        """Handle color wheel selection"""
        wheel_color = self.get_color_from_wheel_position(x, y)
        
        if wheel_color:
            self.state.custom_color = wheel_color
            if self.wheel_hover_start_time == 0:
                self.wheel_hover_start_time = current_time
            elif current_time - self.wheel_hover_start_time > self.config.drawing.SELECTION_DELAY:
                if self.state.current_color != wheel_color:
                    self.state.current_color = wheel_color
                    self.state.custom_color = wheel_color
                    self.state.using_custom_color = True
                    self.state.show_color_wheel = False
                    self.wheel_hover_start_time = 0
                    self.last_color_change_time = current_time
                    # Track color usage
                    self.intelligence.add_color_usage(wheel_color, "wheel_selection")
                    return "Custom color selected"
        else:
            self.wheel_hover_start_time = 0
        
        return None
    
    def _handle_palette_selection(self, x: int, y: int, current_time: float) -> Optional[str]:
        """Handle palette color selection"""
        # Check custom button
        button_pos = self.config.ui.CUSTOM_BUTTON_POS
        button_size = self.config.ui.CUSTOM_BUTTON_SIZE
        
        if (button_pos[0] < x < button_pos[0] + button_size[0] and
            button_pos[1] < y < button_pos[1] + button_size[1]):
            self.state.show_color_wheel = True
            return "Color wheel opened"
        
        # Check palette colors
        hovered_color = self._get_hovered_palette_color(x, y)
        
        if hovered_color is not None:
            if self.palette_last_hovered_color != hovered_color:
                self.palette_hover_start_time = current_time
                self.palette_last_hovered_color = hovered_color
            elif current_time - self.palette_hover_start_time > self.config.drawing.SELECTION_DELAY:
                if self.state.current_color != hovered_color:
                    self.state.current_color = hovered_color
                    self.state.using_custom_color = False
                    self.palette_hover_start_time = 0
                    self.palette_last_hovered_color = None
                    self.last_color_change_time = current_time
                    
                    # Track color usage
                    self.intelligence.add_color_usage(hovered_color, "palette_selection")
                    
                    # Find color name
                    color_name = next((name for color, name, _ in self.config.COLOR_PALETTE 
                                     if color == hovered_color), "Unknown")
                    return f"{color_name} selected"
        else:
            self.palette_hover_start_time = 0
            self.palette_last_hovered_color = None
        
        return None
    
    def _get_hovered_palette_color(self, x: int, y: int) -> Optional[Tuple[int, int, int]]:
        """Get the palette color being hovered over"""
        for i, (color, _, _) in enumerate(self.config.COLOR_PALETTE):
            palette_x = self.config.ui.PALETTE_START_X
            palette_y = self.config.ui.PALETTE_START_Y + i * self.config.ui.PALETTE_SPACING
            
            if (palette_x < x < palette_x + self.config.ui.PALETTE_ITEM_SIZE and
                palette_y < y < palette_y + self.config.ui.PALETTE_ITEM_SIZE):
                return color
        
        return None
    
    def get_current_color(self) -> Tuple[int, int, int]:
        """Get the currently selected color"""
        return self.state.current_color
    
    def get_state(self) -> ColorState:
        """Get current color state"""
        return self.state
    
    def close_color_wheel(self) -> None:
        """Close the color wheel"""
        self.state.show_color_wheel = False
    
    # New intelligent features
    def get_smart_suggestions(self) -> List[ColorSuggestion]:
        """Get intelligent color suggestions based on current color and history"""
        return self.intelligence.get_smart_suggestions(self.state.current_color)
    
    def get_recent_colors(self) -> List[Tuple[int, int, int]]:
        """Get recently used colors"""
        return self.intelligence.get_recent_colors()
    
    def get_popular_colors(self) -> List[Tuple[int, int, int]]:
        """Get most popular colors"""
        return self.intelligence.get_popular_colors()
    
    def get_color_harmony(self, harmony_type: str) -> List[Tuple[int, int, int]]:
        """Get color harmony for current color"""
        return self.intelligence.get_color_harmony(self.state.current_color, harmony_type)
    
    def set_color_from_suggestion(self, color: Tuple[int, int, int], suggestion_type: str = "suggestion") -> None:
        """Set color from a suggestion and track it"""
        if self.state.current_color != color:
            self.state.current_color = color
            self.state.using_custom_color = True  # Mark as custom since it's not from palette
            self.last_color_change_time = time.time()
            self.intelligence.add_color_usage(color, suggestion_type)
    
    def get_intelligence_stats(self) -> Dict:
        """Get color intelligence statistics"""
        return {
            'total_colors_used': len(self.intelligence.color_usage),
            'most_popular_color': self.intelligence.color_usage.most_common(1)[0] if self.intelligence.color_usage else None,
            'recent_colors_count': len(self.intelligence.get_recent_colors()),
            'history_size': len(self.intelligence.color_history)
        }
    
    def save_preferences(self) -> None:
        """Manually save color preferences"""
        self.intelligence._save_preferences()
    
    def toggle_smart_features(self) -> bool:
        """Toggle smart color features visibility"""
        self.state.show_smart_features = not self.state.show_smart_features
        return self.state.show_smart_features
