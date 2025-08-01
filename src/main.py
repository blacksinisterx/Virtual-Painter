import cv2
import time
from pathlib import Path

from config.settings import AppConfig
from core.camera import CameraManager
from core.hand_tracker import GestureDetector, GestureType
from core.color_manager import ColorManager
from core.canvas import DrawingEngine
from ui.ui_renderer import UIRenderer
from ui.info_panel import InfoPanelManager
from core.exceptions import VirtualPainterException

class VirtualPainterApp:
    """Main application orchestrator with clean separation of concerns"""
    
    def __init__(self):
        self.config = AppConfig()
        self.running = False
        
        # Dynamic sizing system
        self.current_brush_size = self.config.drawing.BRUSH_THICKNESS
        self.current_eraser_size = self.config.drawing.ERASER_THICKNESS
        
        # Phase 2: Rotation-based size control
        self.hold_rotation_angle = None  # Previous rotation angle for delta calculation
        self.size_control_active = False  # Track if size control mode is active
        self.rotation_sensitivity = 2.0  # Rotation degrees per size unit
        
        # Initialize core components
        self.camera_manager: CameraManager = None
        self.gesture_detector: GestureDetector = None
        self.color_manager: ColorManager = None
        self.drawing_engine: DrawingEngine = None
        self.ui_renderer: UIRenderer = None
        self.info_panel: InfoPanelManager = None
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all application components with error handling"""
        try:
            # Initialize camera first to get resolution
            self.camera_manager = CameraManager(self.config.camera)
            resolution = self.camera_manager.get_resolution()
            
            # Initialize other components
            self.gesture_detector = GestureDetector(self.config.drawing)
            self.color_manager = ColorManager(self.config)
            self.drawing_engine = DrawingEngine(self.config.drawing, resolution)
            self.ui_renderer = UIRenderer(self.config, resolution)
            self.info_panel = InfoPanelManager(self.config.info_panel, resolution)
            
        except Exception as e:
            raise VirtualPainterException(f"Application initialization failed: {e}")
    
    def run(self) -> None:
        """Main application loop with comprehensive error handling"""
        self.running = True
        
        try:
            while self.running:
                if not self._process_frame():
                    break
                    
                if not self._handle_keyboard_input():
                    break
                    
        except KeyboardInterrupt:
            pass  # Silent exit on user interrupt
        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
        finally:
            self._cleanup()
    
    def _process_frame(self) -> bool:
        """Process single frame - returns False if should exit"""
        try:
            # Capture frame
            frame = self.camera_manager.capture_frame()
            
            # Detect gestures (single MediaPipe call)
            hand_data, mp_results = self.gesture_detector.process_frame(frame)
            
            # Process each detected hand
            for hand_landmarks in hand_data:
                self._process_hand_gestures(hand_landmarks)
            
            # IMPORTANT: Reset drawing state if no hands detected to prevent line jumping
            if not hand_data:
                self.drawing_engine.process_gesture(GestureType.PEN_LIFTED, (0, 0), (0, 0, 0))
            
            # Phase 2: Process rotation-based size control
            self.process_rotation_size_control(hand_data)
            
            # Handle global UI state based on ALL hands
            self._update_global_ui_state(hand_data)
            
            # Merge drawing with frame
            frame = self.drawing_engine.merge_with_frame(frame)
            
            # Render UI with intelligent features
            hover_color = self._get_hover_color(hand_data)
            suggestions = self.color_manager.get_smart_suggestions()
            recent_colors = self.color_manager.get_recent_colors()
            
            self.ui_renderer.render_palette(frame, hover_color, self.color_manager.get_state(), 
                                          suggestions, recent_colors)
            self.info_panel.render(frame)
            
            # Draw hand landmarks (using existing MediaPipe results)
            self._draw_hand_landmarks(frame, hand_data, mp_results)
            
            # Display frame
            cv2.imshow("Ultra-Modern Virtual Painter", frame)
            
            return True
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return False
    
    def _process_hand_gestures(self, hand_landmarks) -> None:
        """Process gestures for a single hand"""
        landmarks, handedness, gesture_type = hand_landmarks
        
        if not landmarks:
            return
        
        # Get finger positions
        index_pos = landmarks[8]  # Index finger tip
        
        # Handle different gesture types
        if gesture_type in [GestureType.DRAWING, GestureType.ERASING, GestureType.PEN_LIFTED]:
            current_color = self.color_manager.get_current_color()
            
            # Use dynamic sizes instead of static config values
            if gesture_type == GestureType.DRAWING:
                status = self.drawing_engine.process_gesture(gesture_type, index_pos, current_color, self.current_brush_size)
            elif gesture_type == GestureType.ERASING:
                status = self.drawing_engine.process_gesture(gesture_type, index_pos, current_color, self.current_eraser_size)
            else:  # PEN_LIFTED
                status = self.drawing_engine.process_gesture(gesture_type, index_pos, current_color)
        
        elif gesture_type == GestureType.COLOR_SELECTION:
            # Handle color selection
            selection_result = self.color_manager.handle_color_selection(*index_pos)
            
            # Handle info panel interaction
            self.info_panel.handle_interaction(*index_pos, gesture_type)
        
        elif gesture_type == GestureType.HOLD:
            # HOLD gesture - circle will be drawn in _draw_hand_landmarks
            # Future functionality will be added here
            pass
            
            # Handle info panel interaction
            self.info_panel.handle_interaction(*index_pos, gesture_type)
        
        # Note: Global UI state (like color wheel) is handled after all hands are processed
    
    def get_current_brush_size(self) -> int:
        """Get current brush size"""
        return self.current_brush_size
    
    def get_current_eraser_size(self) -> int:
        """Get current eraser size"""
        return self.current_eraser_size
    
    def adjust_brush_size(self, delta: int) -> None:
        """Adjust brush size within limits"""
        self.current_brush_size = max(
            self.config.drawing.MIN_BRUSH_SIZE,
            min(self.config.drawing.MAX_BRUSH_SIZE, self.current_brush_size + delta)
        )
    
    def adjust_eraser_size(self, delta: int) -> None:
        """Adjust eraser size within limits"""
        self.current_eraser_size = max(
            self.config.drawing.MIN_ERASER_SIZE,
            min(self.config.drawing.MAX_ERASER_SIZE, self.current_eraser_size + delta)
        )
    
    def calculate_rotation_angle(self, landmarks: list) -> float:
        """Calculate rotation angle from HOLD gesture (thumb, index, middle finger)"""
        import math
        
        if len(landmarks) < 13:
            return 0.0
        
        # Get the three finger tips: thumb (4), index (8), middle (12)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        # Calculate centroid
        center_x = (thumb_tip[0] + index_tip[0] + middle_tip[0]) / 3
        center_y = (thumb_tip[1] + index_tip[1] + middle_tip[1]) / 3
        
        # Use index finger as primary reference for rotation
        # Calculate angle from center to index finger tip
        dx = index_tip[0] - center_x
        dy = index_tip[1] - center_y
        
        # Calculate angle in degrees (0-360)
        angle = math.degrees(math.atan2(dy, dx))
        # Normalize to 0-360 range
        return (angle + 360) % 360
    
    def process_rotation_size_control(self, hand_data) -> None:
        """Process rotation-based size control when HOLD + DRAW/ERASE gestures are active"""
        hold_hand = None
        draw_erase_hand = None
        
        # Find HOLD hand and DRAW/ERASE hand
        for hand_landmarks in hand_data:
            landmarks, handedness, gesture_type = hand_landmarks
            if gesture_type == GestureType.HOLD:
                hold_hand = hand_landmarks
            elif gesture_type in [GestureType.DRAWING, GestureType.ERASING]:
                draw_erase_hand = hand_landmarks
        
        # Check if both required gestures are present
        if hold_hand and draw_erase_hand:
            if not self.size_control_active:
                # Initialize size control mode
                self.size_control_active = True
                self.hold_rotation_angle = self.calculate_rotation_angle(hold_hand.landmarks)
            else:
                # Calculate rotation delta and adjust size
                current_angle = self.calculate_rotation_angle(hold_hand.landmarks)
                if self.hold_rotation_angle is not None:
                    # Calculate shortest rotation path
                    angle_delta = current_angle - self.hold_rotation_angle
                    
                    # Handle angle wraparound (e.g., 350° to 10°)
                    if angle_delta > 180:
                        angle_delta -= 360
                    elif angle_delta < -180:
                        angle_delta += 360
                    
                    # Convert rotation to size change
                    size_delta = int(angle_delta / self.rotation_sensitivity)
                    
                    if abs(size_delta) >= 1:  # Only adjust if significant rotation
                        if draw_erase_hand.gesture_type == GestureType.DRAWING:
                            self.adjust_brush_size(size_delta)
                        elif draw_erase_hand.gesture_type == GestureType.ERASING:
                            self.adjust_eraser_size(size_delta)
                        
                        self.hold_rotation_angle = current_angle
        else:
            # Reset size control mode if gestures are not present
            if self.size_control_active:
                self.size_control_active = False
                self.hold_rotation_angle = None
    
    def _update_global_ui_state(self, hand_data) -> None:
        """Update global UI state based on ALL hands (enables simultaneous multi-hand actions)"""
        # Check if ANY hand is doing color selection
        any_hand_selecting_color = any(
            gesture_type == GestureType.COLOR_SELECTION 
            for landmarks, handedness, gesture_type in hand_data
        )
        
        # Only close color wheel if NO hands are selecting colors
        if not any_hand_selecting_color:
            self.color_manager.close_color_wheel()
    
    def _get_hover_color(self, hand_data) -> tuple:
        """Get color being hovered over for UI feedback"""
        for hand_landmarks in hand_data:
            landmarks, _, gesture_type = hand_landmarks
            if landmarks and gesture_type == GestureType.COLOR_SELECTION:
                index_pos = landmarks[8]
                return self.color_manager._get_hovered_palette_color(*index_pos)
        return None
    
    def _draw_hand_landmarks(self, frame, hand_data, mp_results) -> None:
        """Draw hand landmarks with MediaPipe connections and gesture-specific colors"""
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        # Enhanced hand matching for multi-hand support
        if mp_results.multi_hand_landmarks and hand_data:
            # Try to match hands by handedness first, fall back to order
            if mp_results.multi_handedness and len(hand_data) == len(mp_results.multi_hand_landmarks):
                # Create handedness mapping
                mp_hands_by_handedness = {}
                for hand_landmarks, hand_type in zip(mp_results.multi_hand_landmarks, mp_results.multi_handedness):
                    mp_handedness = hand_type.classification[0].label
                    mp_hands_by_handedness[mp_handedness] = hand_landmarks
                
                # Match our data with MediaPipe data by handedness
                for gesture_data in hand_data:
                    landmarks, handedness, gesture_type = gesture_data.landmarks, gesture_data.handedness, gesture_data.gesture_type
                    
                    # Find corresponding MediaPipe hand
                    if handedness in mp_hands_by_handedness:
                        hand_landmarks = mp_hands_by_handedness[handedness]
                        
                        landmark_color, connection_color = self.gesture_detector.get_drawing_specs(gesture_type)
                        
                        # Custom drawing specs based on gesture
                        landmark_drawing_spec = mp_drawing.DrawingSpec(
                            color=landmark_color, thickness=2, circle_radius=4)
                        connection_drawing_spec = mp_drawing.DrawingSpec(
                            color=connection_color, thickness=2, circle_radius=2)
                        
                        # Draw hand landmarks and connections
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec, connection_drawing_spec)
                        
                        # Draw special gesture indicators
                        if len(landmarks) > 8:
                            index_tip = landmarks[8]
                            if gesture_type.value == "erasing":
                                # Show eraser size circle (radius = thickness/2 for accurate size)
                                cv2.circle(frame, index_tip, self.current_eraser_size // 2, (0, 100, 255), 2)
                            elif gesture_type.value == "drawing":
                                # Show brush size circle (radius = thickness/2 for accurate size)
                                cv2.circle(frame, index_tip, self.current_brush_size // 2, (100, 255, 100), 2)
                            elif gesture_type.value == "hold":
                                # Show hold circle passing through thumb, index, and middle finger tips
                                center, radius = self.gesture_detector.calculate_hold_circle(landmarks)
                                if radius > 0:  # Only draw if valid circle calculated
                                    # Change color based on size control mode
                                    circle_color = (255, 255, 100) if self.size_control_active else (255, 100, 255)
                                    cv2.circle(frame, center, radius, circle_color, 3)  # Yellow when active, magenta when inactive
                                    
                                    # Add size control indicator
                                    if self.size_control_active:
                                        cv2.putText(frame, "SIZE CONTROL", 
                                                  (center[0] - 50, center[1] + radius + 20),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 2)
                            
                            # Gesture type indicator
                            cv2.putText(frame, gesture_type.value.title(), 
                                      (index_tip[0] + 20, index_tip[1] - 20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, connection_color, 2)
            else:
                # Fallback: simple order-based matching
                for i, (hand_landmarks, gesture_data) in enumerate(zip(mp_results.multi_hand_landmarks, hand_data)):
                    landmarks, handedness, gesture_type = gesture_data.landmarks, gesture_data.handedness, gesture_data.gesture_type
                    landmark_color, connection_color = self.gesture_detector.get_drawing_specs(gesture_type)
                    
                    # Custom drawing specs based on gesture
                    landmark_drawing_spec = mp_drawing.DrawingSpec(
                        color=landmark_color, thickness=2, circle_radius=4)
                    connection_drawing_spec = mp_drawing.DrawingSpec(
                        color=connection_color, thickness=2, circle_radius=2)
                    
                    # Draw hand landmarks and connections
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec, connection_drawing_spec)
                    
                    # Draw special gesture indicators
                    if len(landmarks) > 8:
                        index_tip = landmarks[8]
                        if gesture_type.value == "erasing":
                            # Show eraser size circle (radius = thickness/2 for accurate size)
                            cv2.circle(frame, index_tip, self.current_eraser_size // 2, (0, 100, 255), 2)
                        elif gesture_type.value == "drawing":
                            # Show brush size circle (radius = thickness/2 for accurate size)
                            cv2.circle(frame, index_tip, self.current_brush_size // 2, (100, 255, 100), 2)
                            # Show snap-to-grid indicator if enabled
                            self.drawing_engine.render_snap_indicator(frame, index_tip)
                        elif gesture_type.value == "hold":
                            # Show hold circle passing through thumb, index, and middle finger tips
                            center, radius = self.gesture_detector.calculate_hold_circle(landmarks)
                            if radius > 0:  # Only draw if valid circle calculated
                                # Change color based on size control mode
                                circle_color = (255, 255, 100) if self.size_control_active else (255, 100, 255)
                                cv2.circle(frame, center, radius, circle_color, 3)  # Yellow when active, magenta when inactive
                                
                                # Add size control indicator
                                if self.size_control_active:
                                    cv2.putText(frame, "SIZE CONTROL", 
                                              (center[0] - 50, center[1] + radius + 20),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 2)
                        
                        # Gesture type indicator
                        cv2.putText(frame, gesture_type.value.title(), 
                                  (index_tip[0] + 20, index_tip[1] - 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, connection_color, 2)
    
    def _handle_keyboard_input(self) -> bool:
        """Handle keyboard input - returns False if should exit"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('s'):
            try:
                filename = f"assets/saved_drawings/Drawing_{int(time.time())}.png"
                Path("assets/saved_drawings").mkdir(parents=True, exist_ok=True)
                self.drawing_engine.save_canvas(filename)
            except Exception as e:
                print(f"Error saving drawing: {e}")
        elif key == ord('c'):
            self.drawing_engine.clear_canvas()
        
        # Phase 3: Enhanced Drawing Features
        elif key == ord('z'):  # Undo (Ctrl+Z simulation)
            self.drawing_engine.undo()
        elif key == ord('y'):  # Redo (Ctrl+Y simulation)
            self.drawing_engine.redo()
        elif key == ord('a'):  # Toggle anti-aliasing
            self.drawing_engine.toggle_anti_aliasing()
        
        # Color features
        elif key == ord('t'):  # Toggle smart color features
            self.color_manager.toggle_smart_features()
        
        # Brush controls
        elif key == ord('b'):  # Toggle between normal and watercolor mode
            self.drawing_engine.toggle_advanced_brushes()
        elif key == ord('o'):  # Increase opacity
            self.drawing_engine.adjust_brush_opacity(0.1)
        elif key == ord('l'):  # Decrease opacity
            self.drawing_engine.adjust_brush_opacity(-0.1)
        elif key == ord('k'):  # Increase hardness
            self.drawing_engine.adjust_brush_hardness(0.1)
        elif key == ord('j'):  # Decrease hardness
            self.drawing_engine.adjust_brush_hardness(-0.1)
        
        # Brush/Eraser size adjustment
        elif key == ord('=') or key == ord('+'):  # Increase brush size
            self.adjust_brush_size(2)
        elif key == ord('-'):  # Decrease brush size
            self.adjust_brush_size(-2)
        elif key == ord(']'):  # Increase eraser size
            self.adjust_eraser_size(5)
        elif key == ord('['):  # Decrease eraser size
            self.adjust_eraser_size(-5)
        
        # Phase 4.1: Enhanced Grid System Controls
        elif key == ord('g'):  # Toggle grid
            self.drawing_engine.toggle_grid()
        elif key == ord('f'):  # Toggle grid type (F for flip between normal/slanted)
            self.drawing_engine.toggle_grid_type()
        elif key == 20:  # Ctrl+T for snap-to-grid (using ASCII value)
            self.drawing_engine.toggle_snap_to_grid()
        elif key == ord('G'):  # Shift+G for grid spacing (capital G indicates Shift)
            self.drawing_engine.cycle_grid_spacing()
        elif key == ord('p'):  # Toggle pixel art mode
            self.drawing_engine.toggle_drawing_mode()
        elif key == ord('x'):  # Toggle pixel borders (only works in pixel art mode)
            if hasattr(self.drawing_engine, 'canvas_tools') and self.drawing_engine.canvas_tools.state.drawing_mode.value == "pixel":
                self.drawing_engine.toggle_pixel_borders()
        elif key == ord('?') or key == ord('/'):  # Show help
            print("\n=== CV Art Experiments - Controls ===")
            print("Drawing: G(grid) F(grid type) Shift+G(spacing) P(pixel art) X(pixel borders)")
            print("Brushes: B(normal/watercolor) O/L(opacity) K/J(hardness)")
            print("Canvas: C(clear) S(save) Z(undo) Y(redo)")
            print("Colors: T(smart colors)")
            print("Size: +/-(brush) ]/[(eraser)")
            print("=====================================\n")
        
        return True
    
    def _cleanup(self) -> None:
        """Clean up all resources"""
        # Save color preferences before closing
        if self.color_manager:
            self.color_manager.save_preferences()
            print("Color preferences saved")
        
        if self.camera_manager:
            self.camera_manager.release()
        
        if self.gesture_detector:
            self.gesture_detector.cleanup()
        
        cv2.destroyAllWindows()

def main():
    """Application entry point"""
    try:
        app = VirtualPainterApp()
        app.run()
    except Exception as e:
        print(f"Application failed to start: {e}")

if __name__ == "__main__":
    main()