import mediapipe as mp
import numpy as np
import time
from typing import List, Tuple, Optional, NamedTuple, Dict
from enum import Enum
from dataclasses import dataclass
import cv2

from config.settings import DrawingConfig
from core.exceptions import GestureDetectionException

class GestureType(Enum):
    """Enumeration of supported gesture types"""
    DRAWING = "drawing"
    ERASING = "erasing"
    COLOR_SELECTION = "color_selection"
    HOLD = "hold"
    NEUTRAL = "neutral"
    PEN_LIFTED = "pen_lifted"
    UNKNOWN = "unknown"

@dataclass
class GesturePattern:
    """Configurable gesture pattern definition"""
    gesture_type: GestureType
    finger_pattern: Tuple[bool, bool, bool, bool, bool]  # Thumb, Index, Middle, Ring, Pinky
    description: str
    confidence_threshold: float = 0.8

class HandLandmarks(NamedTuple):
    """Structured hand landmark data"""
    landmarks: List[Tuple[int, int]]
    handedness: str
    gesture_type: GestureType

class HandGestureFilter:
    """Enhanced per-hand gesture filtering with adaptive timing"""
    def __init__(self, min_gesture_duration: float = 0.05, sensitivity: float = 1.0):
        self.last_gesture = GestureType.NEUTRAL
        self.gesture_start_time = 0
        self.min_gesture_duration = max(0.03, min_gesture_duration / sensitivity)  # Adaptive timing
        self.gesture_confidence = 0.0
        
    def filter_gesture(self, raw_gesture: GestureType, confidence: float = 1.0) -> GestureType:
        """Enhanced gesture filtering with confidence-based timing"""
        current_time = time.time()
        
        if raw_gesture != self.last_gesture:
            # Faster switching for high-confidence gestures
            min_duration = self.min_gesture_duration * (2.0 - confidence)
            
            if current_time - self.gesture_start_time > min_duration:
                self.last_gesture = raw_gesture
                self.gesture_start_time = current_time
                self.gesture_confidence = confidence
                return raw_gesture
            else:
                return self.last_gesture
        else:
            # Update confidence for current gesture
            self.gesture_confidence = max(self.gesture_confidence, confidence)
            return raw_gesture

class GestureDetector:
    """Enhanced gesture detection with configurable patterns and improved accuracy"""
    
    def __init__(self, config: DrawingConfig, sensitivity: float = 1.0):
        self.config = config
        self.sensitivity = sensitivity
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        # Per-hand gesture filters with adaptive sensitivity
        self.hand_filters: Dict[str, HandGestureFilter] = {}
        
        # Initialize default gesture patterns
        self.gesture_patterns = self._initialize_default_patterns()
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'average_processing_time': 0.0
        }
        
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=config.MAX_NUM_HANDS,
                min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=max(0.5, config.MIN_DETECTION_CONFIDENCE - 0.1)
            )
        except Exception as e:
            raise GestureDetectionException(f"MediaPipe initialization failed: {e}")
    
    def _initialize_default_patterns(self) -> List[GesturePattern]:
        """Initialize the default set of gesture patterns"""
        return [
            GesturePattern(GestureType.DRAWING, (False, True, False, False, False), "Index finger only"),
            GesturePattern(GestureType.COLOR_SELECTION, (True, True, False, False, False), "Thumb + Index"),
            GesturePattern(GestureType.HOLD, (True, True, True, False, False), "Thumb + Index + Middle"),
            GesturePattern(GestureType.ERASING, (True, True, True, True, True), "All fingers"),
            GesturePattern(GestureType.PEN_LIFTED, (False, True, True, False, False), "Index + Middle"),
        ]
    
    def add_custom_gesture(self, pattern: GesturePattern) -> bool:
        """Add a custom gesture pattern"""
        try:
            # Check for conflicts with existing patterns
            for existing_pattern in self.gesture_patterns:
                if existing_pattern.finger_pattern == pattern.finger_pattern:
                    # Update existing pattern
                    existing_pattern.gesture_type = pattern.gesture_type
                    existing_pattern.description = pattern.description
                    existing_pattern.confidence_threshold = pattern.confidence_threshold
                    return True
            
            # Add new pattern
            self.gesture_patterns.append(pattern)
            return True
            
        except Exception:
            return False
    
    def get_detection_stats(self) -> dict:
        """Get gesture detection performance statistics"""
        success_rate = 0.0
        if self.detection_stats['total_detections'] > 0:
            success_rate = self.detection_stats['successful_detections'] / self.detection_stats['total_detections']
        
        return {
            'success_rate': success_rate,
            'total_detections': self.detection_stats['total_detections'],
            'average_processing_time_ms': self.detection_stats['average_processing_time'] * 1000
        }
    
    def _fingers_up(self, hand_landmarks, handedness: str) -> List[bool]:
        """Detect which fingers are extended"""
        try:
            tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
            fingers = []
            
            for i in tips:
                if i == 4:  # Thumb
                    if handedness == 'Right':
                        fingers.append(hand_landmarks.landmark[i].x < hand_landmarks.landmark[i - 1].x)
                    else:  # Left hand
                        fingers.append(hand_landmarks.landmark[i].x > hand_landmarks.landmark[i - 1].x)
                else:
                    fingers.append(hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y)
            
            return fingers
        except Exception as e:
            return [False] * 5
    
    def calculate_hold_circle(self, landmarks: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], int]:
        """Calculate circle center and radius from thumb, index, and middle finger tips"""
        if len(landmarks) < 13:  # Need at least up to middle finger tip (index 12)
            return ((0, 0), 0)
        
        # Get the three finger tips: thumb (4), index (8), middle (12)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8] 
        middle_tip = landmarks[12]
        
        # Calculate centroid (average position) of the three points
        center_x = (thumb_tip[0] + index_tip[0] + middle_tip[0]) // 3
        center_y = (thumb_tip[1] + index_tip[1] + middle_tip[1]) // 3
        center = (center_x, center_y)
        
        # Calculate average distance from center to the three points
        dist1 = np.sqrt((thumb_tip[0] - center_x)**2 + (thumb_tip[1] - center_y)**2)
        dist2 = np.sqrt((index_tip[0] - center_x)**2 + (index_tip[1] - center_y)**2)
        dist3 = np.sqrt((middle_tip[0] - center_x)**2 + (middle_tip[1] - center_y)**2)
        
        radius = int((dist1 + dist2 + dist3) / 3)
        
        return center, radius
    
    def _classify_gesture(self, fingers: List[bool], confidence: float = 1.0) -> GestureType:
        """Classify gesture based on configurable patterns with confidence thresholding"""
        # Convert to tuple for exact matching
        finger_pattern = tuple(fingers)
        
        # Check against all configured gesture patterns
        for pattern in self.gesture_patterns:
            if finger_pattern == pattern.finger_pattern:
                # Apply confidence threshold
                if confidence >= pattern.confidence_threshold:
                    return pattern.gesture_type
                else:
                    # Low confidence - default to neutral
                    return GestureType.NEUTRAL
        
        # No pattern matched - return neutral
        return GestureType.NEUTRAL
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[HandLandmarks], object]:
        """Process frame and return hand landmarks with gesture classification and raw results"""
        try:
            start_time = time.time()
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            hand_data = []
            self.detection_stats['total_detections'] += 1
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, hand_type in zip(results.multi_hand_landmarks, results.multi_handedness):
                    handedness = hand_type.classification[0].label
                    hand_confidence = hand_type.classification[0].score
                    
                    # Create filter for this hand if it doesn't exist  
                    if handedness not in self.hand_filters:
                        self.hand_filters[handedness] = HandGestureFilter(
                            min_gesture_duration=self.config.MIN_GESTURE_DURATION,
                            sensitivity=self.sensitivity
                        )
                    
                    # Extract landmark coordinates
                    landmarks = []
                    h, w, _ = frame.shape
                    for landmark in hand_landmarks.landmark:
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        landmarks.append((cx, cy))
                    
                    # Detect fingers and classify gesture with confidence
                    fingers = self._fingers_up(hand_landmarks, handedness)
                    raw_gesture = self._classify_gesture(fingers, hand_confidence)
                    
                    # Apply per-hand filtering with confidence-based timing
                    filtered_gesture = self.hand_filters[handedness].filter_gesture(
                        raw_gesture, hand_confidence
                    )
                    
                    hand_data.append(HandLandmarks(landmarks, handedness, filtered_gesture))
                    self.detection_stats['successful_detections'] += 1
            
            # Update performance statistics
            processing_time = time.time() - start_time
            self.detection_stats['average_processing_time'] = (
                self.detection_stats['average_processing_time'] * 0.9 + processing_time * 0.1
            )
            
            return hand_data, results
            
        except Exception as e:
            raise GestureDetectionException(f"Frame processing failed: {e}")
    
    def get_drawing_specs(self, gesture_type: GestureType) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Get drawing specifications based on gesture type"""
        color_map = {
            GestureType.ERASING: ((0, 100, 255), (0, 150, 255)),
            GestureType.COLOR_SELECTION: ((100, 255, 255), (150, 255, 255)),
            GestureType.DRAWING: ((100, 255, 100), (150, 255, 150)),
            GestureType.HOLD: ((255, 100, 255), (255, 150, 255)),  # Magenta/purple for hold
            GestureType.PEN_LIFTED: ((100, 255, 100), (150, 255, 150)),
            GestureType.NEUTRAL: ((255, 255, 255), (200, 200, 200)),
            GestureType.UNKNOWN: ((128, 128, 128), (160, 160, 160))
        }
        return color_map.get(gesture_type, color_map[GestureType.UNKNOWN])
    
    def cleanup(self) -> None:
        """Clean up MediaPipe resources"""
        if hasattr(self, 'hands'):
            self.hands.close()