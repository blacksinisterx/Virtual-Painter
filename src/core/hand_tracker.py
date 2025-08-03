import mediapipe as mp
import numpy as np
import time
from typing import List, Tuple, Optional, NamedTuple, Dict
from enum import Enum
from dataclasses import dataclass
import cv2

from config.settings import DrawingConfig
from core.exceptions import GestureDetectionException

# Import dual-stream processor for Phase 2.3 optimization
try:
    from core.dual_stream_processor import DualStreamProcessor, StreamConfig
except ImportError:
    # Fallback if dual-stream processor is not available
    DualStreamProcessor = None
    StreamConfig = None

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
    """Enhanced gesture detection with MediaPipe optimization, frame caching, and memory pooling"""

    def __init__(self, config: DrawingConfig, sensitivity: float = 1.0):
        self.config = config
        self.sensitivity = sensitivity
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        # Per-hand gesture filters with adaptive sensitivity
        self.hand_filters: Dict[str, HandGestureFilter] = {}

        # Initialize default gesture patterns
        self.gesture_patterns = self._initialize_default_patterns()

        # MediaPipe optimization and caching
        self._rgb_cache = None  # Cached RGB frame to avoid conversion
        self._cache_frame_id = -1  # Frame ID for cache validation
        self._warmup_complete = False
        self._warmup_counter = 0

        # Memory pooling for landmarks (reduce GC pressure)
        self._landmark_pool = []
        self._max_pool_size = 10

        # Frame processing optimization
        self._last_frame_hash = None
        self._last_results = None
        self._frame_skip_threshold = 0.95  # Skip if 95% similar to last frame

        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'frames_skipped': 0
        }

        # Dual-stream processor for Phase 2.3 optimization
        self.dual_stream_processor = None
        if DualStreamProcessor and hasattr(config, 'ENABLE_DUAL_STREAM') and config.ENABLE_DUAL_STREAM:
            stream_config = StreamConfig(
                display_width=getattr(config, 'DISPLAY_WIDTH', 1280),
                display_height=getattr(config, 'DISPLAY_HEIGHT', 720),
                process_width=getattr(config, 'PROCESS_WIDTH', 480),
                process_height=getattr(config, 'PROCESS_HEIGHT', 270),
                enable_gaussian_blur=getattr(config, 'ENABLE_GAUSSIAN_BLUR', False),
                gaussian_kernel_size=getattr(config, 'GAUSSIAN_KERNEL_SIZE', 3),
                enable_histogram_eq=getattr(config, 'ENABLE_HISTOGRAM_EQ', False),
                histogram_eq_threshold=getattr(config, 'HISTOGRAM_EQ_THRESHOLD', 0.3)
            )
            self.dual_stream_processor = DualStreamProcessor(stream_config)
            print("✓ Dual-stream processing enabled for optimal MediaPipe performance")
        else:
            print("ℹ Using single-stream processing (legacy mode)")

        try:
            # Optimized MediaPipe configuration
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=config.MAX_NUM_HANDS,
                min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
                model_complexity=config.MODEL_COMPLEXITY
            )

            # Warm up MediaPipe model
            self._warmup_mediapipe()

        except Exception as e:
            raise GestureDetectionException(f"MediaPipe initialization failed: {e}")

    def _warmup_mediapipe(self):
        """Warm up MediaPipe model with dummy frames for optimal performance"""
        try:
            # Create dummy frame for warmup
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            for _ in range(self.config.WARMUP_FRAMES):
                self.hands.process(dummy_frame)

            self._warmup_complete = True

        except Exception:
            # If warmup fails, continue without it
            self._warmup_complete = True

    def _get_pooled_landmarks(self) -> List:
        """Get landmarks list from pool or create new one"""
        if self._landmark_pool:
            return self._landmark_pool.pop()
        return []

    def _return_to_pool(self, landmarks: List):
        """Return landmarks list to pool for reuse"""
        if len(self._landmark_pool) < self._max_pool_size:
            landmarks.clear()
            self._landmark_pool.append(landmarks)

    def _calculate_frame_similarity(self, frame: np.ndarray) -> float:
        """Calculate similarity to last frame to enable intelligent skipping"""
        if self._last_frame_hash is None:
            return 0.0

        # Simple hash-based similarity (fast)
        current_hash = hash(frame.tobytes()[::1000])  # Sample every 1000th byte
        similarity = 1.0 if current_hash == self._last_frame_hash else 0.0
        self._last_frame_hash = current_hash
        return similarity

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
        """Get comprehensive gesture detection performance statistics"""
        success_rate = 0.0
        cache_hit_rate = 0.0
        skip_rate = 0.0

        if self.detection_stats['total_detections'] > 0:
            success_rate = self.detection_stats['successful_detections'] / self.detection_stats['total_detections']
            cache_hit_rate = self.detection_stats['cache_hits'] / self.detection_stats['total_detections']
            skip_rate = self.detection_stats['frames_skipped'] / self.detection_stats['total_detections']

        return {
            'success_rate': success_rate,
            'total_detections': self.detection_stats['total_detections'],
            'average_processing_time_ms': self.detection_stats['average_processing_time'] * 1000,
            'cache_hit_rate': cache_hit_rate,
            'frame_skip_rate': skip_rate,
            'warmup_complete': self._warmup_complete,
            'memory_pool_size': len(self._landmark_pool)
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
        """Optimized frame processing with dual-stream architecture, caching, and memory pooling"""
        try:
            start_time = time.time()

            # Phase 2.3: Dual-stream processing for optimal MediaPipe performance
            if self.dual_stream_processor:
                display_frame, process_frame, preprocessing_info = self.dual_stream_processor.process_frame_dual_stream(frame)

                # Use optimized frame for MediaPipe processing
                detection_frame = process_frame

                # Track preprocessing statistics
                self.detection_stats['preprocessing_applied'] = preprocessing_info
            else:
                # Legacy single-stream processing
                detection_frame = frame
                display_frame = frame

            # Frame similarity check for intelligent processing
            frame_similarity = self._calculate_frame_similarity(detection_frame)
            if frame_similarity > self._frame_skip_threshold and self._last_results is not None:
                # Use cached results for very similar frames
                self.detection_stats['frames_skipped'] += 1
                self.detection_stats['cache_hits'] += 1
                return self._last_results

            # Optimized RGB conversion with caching
            frame_id = id(detection_frame)
            if self._cache_frame_id != frame_id or self._rgb_cache is None:
                self._rgb_cache = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                self._cache_frame_id = frame_id
            else:
                self.detection_stats['cache_hits'] += 1

            # MediaPipe processing on optimized frame
            results = self.hands.process(self._rgb_cache)

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

                    # Extract landmark coordinates using pooled memory
                    landmarks = self._get_pooled_landmarks()

                    # Handle coordinate scaling for dual-stream processing
                    if self.dual_stream_processor:
                        # Use detection frame dimensions for initial extraction
                        h, w, _ = detection_frame.shape
                    else:
                        # Legacy single-stream processing
                        h, w, _ = frame.shape

                    # Efficient landmark extraction
                    raw_landmarks = [
                        (int(landmark.x * w), int(landmark.y * h))
                        for landmark in hand_landmarks.landmark
                    ]

                    # Scale coordinates back to display resolution if using dual-stream
                    if self.dual_stream_processor:
                        scale_x = self.dual_stream_processor.scale_x
                        scale_y = self.dual_stream_processor.scale_y

                        for x, y in raw_landmarks:
                            scaled_x = int(x * scale_x)
                            scaled_y = int(y * scale_y)
                            landmarks.append((scaled_x, scaled_y))
                    else:
                        landmarks.extend(raw_landmarks)

                    # Detect fingers and classify gesture with confidence
                    fingers = self._fingers_up(hand_landmarks, handedness)
                    raw_gesture = self._classify_gesture(fingers, hand_confidence)

                    # Apply per-hand filtering with confidence-based timing
                    filtered_gesture = self.hand_filters[handedness].filter_gesture(
                        raw_gesture, hand_confidence
                    )

                    hand_landmarks_data = HandLandmarks(landmarks.copy(), handedness, filtered_gesture)
                    hand_data.append(hand_landmarks_data)

                    # Return landmarks to pool
                    self._return_to_pool(landmarks)

                    self.detection_stats['successful_detections'] += 1

            # Cache results for potential reuse
            self._last_results = (hand_data, results)

            # Update performance statistics
            processing_time = time.time() - start_time
            self.detection_stats['average_processing_time'] = (
                self.detection_stats['average_processing_time'] * 0.9 + processing_time * 0.1
            )

            # Add dual-stream performance metrics
            if self.dual_stream_processor:
                dual_stream_stats = self.dual_stream_processor.get_preprocessing_stats()
                self.detection_stats['dual_stream_stats'] = {
                    'total_frames': dual_stream_stats.total_frames_processed,
                    'avg_preprocessing_time': dual_stream_stats.average_processing_time_ms,
                    'gaussian_blur_rate': (dual_stream_stats.gaussian_blur_applied / max(1, dual_stream_stats.total_frames_processed)) * 100,
                    'histogram_eq_rate': (dual_stream_stats.histogram_eq_applied / max(1, dual_stream_stats.total_frames_processed)) * 100
                }

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

    def get_dual_stream_performance_summary(self) -> dict:
        """Get comprehensive dual-stream performance summary"""
        if not self.dual_stream_processor:
            return {"dual_stream_enabled": False}

        return {
            "dual_stream_enabled": True,
            "performance_summary": self.dual_stream_processor.get_performance_summary(),
            "preprocessing_stats": self.dual_stream_processor.get_preprocessing_stats().__dict__
        }

    def update_dual_stream_config(self, **kwargs):
        """Update dual-stream configuration dynamically"""
        if self.dual_stream_processor:
            self.dual_stream_processor.update_config(**kwargs)
            print("Dual-stream configuration updated")
        else:
            print("Dual-stream processing not enabled")

    def cleanup(self) -> None:
        """Clean up MediaPipe resources and optimization caches"""
        if hasattr(self, 'hands'):
            self.hands.close()

        # Clear optimization caches
        self._rgb_cache = None
        self._last_results = None
        self._last_frame_hash = None

        # Clear memory pools
        self._landmark_pool.clear()

        # Clear hand filters
        self.hand_filters.clear()
