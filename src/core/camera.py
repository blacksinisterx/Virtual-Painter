from core.lazy_imports import cv2
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
import time

# Import cv2 properly for type annotations
if TYPE_CHECKING:
    import cv2 as cv2_types

from config.settings import CameraConfig
from core.exceptions import CameraException

class CameraManager:
    """Optimized camera manager with advanced buffering and frame processing pipeline"""

    # Optimal resolution presets based on common laptop cameras
    OPTIMAL_PRESETS = [
        (1280, 720),   # Most common laptop camera resolution
        (1920, 1080),  # HD webcams
        (800, 600),    # Older laptops
        (640, 480)     # Fallback
    ]

    def __init__(self, config: CameraConfig, preload_optimal: bool = True):
        self.config = config
        self.cap = None  # cv2.VideoCapture object
        self.current_resolution: Optional[Tuple[int, int]] = None
        self.current_camera_index: int = 0  # Always use main laptop camera
        self.preload_optimal = preload_optimal

        # Frame processing optimization
        self._frame_buffer = []
        self._max_buffer_size = 2  # Small buffer for latest frames
        self._last_frame = None
        self._frame_ready = False

        # Performance tracking
        self._capture_stats = {
            'total_captures': 0,
            'failed_captures': 0,
            'average_capture_time': 0.0,
            'buffer_hits': 0
        }

        self._initialize_camera()

    def _initialize_camera(self) -> None:
        """Initialize main laptop camera with optimized settings and buffer management"""
        try:
            self.cap = cv2.VideoCapture(self.current_camera_index)
            if not self.cap.isOpened():
                raise CameraException(f"Main laptop camera (index 0) could not be opened")

            # Optimize camera settings for hand tracking
            self._optimize_camera_settings()
            self._set_best_resolution()

            # Pre-warm the camera with initial frames
            self._warm_up_camera()

        except Exception as e:
            raise CameraException(f"Failed to initialize main laptop camera: {e}")

    def _warm_up_camera(self):
        """Warm up camera and populate initial frame buffer"""
        try:
            # Capture a few frames to stabilize camera
            for _ in range(3):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self._last_frame = cv2.flip(frame, 1)  # Pre-flip for efficiency
                    self._frame_ready = True
        except Exception:
            pass  # Continue even if warmup fails

    def _optimize_camera_settings(self) -> None:
        """Optimize camera settings for better hand tracking performance"""
        if not self.cap:
            return

        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Set frame rate for smooth tracking (30 FPS ideal for hand tracking)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Auto-focus and auto-exposure for consistent image quality
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Slight auto-exposure

    def set_camera_property(self, property_id: int, value: float) -> bool:
        """Set a camera property (exposure, brightness, etc.)"""
        if not self.cap or not self.cap.isOpened():
            return False

        try:
            return self.cap.set(property_id, value)
        except Exception:
            return False

    def get_camera_property(self, property_id: int) -> Optional[float]:
        """Get a camera property value"""
        if not self.cap or not self.cap.isOpened():
            return None

        try:
            return self.cap.get(property_id)
        except Exception:
            return None

    def _set_best_resolution(self) -> None:
        """Set the highest available resolution with smart preallocation"""
        if self.preload_optimal:
            # Try optimal presets first for faster initialization
            for width, height in self.OPTIMAL_PRESETS:
                if self._try_resolution(width, height):
                    return

            # If presets fail, fall back to config resolutions
            for width, height in self.config.RESOLUTIONS:
                if (width, height) not in self.OPTIMAL_PRESETS:  # Skip already tried
                    if self._try_resolution(width, height):
                        return
        else:
            # Original behavior - try config resolutions in order
            for width, height in self.config.RESOLUTIONS:
                if self._try_resolution(width, height):
                    return

        # Final fallback: use actual camera resolution
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.current_resolution = (frame.shape[1], frame.shape[0])
        else:
            raise CameraException("Unable to capture initial frame for resolution detection")

    def _try_resolution(self, width: int, height: int) -> bool:
        """Try to set a specific resolution - returns True if successful"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Test resolution by capturing a frame
        ret, frame = self.cap.read()
        if ret and frame is not None:
            actual_width, actual_height = frame.shape[1], frame.shape[0]
            # Allow small tolerance for resolution mismatch
            if abs(actual_width - width) <= 10 and abs(actual_height - height) <= 10:
                self.current_resolution = (actual_width, actual_height)
                return True
        return False

    def change_resolution(self, width: int, height: int) -> bool:
        """Change camera resolution during runtime"""
        if not self.cap or not self.cap.isOpened():
            return False

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Verify resolution change
            ret, frame = self.cap.read()
            if ret and frame is not None:
                actual_width, actual_height = frame.shape[1], frame.shape[0]
                if abs(actual_width - width) <= 10 and abs(actual_height - height) <= 10:
                    self.current_resolution = (actual_width, actual_height)
                    return True
            return False

        except Exception:
            return False

    def capture_frame(self) -> np.ndarray:
        """Optimized frame capture with buffering, error recovery, and performance tracking"""
        if not self.cap or not self.cap.isOpened():
            raise CameraException("Camera not initialized")

        start_time = time.time()

        try:
            # Try to capture frame with optimized retry logic
            ret, frame = self.cap.read()

            if ret and frame is not None:
                # Efficient frame processing
                flipped_frame = cv2.flip(frame, 1)

                # Update frame buffer
                self._update_frame_buffer(flipped_frame)

                # Update statistics
                self._capture_stats['total_captures'] += 1
                capture_time = time.time() - start_time
                self._capture_stats['average_capture_time'] = (
                    self._capture_stats['average_capture_time'] * 0.9 + capture_time * 0.1
                )

                return flipped_frame
            else:
                # Failed capture - try to use last good frame if available
                if self._last_frame is not None:
                    self._capture_stats['buffer_hits'] += 1
                    return self._last_frame.copy()

                # Multiple retry attempts with increasing delays
                for attempt in range(2):
                    time.sleep(0.01 * (attempt + 1))  # Progressive delay
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        flipped_frame = cv2.flip(frame, 1)
                        self._update_frame_buffer(flipped_frame)
                        return flipped_frame

                self._capture_stats['failed_captures'] += 1
                raise CameraException("Failed to capture frame after multiple attempts")

        except Exception as e:
            self._capture_stats['failed_captures'] += 1
            # Last resort: return cached frame if available
            if self._last_frame is not None:
                self._capture_stats['buffer_hits'] += 1
                return self._last_frame.copy()
            raise CameraException(f"Frame capture failed: {e}")

    def _update_frame_buffer(self, frame: np.ndarray):
        """Update internal frame buffer for optimization"""
        self._last_frame = frame.copy()

        # Maintain small buffer of recent frames
        if len(self._frame_buffer) >= self._max_buffer_size:
            self._frame_buffer.pop(0)
        self._frame_buffer.append(frame)
        self._frame_ready = True

    def get_resolution(self) -> Tuple[int, int]:
        """Get current camera resolution"""
        if not self.current_resolution:
            raise CameraException("Camera resolution not available")
        return self.current_resolution

    def get_camera_info(self) -> dict:
        """Get comprehensive camera information and performance statistics"""
        if not self.cap or not self.cap.isOpened():
            return {}

        # Calculate performance metrics
        success_rate = 0.0
        buffer_hit_rate = 0.0
        if self._capture_stats['total_captures'] > 0:
            success_rate = 1.0 - (self._capture_stats['failed_captures'] / self._capture_stats['total_captures'])
            buffer_hit_rate = self._capture_stats['buffer_hits'] / self._capture_stats['total_captures']

        return {
            'camera_index': 0,  # Always main laptop camera
            'resolution': self.current_resolution,
            'fps': self.get_camera_property(cv2.CAP_PROP_FPS),
            'brightness': self.get_camera_property(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.get_camera_property(cv2.CAP_PROP_CONTRAST),
            'saturation': self.get_camera_property(cv2.CAP_PROP_SATURATION),
            'auto_exposure': self.get_camera_property(cv2.CAP_PROP_AUTO_EXPOSURE),
            'buffer_size': self.get_camera_property(cv2.CAP_PROP_BUFFERSIZE),

            # Performance statistics
            'capture_success_rate': success_rate,
            'buffer_hit_rate': buffer_hit_rate,
            'average_capture_time_ms': self._capture_stats['average_capture_time'] * 1000,
            'total_captures': self._capture_stats['total_captures'],
            'frame_ready': self._frame_ready,
            'buffer_length': len(self._frame_buffer)
        }

    def is_camera_ready(self) -> bool:
        """Check if camera is properly initialized and ready"""
        return (self.cap is not None and
                self.cap.isOpened() and
                self.current_resolution is not None)

    def release(self) -> None:
        """Release camera resources and cleanup optimization buffers"""
        if self.cap:
            self.cap.release()
            self.cap = None
            self.current_resolution = None

        # Clear optimization buffers
        self._frame_buffer.clear()
        self._last_frame = None
        self._frame_ready = False
