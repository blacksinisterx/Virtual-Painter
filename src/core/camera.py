import cv2
from typing import Optional, Tuple, List
import numpy as np

from config.settings import CameraConfig
from core.exceptions import CameraException

class CameraManager:
    """Enhanced camera manager with multi-camera support and advanced features"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_resolution: Optional[Tuple[int, int]] = None
        self.available_cameras: List[int] = []
        self.current_camera_index: int = config.DEFAULT_CAMERA_INDEX
        self._discover_cameras()
        self._initialize_camera()
    
    def _discover_cameras(self) -> None:
        """Discover all available cameras in the system"""
        self.available_cameras = []
        
        # Test camera indices 0-10 for availability
        for i in range(11):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.available_cameras.append(i)
            cap.release()
        
        if not self.available_cameras:
            raise CameraException("No cameras detected on the system")
    
    def get_available_cameras(self) -> List[int]:
        """Get list of available camera indices"""
        return self.available_cameras.copy()
    
    def switch_camera(self, camera_index: int) -> bool:
        """Switch to a different camera"""
        if camera_index not in self.available_cameras:
            return False
        
        try:
            # Release current camera
            if self.cap:
                self.cap.release()
            
            # Initialize new camera
            self.current_camera_index = camera_index
            self._initialize_camera()
            return True
            
        except Exception:
            # Try to restore previous camera if switch fails
            try:
                self.current_camera_index = self.config.DEFAULT_CAMERA_INDEX
                self._initialize_camera()
            except Exception:
                pass
            return False
    
    def _initialize_camera(self) -> None:
        """Initialize camera with best available resolution and optimized settings"""
        try:
            self.cap = cv2.VideoCapture(self.current_camera_index)
            if not self.cap.isOpened():
                raise CameraException(f"Camera {self.current_camera_index} could not be opened")
            
            # Optimize camera settings for hand tracking
            self._optimize_camera_settings()
            self._set_best_resolution()
            
        except Exception as e:
            raise CameraException(f"Failed to initialize camera {self.current_camera_index}: {e}")
    
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
        """Set the highest available resolution from config with validation"""
        for width, height in self.config.RESOLUTIONS:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Test resolution by capturing a frame
            ret, frame = self.cap.read()
            if ret and frame is not None:
                actual_width, actual_height = frame.shape[1], frame.shape[0]
                # Allow small tolerance for resolution mismatch
                if abs(actual_width - width) <= 10 and abs(actual_height - height) <= 10:
                    self.current_resolution = (actual_width, actual_height)
                    return
        
        # Fallback: use actual camera resolution
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.current_resolution = (frame.shape[1], frame.shape[0])
        else:
            raise CameraException("Unable to capture initial frame for resolution detection")
    
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
        """Capture and return optimized frame with error recovery"""
        if not self.cap or not self.cap.isOpened():
            raise CameraException("Camera not initialized")
        
        # Try to capture frame with retry logic
        for attempt in range(3):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Flip frame for mirror effect (standard for drawing apps)
                return cv2.flip(frame, 1)
            
            # Brief wait before retry
            if attempt < 2:
                import time
                time.sleep(0.01)
        
        raise CameraException("Failed to capture frame after multiple attempts")
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get current camera resolution"""
        if not self.current_resolution:
            raise CameraException("Camera resolution not available")
        return self.current_resolution
    
    def get_camera_info(self) -> dict:
        """Get comprehensive camera information"""
        if not self.cap or not self.cap.isOpened():
            return {}
        
        return {
            'camera_index': self.current_camera_index,
            'resolution': self.current_resolution,
            'fps': self.get_camera_property(cv2.CAP_PROP_FPS),
            'brightness': self.get_camera_property(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.get_camera_property(cv2.CAP_PROP_CONTRAST),
            'saturation': self.get_camera_property(cv2.CAP_PROP_SATURATION),
            'auto_exposure': self.get_camera_property(cv2.CAP_PROP_AUTO_EXPOSURE),
            'buffer_size': self.get_camera_property(cv2.CAP_PROP_BUFFERSIZE)
        }
    
    def is_camera_ready(self) -> bool:
        """Check if camera is properly initialized and ready"""
        return (self.cap is not None and 
                self.cap.isOpened() and 
                self.current_resolution is not None)
    
    def release(self) -> None:
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
            self.current_resolution = None