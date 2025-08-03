"""
Dual-Stream MediaPipe Processing Architecture
Optimizes MediaPipe performance while maintaining display quality
"""

import cv2
import numpy as np
import time
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class StreamConfig:
    """Configuration for dual-stream processing"""
    # Display stream (high quality for UI)
    display_width: int = 1280
    display_height: int = 720

    # Processing stream (optimized for MediaPipe)
    process_width: int = 480
    process_height: int = 270

    # Preprocessing options
    enable_gaussian_blur: bool = False
    gaussian_kernel_size: int = 3
    enable_histogram_eq: bool = False
    histogram_eq_threshold: float = 0.3  # Apply if contrast below this

    # Performance tracking
    enable_preprocessing_stats: bool = True


@dataclass
class PreprocessingStats:
    """Statistics for preprocessing operations"""
    gaussian_blur_applied: int = 0
    histogram_eq_applied: int = 0
    total_frames_processed: int = 0
    average_processing_time_ms: float = 0.0
    scale_factor_x: float = 1.0
    scale_factor_y: float = 1.0


class LightingAnalyzer:
    """Analyzes frame lighting conditions to determine preprocessing needs"""

    def __init__(self, contrast_threshold: float = 0.3):
        self.contrast_threshold = contrast_threshold
        self.history_size = 10
        self.contrast_history = []

    def analyze_lighting(self, frame: np.ndarray) -> Dict[str, float]:
        """Analyze lighting conditions in the frame"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate contrast using standard deviation
        contrast = np.std(gray) / 255.0

        # Calculate brightness
        brightness = np.mean(gray) / 255.0

        # Update history
        self.contrast_history.append(contrast)
        if len(self.contrast_history) > self.history_size:
            self.contrast_history.pop(0)

        # Calculate average contrast over recent frames
        avg_contrast = np.mean(self.contrast_history)

        return {
            'contrast': contrast,
            'brightness': brightness,
            'avg_contrast': avg_contrast,
            'needs_histogram_eq': avg_contrast < self.contrast_threshold
        }


class NoiseDetector:
    """Detects noise levels in frames to determine blur necessity"""

    def __init__(self, noise_threshold: float = 15.0):
        self.noise_threshold = noise_threshold

    def detect_noise_level(self, frame: np.ndarray) -> Dict[str, float]:
        """Detect noise level in the frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use Laplacian variance to detect noise/sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Simple noise estimation based on high-frequency content
        noise_level = laplacian_var

        return {
            'noise_level': noise_level,
            'needs_blur': noise_level > self.noise_threshold
        }


class DualStreamProcessor:
    """
    Dual-stream processor for optimal MediaPipe performance
    Maintains high-quality display while optimizing detection input
    """

    def __init__(self, config: StreamConfig):
        self.config = config

        # Calculate scale factors for coordinate mapping
        self.scale_x = config.display_width / config.process_width
        self.scale_y = config.display_height / config.process_height

        # Initialize analyzers
        self.lighting_analyzer = LightingAnalyzer(config.histogram_eq_threshold)
        self.noise_detector = NoiseDetector()

        # Statistics tracking
        self.stats = PreprocessingStats(
            scale_factor_x=self.scale_x,
            scale_factor_y=self.scale_y
        )

        # Performance monitoring
        self.processing_times = []
        self.max_time_history = 100

        print(f"DualStreamProcessor initialized:")
        print(f"  Display: {config.display_width}×{config.display_height}")
        print(f"  Process: {config.process_width}×{config.process_height}")
        print(f"  Scale factors: {self.scale_x:.2f}×, {self.scale_y:.2f}×")

    def process_frame_dual_stream(self, raw_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Process frame using dual-stream architecture

        Returns:
            display_frame: High-quality frame for UI display
            process_frame: Optimized frame for MediaPipe detection
            preprocessing_info: Information about applied preprocessing
        """
        start_time = time.time()

        # Stream 1: Display frame (pristine quality)
        display_frame = raw_frame.copy()

        # Stream 2: Processing frame (optimized for MediaPipe)
        process_frame, preprocessing_info = self._prepare_detection_frame(raw_frame)

        # Update statistics
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        self._update_performance_stats(processing_time, preprocessing_info)

        return display_frame, process_frame, preprocessing_info

    def _prepare_detection_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Prepare optimized frame for MediaPipe detection"""
        preprocessing_info = {
            'gaussian_blur_applied': False,
            'histogram_eq_applied': False,
            'original_resolution': frame.shape[:2],
            'target_resolution': (self.config.process_height, self.config.process_width)
        }

        # Step 1: Smart downscaling (always applied for optimal MediaPipe performance)
        detection_frame = cv2.resize(
            frame,
            (self.config.process_width, self.config.process_height),
            interpolation=cv2.INTER_LINEAR
        )

        # Step 2: Optional preprocessing based on frame analysis
        if self.config.enable_gaussian_blur or self.config.enable_histogram_eq:
            # Analyze frame conditions
            lighting_analysis = self.lighting_analyzer.analyze_lighting(detection_frame)
            noise_analysis = self.noise_detector.detect_noise_level(detection_frame)

            # Apply Gaussian blur if noise detected or explicitly enabled
            if self.config.enable_gaussian_blur or noise_analysis['needs_blur']:
                detection_frame = cv2.GaussianBlur(
                    detection_frame,
                    (self.config.gaussian_kernel_size, self.config.gaussian_kernel_size),
                    0
                )
                preprocessing_info['gaussian_blur_applied'] = True
                self.stats.gaussian_blur_applied += 1

            # Apply histogram equalization if poor lighting or explicitly enabled
            if self.config.enable_histogram_eq or lighting_analysis['needs_histogram_eq']:
                detection_frame = self._apply_adaptive_histogram_eq(detection_frame)
                preprocessing_info['histogram_eq_applied'] = True
                preprocessing_info['lighting_analysis'] = lighting_analysis
                self.stats.histogram_eq_applied += 1

        return detection_frame, preprocessing_info

    def _apply_adaptive_histogram_eq(self, frame: np.ndarray) -> np.ndarray:
        """Apply adaptive histogram equalization to improve contrast"""
        # Convert to YUV color space
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to Y channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])

        # Convert back to BGR
        enhanced_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        return enhanced_frame

    def _update_performance_stats(self, processing_time_ms: float, preprocessing_info: Dict[str, Any]):
        """Update performance statistics"""
        self.stats.total_frames_processed += 1

        # Update processing time average
        self.processing_times.append(processing_time_ms)
        if len(self.processing_times) > self.max_time_history:
            self.processing_times.pop(0)

        self.stats.average_processing_time_ms = np.mean(self.processing_times)

    def get_preprocessing_stats(self) -> PreprocessingStats:
        """Get current preprocessing statistics"""
        return self.stats

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        total_frames = self.stats.total_frames_processed
        if total_frames == 0:
            return {"error": "No frames processed yet"}

        return {
            "total_frames": total_frames,
            "average_processing_time_ms": self.stats.average_processing_time_ms,
            "gaussian_blur_rate": (self.stats.gaussian_blur_applied / total_frames) * 100,
            "histogram_eq_rate": (self.stats.histogram_eq_applied / total_frames) * 100,
            "scale_factors": {
                "x": self.stats.scale_factor_x,
                "y": self.stats.scale_factor_y
            },
            "resolution_info": {
                "display": f"{self.config.display_width}×{self.config.display_height}",
                "process": f"{self.config.process_width}×{self.config.process_height}",
                "reduction_factor": f"{(self.config.display_width * self.config.display_height) / (self.config.process_width * self.config.process_height):.1f}×"
            }
        }

    def update_config(self, **kwargs):
        """Update configuration parameters dynamically"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                print(f"Updated {key} to {value}")

        # Recalculate scale factors if resolution changed
        if 'display_width' in kwargs or 'display_height' in kwargs or \
           'process_width' in kwargs or 'process_height' in kwargs:
            self.scale_x = self.config.display_width / self.config.process_width
            self.scale_y = self.config.display_height / self.config.process_height
            self.stats.scale_factor_x = self.scale_x
            self.stats.scale_factor_y = self.scale_y
            print(f"Updated scale factors: {self.scale_x:.2f}×, {self.scale_y:.2f}×")
