"""
Performance monitoring utilities for tracking optimization improvements
"""
import time
import psutil
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    startup_time: float = 0.0
    memory_usage: float = 0.0
    import_time: float = 0.0
    config_load_time: float = 0.0
    component_init_time: float = 0.0

class PerformanceMonitor:
    """Lightweight performance monitoring for startup optimization"""

    def __init__(self):
        self.start_time = time.time()
        self.checkpoints: Dict[str, float] = {}
        self.memory_snapshots: Dict[str, float] = {}
        self._initial_memory = self._get_memory_usage()

    def checkpoint(self, name: str):
        """Record a performance checkpoint"""
        current_time = time.time()
        self.checkpoints[name] = current_time - self.start_time
        self.memory_snapshots[name] = self._get_memory_usage()

    def get_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics"""
        total_time = time.time() - self.start_time
        current_memory = self._get_memory_usage()

        return PerformanceMetrics(
            startup_time=total_time,
            memory_usage=current_memory - self._initial_memory,
            import_time=self.checkpoints.get('imports_complete', 0.0),
            config_load_time=self.checkpoints.get('config_loaded', 0.0),
            component_init_time=self.checkpoints.get('components_ready', 0.0)
        )

    def print_summary(self, verbose: bool = False):
        """Print performance summary"""
        metrics = self.get_metrics()

        print("\n=== Performance Summary ===")
        print(f"Total Startup Time: {metrics.startup_time:.3f}s")
        print(f"Memory Usage: {metrics.memory_usage:.1f}MB")

        if verbose and self.checkpoints:
            print("\nDetailed Checkpoints:")
            for name, elapsed in sorted(self.checkpoints.items(), key=lambda x: x[1]):
                memory = self.memory_snapshots.get(name, 0)
                print(f"  {name}: {elapsed:.3f}s ({memory:.1f}MB)")

        print("===========================\n")

    def get_runtime_performance(self, gesture_detector=None, camera_manager=None) -> dict:
        """Get comprehensive runtime performance metrics including optimization stats"""
        runtime_metrics = {
            'uptime_seconds': time.time() - self.start_time,
            'current_memory_mb': self._get_memory_usage(),
            'memory_delta_mb': self._get_memory_usage() - self._initial_memory,
            'checkpoints_count': len(self.checkpoints)
        }

        # Add MediaPipe optimization metrics if available
        if gesture_detector and hasattr(gesture_detector, 'get_detection_stats'):
            try:
                mp_stats = gesture_detector.get_detection_stats()
                runtime_metrics['mediapipe'] = {
                    'detection_success_rate': mp_stats.get('success_rate', 0.0),
                    'avg_processing_time_ms': mp_stats.get('average_processing_time_ms', 0.0),
                    'cache_hit_rate': mp_stats.get('cache_hit_rate', 0.0),
                    'frame_skip_rate': mp_stats.get('frame_skip_rate', 0.0),
                    'warmup_complete': mp_stats.get('warmup_complete', False),
                    'memory_pool_size': mp_stats.get('memory_pool_size', 0),
                    'total_detections': mp_stats.get('total_detections', 0)
                }
            except Exception:
                runtime_metrics['mediapipe'] = {'error': 'MediaPipe stats unavailable'}

        # Add camera optimization metrics if available
        if camera_manager and hasattr(camera_manager, 'get_camera_info'):
            try:
                cam_info = camera_manager.get_camera_info()
                runtime_metrics['camera'] = {
                    'capture_success_rate': cam_info.get('capture_success_rate', 0.0),
                    'buffer_hit_rate': cam_info.get('buffer_hit_rate', 0.0),
                    'avg_capture_time_ms': cam_info.get('average_capture_time_ms', 0.0),
                    'total_captures': cam_info.get('total_captures', 0),
                    'buffer_length': cam_info.get('buffer_length', 0),
                    'frame_ready': cam_info.get('frame_ready', False)
                }
            except Exception:
                runtime_metrics['camera'] = {'error': 'Camera stats unavailable'}

        return runtime_metrics

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0

# Global performance monitor instance
_monitor: Optional[PerformanceMonitor] = None

def start_monitoring() -> 'PerformanceMonitor':
    """Start performance monitoring"""
    global _monitor
    _monitor = PerformanceMonitor()
    return _monitor

def checkpoint(name: str):
    """Record a performance checkpoint"""
    if _monitor:
        _monitor.checkpoint(name)

def get_monitor() -> Optional[PerformanceMonitor]:
    """Get the current performance monitor"""
    return _monitor

def print_performance_summary(verbose: bool = False):
    """Print performance summary if monitoring is active"""
    if _monitor:
        _monitor.print_summary(verbose)
