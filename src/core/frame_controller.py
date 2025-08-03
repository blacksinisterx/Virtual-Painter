"""
Frame rate controller and pipeline optimization for smooth performance
"""
import time
import threading
from queue import Queue, Empty
from typing import Optional, Callable, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class FrameStats:
    """Frame rate and timing statistics"""
    target_fps: float = 30.0
    actual_fps: float = 0.0
    frame_time_ms: float = 0.0
    dropped_frames: int = 0
    total_frames: int = 0
    cpu_usage: float = 0.0

class FrameRateController:
    """Fixed FPS controller with precise timing and statistics"""

    def __init__(self, target_fps: float = 30.0, enable_vsync: bool = True):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.enable_vsync = enable_vsync

        # Timing tracking
        self.last_frame_time = 0.0
        self.frame_times = []
        self.max_frame_history = 30  # Track last 30 frames for average

        # Statistics
        self.stats = FrameStats(target_fps=target_fps)
        self.start_time = time.time()

    def wait_for_next_frame(self) -> float:
        """Wait for next frame with precise timing, returns actual frame time"""
        current_time = time.time()

        if self.last_frame_time > 0:
            elapsed = current_time - self.last_frame_time

            if self.enable_vsync and elapsed < self.frame_interval:
                # Sleep for remaining time to maintain target FPS
                sleep_time = self.frame_interval - elapsed
                time.sleep(sleep_time)
                current_time = time.time()

            # Update frame timing statistics
            actual_frame_time = current_time - self.last_frame_time
            self._update_stats(actual_frame_time)

        self.last_frame_time = current_time
        return current_time

    def _update_stats(self, frame_time: float):
        """Update frame rate statistics"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)

        # Calculate average FPS from recent frames
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.stats.actual_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
            self.stats.frame_time_ms = avg_frame_time * 1000

        self.stats.total_frames += 1

    def get_stats(self) -> FrameStats:
        """Get current frame rate statistics"""
        return self.stats

    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = FrameStats(target_fps=self.target_fps)
        self.frame_times.clear()
        self.start_time = time.time()

class FramePipeline:
    """Asynchronous frame processing pipeline with double buffering"""

    def __init__(self, frame_source: Callable[[], np.ndarray], buffer_size: int = 3):
        self.frame_source = frame_source
        self.buffer_size = buffer_size

        # Frame buffers
        self.capture_queue = Queue(maxsize=buffer_size)
        self.processing_queue = Queue(maxsize=buffer_size)

        # Threading
        self.capture_thread = None
        self.running = False
        self.frame_ready_event = threading.Event()

        # Statistics
        self.frames_captured = 0
        self.frames_processed = 0
        self.frames_dropped = 0

    def start(self):
        """Start the asynchronous frame pipeline"""
        if self.running:
            return

        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop(self):
        """Stop the frame pipeline"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)

    def _capture_loop(self):
        """Background frame capture loop"""
        while self.running:
            try:
                # Capture frame from source
                frame = self.frame_source()
                self.frames_captured += 1

                # Add to queue (non-blocking)
                try:
                    self.capture_queue.put_nowait(frame)
                    self.frame_ready_event.set()
                except:
                    # Queue full - drop oldest frame
                    try:
                        self.capture_queue.get_nowait()
                        self.capture_queue.put_nowait(frame)
                        self.frames_dropped += 1
                    except:
                        pass

                # Small sleep to prevent excessive CPU usage
                time.sleep(0.001)

            except Exception:
                # Continue on capture errors
                time.sleep(0.01)

    def get_frame(self, timeout: float = 0.033) -> Optional[np.ndarray]:
        """Get next processed frame with timeout"""
        try:
            # Wait for frame to be available
            if self.frame_ready_event.wait(timeout):
                frame = self.capture_queue.get_nowait()
                self.frames_processed += 1

                # Reset event if queue is empty
                if self.capture_queue.empty():
                    self.frame_ready_event.clear()

                return frame
        except Empty:
            pass

        return None

    def get_pipeline_stats(self) -> dict:
        """Get pipeline performance statistics"""
        return {
            'frames_captured': self.frames_captured,
            'frames_processed': self.frames_processed,
            'frames_dropped': self.frames_dropped,
            'queue_size': self.capture_queue.qsize(),
            'drop_rate': self.frames_dropped / max(1, self.frames_captured)
        }

class FrameMemoryPool:
    """Memory pool for frame buffers to reduce allocation overhead"""

    def __init__(self, frame_shape: tuple, pool_size: int = 10, dtype=np.uint8):
        self.frame_shape = frame_shape
        self.pool_size = pool_size
        self.dtype = dtype

        # Pre-allocate frame buffers
        self.available_frames = []
        self.in_use_frames = set()

        for _ in range(pool_size):
            frame = np.zeros(frame_shape, dtype=dtype)
            self.available_frames.append(frame)

    def get_frame(self) -> np.ndarray:
        """Get a frame buffer from the pool"""
        if self.available_frames:
            frame = self.available_frames.pop()
            self.in_use_frames.add(id(frame))
            return frame
        else:
            # Pool exhausted - allocate new frame
            return np.zeros(self.frame_shape, dtype=self.dtype)

    def return_frame(self, frame: np.ndarray):
        """Return a frame buffer to the pool"""
        frame_id = id(frame)
        if frame_id in self.in_use_frames:
            self.in_use_frames.remove(frame_id)

            if len(self.available_frames) < self.pool_size:
                # Clear frame data and return to pool
                frame.fill(0)
                self.available_frames.append(frame)

    def get_pool_stats(self) -> dict:
        """Get memory pool statistics"""
        return {
            'pool_size': self.pool_size,
            'available_frames': len(self.available_frames),
            'in_use_frames': len(self.in_use_frames),
            'utilization': len(self.in_use_frames) / self.pool_size
        }
