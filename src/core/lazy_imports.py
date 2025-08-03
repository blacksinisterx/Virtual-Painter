"""
Lazy import system for performance optimization
Delays heavy module loading until actually needed
"""
import importlib
from typing import Any, Optional

class LazyImporter:
    """Lazy importer that loads modules on first access"""

    def __init__(self, module_name: str, fallback_name: Optional[str] = None):
        self.module_name = module_name
        self.fallback_name = fallback_name
        self._module = None
        self._loaded = False

    def __getattr__(self, name: str) -> Any:
        if not self._loaded:
            self._load_module()
        return getattr(self._module, name)

    def _load_module(self):
        """Load the module with fallback support"""
        try:
            self._module = importlib.import_module(self.module_name)
            self._loaded = True
        except ImportError as e:
            if self.fallback_name:
                try:
                    self._module = importlib.import_module(self.fallback_name)
                    self._loaded = True
                except ImportError:
                    raise ImportError(f"Failed to import {self.module_name} or {self.fallback_name}: {e}")
            else:
                raise ImportError(f"Failed to import {self.module_name}: {e}")

    def is_loaded(self) -> bool:
        """Check if module is already loaded"""
        return self._loaded

    def force_load(self):
        """Force immediate loading of the module"""
        if not self._loaded:
            self._load_module()
        return self._module

# Heavy dependencies that benefit from lazy loading
cv2 = LazyImporter('cv2')
mediapipe = LazyImporter('mediapipe', 'mediapipe')
numpy = LazyImporter('numpy', 'np')
