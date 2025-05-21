"""
Utility functions for the VAROMATIC+ system.

This package contains common utilities and helper functions used across
the VAROMATIC+ application, including:
- Logging utilities
- FPS counter
- Video processing helpers
- Data conversion utilities
"""

from .logger import create_logger
from .fps_counter import FPS_Counter

__all__ = ['create_logger', 'FPS_Counter']
