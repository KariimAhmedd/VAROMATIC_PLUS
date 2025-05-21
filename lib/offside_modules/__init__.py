"""
VAROMATIC+ Offside Detection Modules

This package contains modules for soccer offside detection including:
- Player detection and tracking
- Field line detection
- Color analysis
- Offside detection logic
"""

from .offside_detection import OffsideDetector
from .player_detection import PlayerDetector
from .color_analysis import ColorAnalyzer
from .field_detection import FieldLineDetector
from .player_tracking import PlayerTracker

__all__ = [
    'OffsideDetector',
    'PlayerDetector',
    'ColorAnalyzer',
    'FieldLineDetector',
    'PlayerTracker'
] 