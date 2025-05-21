"""
Offside detection modules for VAROMATIC+.

This package contains modules for detecting and analyzing offside situations
in soccer matches, including:
- Player detection and tracking
- Team color analysis
- Offside line detection
- Goal detection
"""

from .player_detection import PlayerDetector
from .color_analysis import ColorAnalyzer
from .offside_detection import OffsideDetector, Player
from .goal_detection import GoalDetector

__all__ = [
    'PlayerDetector',
    'ColorAnalyzer',
    'OffsideDetector',
    'Player',
    'GoalDetector'
] 