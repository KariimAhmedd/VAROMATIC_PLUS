"""
Utility functions for VAROMATIC+
"""

import logging
import time
import os
from typing import List, Optional


def create_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger with the specified name and level
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatter to ch
    ch.setFormatter(formatter)
    
    # Add ch to logger
    logger.addHandler(ch)
    
    return logger


class FPS_Counter:
    """Track frames per second for performance monitoring"""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times: List[float] = []
        self.last_frame_time: Optional[float] = None
    
    def start_frame(self) -> None:
        """Start timing a new frame"""
        self.last_frame_time = time.time()
    
    def end_frame(self) -> None:
        """End timing current frame and update stats"""
        if self.last_frame_time is None:
            return
            
        # Calculate frame time
        frame_time = time.time() - self.last_frame_time
        self.frame_times.append(frame_time)
        
        # Keep only the last window_size frame times
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
    
    def get_fps(self) -> float:
        """
        Get the current FPS based on recorded frame times
        
        Returns:
            Current FPS or 0 if no frames recorded
        """
        if not self.frame_times:
            return 0.0
            
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0 