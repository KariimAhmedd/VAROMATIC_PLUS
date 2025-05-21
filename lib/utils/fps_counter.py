"""
FPS counter utility for VAROMATIC+
"""

import time
from collections import deque

class FPS_Counter:
    """
    Utility class for measuring and calculating frames per second
    """
    
    def __init__(self, max_samples=30):
        """
        Initialize the FPS counter
        
        Args:
            max_samples: Maximum number of samples to keep for calculating average FPS
        """
        self.frame_times = deque(maxlen=max_samples)
        self.last_tick = None
        self.fps = 0
        
    def tick(self):
        """
        Record a new frame tick
        
        Returns:
            Current FPS estimate
        """
        current_time = time.time()
        
        if self.last_tick is not None:
            # Calculate time difference
            delta = current_time - self.last_tick
            self.frame_times.append(delta)
            
        self.last_tick = current_time
        
        # Calculate FPS
        if len(self.frame_times) > 0:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_time if avg_time > 0 else 0
            
        return self.fps
        
    def get_fps(self):
        """
        Get the current FPS estimate
        
        Returns:
            Current frames per second
        """
        return self.fps
        
    def reset(self):
        """
        Reset the FPS counter
        """
        self.frame_times.clear()
        self.last_tick = None
        self.fps = 0 