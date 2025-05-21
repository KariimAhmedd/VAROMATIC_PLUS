import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# Temporary fix for np.asscalar deprecation
def patch_asscalar(a):
    return a.item()
setattr(np, "asscalar", patch_asscalar)

@dataclass
class TeamColor:
    name: str
    lab_values: Tuple[float, float, float]
    hsv_range: Tuple[List[int], List[int]]
    display_color: Tuple[int, int, int]  # BGR color for visualization

class ColorAnalyzer:
    def __init__(self):
        """Initialize the color analyzer with predefined team colors."""
        # Define common team colors in LAB space for better perceptual matching
        self.team_colors = {
            'red': TeamColor(
                'red',
                (53.2, 80.1, 67.2),
                ([0, 70, 50], [10, 255, 255]),
                (0, 0, 255)  # Red in BGR
            ),
            'blue': TeamColor(
                'blue',
                (32.3, 79.2, -107.9),
                ([100, 70, 50], [130, 255, 255]),
                (255, 0, 0)  # Blue in BGR
            ),
            'yellow': TeamColor(
                'yellow',
                (97.1, -21.6, 94.5),
                ([20, 100, 100], [30, 255, 255]),
                (0, 255, 255)  # Yellow in BGR
            ),
            'green': TeamColor(
                'green',
                (87.7, -86.2, 83.2),
                ([45, 70, 50], [75, 255, 255]),
                (0, 255, 0)  # Green in BGR
            ),
            'purple': TeamColor(
                'purple',
                (29.8, 58.9, -36.5),
                ([130, 70, 50], [150, 255, 255]),
                (255, 0, 255)  # Purple in BGR
            ),
            'orange': TeamColor(
                'orange',
                (65.0, 48.0, 68.0),
                ([10, 100, 100], [20, 255, 255]),
                (0, 128, 255)  # Orange in BGR
            )
        }
        
        self.team_cache = {
            'team1': None,
            'team2': None,
            'team1_color': None,  # BGR color for visualization
            'team2_color': None   # BGR color for visualization
        }
        
        # Add color analysis cache
        self.color_cache = {}
        self.cache_frame_count = 0
        self.cache_lifetime = 5  # Cache colors for 5 frames
        
    def get_dominant_color(self, image: np.ndarray) -> Tuple[float, float, float]:
        """Get the dominant color in LAB color space.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            LAB color values as tuple
        """
        # Resize image for faster processing
        target_size = (32, 32)  # Small size for color analysis
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # Reshape image
        pixels = resized.reshape(-1, 3)
        
        # Use k-means to find dominant colors
        kmeans = KMeans(n_clusters=1, n_init=10)
        kmeans.fit(pixels)
        
        # Get dominant color in BGR and convert to float
        dominant_color = kmeans.cluster_centers_[0].astype(float)
        
        # Convert BGR to RGB
        rgb_color = dominant_color[::-1]
        
        # Convert RGB to LAB
        rgb_obj = sRGBColor(float(rgb_color[0])/255, float(rgb_color[1])/255, float(rgb_color[2])/255)
        lab_obj = convert_color(rgb_obj, LabColor)
        
        # Convert LAB values to float
        lab_l = float(lab_obj.lab_l)
        lab_a = float(lab_obj.lab_a)
        lab_b = float(lab_obj.lab_b)
        
        return (lab_l, lab_a, lab_b)
        
    def match_team_color(self, lab_color: Tuple[float, float, float]) -> str:
        """Match LAB color to closest predefined team color.
        
        Args:
            lab_color: Input color in LAB space
            
        Returns:
            Name of closest matching team color
        """
        # Check cache first
        cache_key = f"{lab_color[0]:.1f},{lab_color[1]:.1f},{lab_color[2]:.1f}"
        if cache_key in self.color_cache:
            return self.color_cache[cache_key]
            
        min_diff = float('inf')
        best_match = None
        
        lab1 = LabColor(*lab_color)
        
        for name, team_color in self.team_colors.items():
            lab2 = LabColor(*team_color.lab_values)
            diff = delta_e_cie2000(lab1, lab2)
            
            if diff < min_diff:
                min_diff = diff
                best_match = name
                
        # Cache the result
        self.color_cache[cache_key] = best_match
        return best_match
        
    def get_team_colors(self, frame: np.ndarray, boxes: List[List[int]], 
                       kept_boxes: List[int]) -> Tuple[Optional[str], Optional[str]]:
        """Determine the two team colors from all detected players.
        
        Args:
            frame: Input frame
            boxes: List of player bounding boxes
            kept_boxes: List of valid detection indices
            
        Returns:
            Tuple of (team1_color, team2_color)
        """
        # Clear cache periodically
        self.cache_frame_count += 1
        if self.cache_frame_count >= self.cache_lifetime:
            self.color_cache.clear()
            self.cache_frame_count = 0
            
        if not kept_boxes or not boxes:
            return self.team_cache['team1'], self.team_cache['team2']
            
        player_colors = []
        
        # Process boxes in batches
        batch_size = 4
        for i in range(0, len(kept_boxes), batch_size):
            batch_indices = kept_boxes[i:i + batch_size]
            # Convert indices to integers and ensure they are valid
            valid_indices = [int(idx) for idx in batch_indices if isinstance(idx, (int, float)) and 0 <= int(idx) < len(boxes)]
            batch_boxes = [boxes[idx] for idx in valid_indices]
            
            for box in batch_boxes:
                x, y, w, h = [int(coord) for coord in box]
                
                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    continue
                    
                player_img = frame[y:y+h, x:x+w]
                
                if player_img.size == 0:
                    continue
                    
                try:
                    # Get dominant color
                    lab_color = self.get_dominant_color(player_img)
                    matched_color = self.match_team_color(lab_color)
                    player_colors.append(matched_color)
                except Exception as e:
                    print(f"Error in color analysis: {str(e)}")
                    continue
                    
        if not player_colors:
            return self.team_cache['team1'], self.team_cache['team2']
            
        # Count occurrences of each color using numpy
        unique_colors, counts = np.unique(player_colors, return_counts=True)
        sorted_indices = np.argsort(-counts)
        sorted_colors = [(unique_colors[i], counts[i]) for i in sorted_indices]
        
        if len(sorted_colors) >= 2:
            self.team_cache['team1'] = sorted_colors[0][0]
            self.team_cache['team2'] = sorted_colors[1][0]
            self.team_cache['team1_color'] = self.team_colors[sorted_colors[0][0]].display_color
            self.team_cache['team2_color'] = self.team_colors[sorted_colors[1][0]].display_color
            
        return self.team_cache['team1'], self.team_cache['team2']
        
    def get_team_display_colors(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Get BGR display colors for both teams."""
        # Always return valid default colors even if detection fails
        default_team1_color = (0, 0, 255)  # Red in BGR
        default_team2_color = (0, 255, 255)  # Yellow in BGR
        
        if 'team1_color' not in self.team_cache or 'team2_color' not in self.team_cache:
            print("Using default team colors as team colors haven't been detected yet")
            self.team_cache['team1_color'] = default_team1_color
            self.team_cache['team2_color'] = default_team2_color
            return default_team1_color, default_team2_color
            
        team1_color = self.team_cache.get('team1_color', default_team1_color)
        team2_color = self.team_cache.get('team2_color', default_team2_color)
        
        return team1_color, team2_color
        
    def get_color_range(self, color_name: str) -> Tuple[List[int], List[int]]:
        """Get HSV color range for a team color.
        
        Args:
            color_name: Name of the team color
            
        Returns:
            Tuple of (lower_bound, upper_bound) in HSV space
        """
        if color_name in self.team_colors:
            return self.team_colors[color_name].hsv_range
        return ([0, 0, 0], [180, 255, 255])  # Full range if color not found 