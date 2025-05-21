import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import math
import time

class FieldLineDetector:
    def __init__(self):
        """Initialize field line detector with optimized parameters."""
        # Basic line detection parameters
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        self.hough_threshold = 50
        self.min_line_length = 40
        self.max_line_gap = 10
        
        # Field color detection (green)
        self.field_color_lower = np.array([35, 40, 40])  # Green in HSV
        self.field_color_upper = np.array([90, 255, 255])
        
        # Line tracking for stability
        self.previous_lines = []
        self.history_length = 5
        self.history_weight = 0.7
        
        # Field markers 
        self.center_line = None
        self.goal_lines = {"left": None, "right": None}
        self.penalty_areas = {"left": None, "right": None}
        
        # Field dimensions (standard soccer field in meters)
        self.field_width = 68  # Standard width 64-75m
        self.field_length = 105  # Standard length 100-110m
        
        self.line_detector = None
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the line detector models."""
        # Set flag to indicate we're using basic line detection
        # We don't log a warning as this is expected behavior
        self.using_basic_detection = True
        
        # Future model loading could happen here when available
        # For now we're using OpenCV-based detection techniques
        
    def detect_field(self, frame: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Detect field lines and mask with better error handling.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (field lines, field mask)
        """
        try:
            # Create field mask
            field_mask = self.create_field_mask(frame)
            
            # Extract edges from masked frame
            edges = self.extract_edges(frame, field_mask)
            
            # Detect lines in edges
            lines = self.detect_lines(edges)
            
            # Filter lines
            lines = self.filter_lines(lines, frame.shape[:2])
            
            # Update field model
            if lines is not None and len(lines) > 0:
                self.update_field_model(lines, frame.shape[:2])
                
            return lines, field_mask
            
        except Exception as e:
            print(f"Error in detect_field: {str(e)}")
            return [], np.zeros_like(frame[:,:,0])
        
    def create_field_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create a binary mask for the soccer field area with improved detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Binary mask of field area
        """
        try:
            # Convert to different color spaces for better segmentation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # Extract field using multiple color ranges for better coverage
            # Standard green range in HSV
            mask1 = cv2.inRange(hsv, self.field_color_lower, self.field_color_upper)
            
            # Darker green range for shadows
            dark_lower = np.array([35, 40, 20])
            dark_upper = np.array([90, 255, 150])
            mask2 = cv2.inRange(hsv, dark_lower, dark_upper)
            
            # Lighter green range for bright areas
            bright_lower = np.array([35, 30, 150])
            bright_upper = np.array([90, 180, 255])
            mask3 = cv2.inRange(hsv, bright_lower, bright_upper)
            
            # LAB color space for additional detection
            # Green in LAB space
            green_lab_lower = np.array([100, 110, 120])
            green_lab_upper = np.array([200, 140, 180])
            mask4 = cv2.inRange(lab, green_lab_lower, green_lab_upper)
            
            # Combine masks
            field_mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, cv2.bitwise_or(mask3, mask4)))
            
            # Clean up the mask
            kernel_open = np.ones((5, 5), np.uint8)
            kernel_close = np.ones((15, 15), np.uint8)
            
            field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel_open)
            field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel_close)
            
            # Fill holes in the mask
            # Find contours in the mask
            contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                # Find the largest contour (which should be the field)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Create a new mask with only the largest contour
                mask_largest = np.zeros_like(field_mask)
                cv2.drawContours(mask_largest, [largest_contour], 0, 255, -1)
                
                field_mask = mask_largest
            
            return field_mask
            
        except Exception as e:
            print(f"Error in create_field_mask: {str(e)}")
            return np.zeros_like(frame[:,:,0])
    
    def extract_edges(self, frame: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
        """Extract edges from the frame, limited to the field area.
        
        Args:
            frame: Input frame
            field_mask: Binary mask of field area
            
        Returns:
            Edge detection image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply mask to limit to field area
        masked_gray = cv2.bitwise_and(gray, gray, mask=field_mask)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(masked_gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        
        return edges
        
    def detect_lines(self, edges: np.ndarray) -> List[np.ndarray]:
        """Detect lines in the edge image.
        
        Args:
            edges: Edge detection image
            
        Returns:
            List of detected lines
        """
        # Use HoughLinesP to detect lines
        lines = cv2.HoughLinesP(edges, 
                              rho=1, 
                              theta=np.pi/180, 
                              threshold=self.hough_threshold,
                              minLineLength=self.min_line_length, 
                              maxLineGap=self.max_line_gap)
        
        if lines is None:
            return []
        
        return lines
        
    def filter_lines(self, lines: List[np.ndarray], frame_shape: Tuple[int, int]) -> List[np.ndarray]:
        """Filter detected lines to keep only relevant pitch lines.
        
        Args:
            lines: Detected lines
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            Filtered list of lines
        """
        if lines is None or len(lines) == 0:
            return []
            
        filtered_lines = []
        h, w = frame_shape[:2]
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line length
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            # Calculate line angle
            if x2 != x1:  # Avoid division by zero
                angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
            else:
                angle = 90
                
            # Keep horizontal and vertical lines (with tolerance)
            is_horizontal = (angle < 20) or (angle > 160)
            is_vertical = (70 < angle < 110)
            
            # Minimum length requirement depends on orientation
            min_h_length = w * 0.1  # 10% of frame width for horizontal
            min_v_length = h * 0.15  # 15% of frame height for vertical
            
            # Keep lines that are long enough and have the right orientation
            if (is_horizontal and length > min_h_length) or (is_vertical and length > min_v_length):
                filtered_lines.append(line)
        
        return filtered_lines
        
    def update_field_model(self, lines: List[np.ndarray], frame_shape: Tuple[int, int]):
        """Update the field model with detected lines.
        
        Args:
            lines: Detected field lines
            frame_shape: Shape of the frame
        """
        if lines is None or len(lines) == 0:
            return
            
        h, w = frame_shape[:2]
        vertical_lines = []
        horizontal_lines = []
        
        # Separate horizontal and vertical lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            if x2 != x1:  # Avoid division by zero
                angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
            else:
                angle = 90
                
            # Categorize lines
            if angle < 20 or angle > 160:  # Horizontal
                horizontal_lines.append(line[0])
            elif 70 < angle < 110:  # Vertical
                vertical_lines.append(line[0])
        
        # Find center line (vertical line near center)
        center_x = w // 2
        nearest_center_line = None
        min_center_dist = float('inf')
        
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            line_x = (x1 + x2) / 2
            dist = abs(line_x - center_x)
            
            if dist < min_center_dist and dist < w * 0.15:  # Within 15% of center
                min_center_dist = dist
                nearest_center_line = line
                
        self.center_line = nearest_center_line
        
        # Find goal lines (vertical lines near edges)
        left_goal_x = w * 0.05  # 5% from left edge
        right_goal_x = w * 0.95  # 5% from right edge
        
        nearest_left_goal_line = None
        nearest_right_goal_line = None
        min_left_dist = float('inf')
        min_right_dist = float('inf')
        
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            line_x = (x1 + x2) / 2
            
            # Left goal line
            dist_left = abs(line_x - left_goal_x)
            if dist_left < min_left_dist and line_x < w * 0.2:  # Within 20% of left edge
                min_left_dist = dist_left
                nearest_left_goal_line = line
                
            # Right goal line
            dist_right = abs(line_x - right_goal_x)
            if dist_right < min_right_dist and line_x > w * 0.8:  # Within 20% of right edge
                min_right_dist = dist_right
                nearest_right_goal_line = line
                
        self.goal_lines["left"] = nearest_left_goal_line
        self.goal_lines["right"] = nearest_right_goal_line
        
    def draw_field_lines(self, frame: np.ndarray) -> np.ndarray:
        """Draw the detected field lines on the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with field lines drawn
        """
        result = frame.copy()
        
        # Draw center line
        if self.center_line is not None:
            x1, y1, x2, y2 = self.center_line
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw goal lines
        if self.goal_lines["left"] is not None:
            x1, y1, x2, y2 = self.goal_lines["left"]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        if self.goal_lines["right"] is not None:
            x1, y1, x2, y2 = self.goal_lines["right"]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return result
    
    def get_offside_line(self, player_boxes: List[List[float]], team_assignments: List[int], 
                        defending_team: int) -> Tuple[Optional[int], np.ndarray]:
        """Calculate the offside line position.
        
        Args:
            player_boxes: List of player bounding boxes [x, y, w, h]
            team_assignments: List of team assignments (0 or 1)
            defending_team: Team that is defending (0 or 1)
            
        Returns:
            Tuple of (offside line x-position, frame with visualization)
        """
        if not player_boxes or not team_assignments:
            return None, np.array([])
            
        frame_width = 1000  # Default width
        
        # Find the second-to-last defender of the defending team
        defending_players_x = []
        
        for i, (box, team) in enumerate(zip(player_boxes, team_assignments)):
            if team == defending_team:
                x, y, w, h = box
                player_x = x + w / 2  # Center x-coordinate
                defending_players_x.append(player_x)
                
        if len(defending_players_x) < 2:
            return None, np.array([])
            
        # Sort to find second-to-last defender (last is usually goalkeeper)
        defending_players_x.sort()
        
        # Get the position based on attacking direction
        if defending_team == 0:  # Team 0 defends left side
            offside_line_x = defending_players_x[1]  # Second player from the left
        else:  # Team 1 defends right side
            offside_line_x = defending_players_x[-2]  # Second player from the right
            
        return int(offside_line_x), np.array([])
            
    def draw_offside_line(self, frame: np.ndarray, offside_line_x: Optional[int], 
                         is_offside: bool) -> np.ndarray:
        """Draw enhanced offside line with visual indicators.
        
        Args:
            frame: Input frame
            offside_line_x: X-coordinate of the offside line
            is_offside: Whether an offside situation is detected
            
        Returns:
            Annotated frame with offside line
        """
        if offside_line_x is None:
            return frame
            
        h, w = frame.shape[:2]
        
        # Create a copy to avoid modifying the original
        result_frame = frame.copy()
        
        # Add line with pulsing effect for offsides
        if is_offside:
            # Create pulsing effect based on time for offside line
            pulse = 0.7 + 0.3 * np.sin(time.time() * 10)
            thickness = max(2, int(5 * pulse))
            line_color = (0, 0, 255)  # Red for offside
            
            # Draw the main offside line
            cv2.line(result_frame, (offside_line_x, 0), (offside_line_x, h), line_color, thickness)
            
            # Add a semi-transparent overlay region for the offside area
            overlay = result_frame.copy()
            side_of_field = "left" if offside_line_x < w // 2 else "right"
            
            if side_of_field == "left":
                # Create the shaded region (left of the line)
                cv2.rectangle(overlay, (0, 0), (offside_line_x, h), (0, 0, 255), -1)
            else:
                # Create the shaded region (right of the line)
                cv2.rectangle(overlay, (offside_line_x, 0), (w, h), (0, 0, 255), -1)
            
            # Apply transparency
            alpha = 0.15 * pulse
            cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)
            
            # Add offside text
            cv2.putText(result_frame, "OFFSIDE", 
                       (w - 200 if side_of_field == "left" else 20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Add animated arrow indicators
            arrow_x = offside_line_x + (15 if side_of_field == "right" else -15)
            arrow_length = 20
            arrow_spacing = 100
            arrow_animation = int(time.time() * 4) % arrow_spacing
            
            for y in range(arrow_animation, h, arrow_spacing):
                if side_of_field == "left":
                    # Draw arrow pointing right
                    cv2.arrowedLine(result_frame, 
                                   (arrow_x - arrow_length, y), 
                                   (arrow_x, y), 
                                   (0, 0, 255), 3, tipLength=0.5)
                else:
                    # Draw arrow pointing left
                    cv2.arrowedLine(result_frame, 
                                   (arrow_x + arrow_length, y), 
                                   (arrow_x, y), 
                                   (0, 0, 255), 3, tipLength=0.5)
        else:
            # Normal line for non-offside situations
            cv2.line(result_frame, (offside_line_x, 0), (offside_line_x, h), (0, 255, 0), 2)
            
        return result_frame
        
    def detect_field_lines(self, frame: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Detect field lines with improved robustness.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (list of line segments, line mask)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Multi-scale line detection
        line_mask = np.zeros_like(gray)
        detected_lines = []
        
        scales = [0.5, 1.0, 2.0]  # Multiple scales for better detection
        for scale in scales:
            # Resize for current scale
            if scale != 1.0:
                scaled = cv2.resize(gray, None, fx=scale, fy=scale)
            else:
                scaled = gray
                
            # Edge detection with dynamic thresholding
            med_val = np.median(scaled)
            lower = int(max(0, (1.0 - 0.33) * med_val))
            upper = int(min(255, (1.0 + 0.33) * med_val))
            edges = cv2.Canny(scaled, lower, upper)
            
            # Probabilistic Hough transform with optimized parameters
            min_length = int(scaled.shape[1] * 0.05)  # Minimum 5% of width
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                  threshold=50,
                                  minLineLength=min_length,
                                  maxLineGap=min_length//2)
                                  
            if lines is not None:
                # Scale back to original size
                if scale != 1.0:
                    lines = (lines / scale).astype(np.int32)
                detected_lines.extend(lines)
                
                # Draw lines on mask
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
                    
        # Clean up line mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)
        
        return detected_lines, line_mask
        
    def get_vanishing_point(self, lines: List[np.ndarray]) -> Tuple[int, int]:
        """Calculate the vanishing point from detected lines.
        
        Args:
            lines: List of detected lines
            
        Returns:
            Tuple of (x, y) coordinates of vanishing point
        """
        if not lines:
            return None
            
        # Convert lines to points format
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.append([[x1, y1], [x2, y2]])
            
        # Use RANSAC to find vanishing point
        best_point = None
        max_inliers = 0
        
        for _ in range(100):  # RANSAC iterations
            # Randomly select two lines
            line1, line2 = np.random.choice(points, 2, replace=False)
            
            # Calculate intersection
            x1, y1 = line1[0]
            x2, y2 = line1[1]
            x3, y3 = line2[0]
            x4, y4 = line2[1]
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                continue
                
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            
            # Count inliers
            inliers = 0
            for point in points:
                # Calculate distance from point to line
                x1, y1 = point[0]
                x2, y2 = point[1]
                dist = np.abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
                if dist < 10:  # Threshold for inlier
                    inliers += 1
                    
            if inliers > max_inliers:
                max_inliers = inliers
                best_point = (int(px), int(py))
                
        return best_point if best_point is not None else (0, 0)
        
    def refine_offside_line(self, frame: np.ndarray, player_position: Tuple[int, int], 
                           vanishing_point: Tuple[int, int]) -> Optional[Tuple[float, float, float]]:
        """Refine the offside line using field lines.
        
        Args:
            frame: Input frame
            player_position: Position of player to draw line from
            vanishing_point: Calculated vanishing point
            
        Returns:
            Optional tuple of (slope, intercept, angle)
        """
        try:
            # Detect field lines
            lines, line_mask = self.detect_field_lines(frame)
            
            if not lines or len(lines) == 0:
                print("No field lines detected")
                return None
                
            # Find closest field line to player
            min_dist = float('inf')
            closest_line = None
            
            for line in lines:
                try:
                    x1, y1, x2, y2 = line[0]
                    # Check for valid line coordinates
                    if not all(isinstance(x, (int, float)) for x in [x1, y1, x2, y2]):
                        continue
                        
                    # Avoid division by zero
                    if (y2-y1)**2 + (x2-x1)**2 == 0:
                        continue
                        
                    dist = np.abs((y2-y1)*player_position[0] - (x2-x1)*player_position[1] + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_line = line
                except Exception as e:
                    print(f"Error processing line: {str(e)}")
                    continue
                    
            if closest_line is None:
                print("No valid closest line found")
                return None
                
            # Calculate refined line parameters
            x1, y1, x2, y2 = closest_line[0]
            
            # Handle vertical line case
            if abs(x2 - x1) < 1e-6:
                return (float('inf'), x1, 90.0)
                
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            angle = np.arctan2(y2 - y1, x2 - x1)
            
            return (slope, intercept, angle)
            
        except Exception as e:
            print(f"Error in refine_offside_line: {str(e)}")
            return None 