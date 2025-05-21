import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time
from .player_detection import PlayerDetector
from .field_detection import FieldLineDetector
import math

@dataclass
class Ball:
    """Class representing the detected ball."""
    position: Tuple[float, float]  # (x, y) center position
    velocity: Tuple[float, float]  # (vx, vy) velocity vector
    confidence: float  # Detection confidence
    box: List[float]  # [x, y, w, h] bounding box

class GoalDetector:
    def __init__(self):
        """Initialize the goal detector with enhanced components."""
        # Initialize ball tracking
        self.last_ball_pos = None
        self.ball_history = []
        self.max_history = 30  # Store last 30 frames
        
        # Goal line parameters
        self.left_goal_line = None   # (x1, y1, x2, y2)
        self.right_goal_line = None  # (x1, y1, x2, y2)
        self.goal_line_margin = 10   # Increased margin
        
        # Goal posts parameters
        self.left_goal_posts = None  # [(x1, y1), (x2, y2)]
        self.right_goal_posts = None # [(x1, y1), (x2, y2)]
        self.post_height_ratio = 2.5  # Expected ratio of post height to width
        
        # Goal state
        self.goal_scored = False
        self.goal_celebration_frames = 0
        self.max_celebration_frames = 90  # Increased from 45 to 90 (3 seconds at 30 fps)
        
        # Initialize field detector for goal line detection
        self.field_detector = FieldLineDetector()
        
        # Enhanced ball detection parameters
        self.ball_hsv_ranges = [
            # White ball with wider range
            (np.array([0, 0, 180]), np.array([180, 40, 255])),
            # Yellow-white ball with wider range
            (np.array([15, 0, 180]), np.array([40, 40, 255])),
            # Pure white with high saturation tolerance
            (np.array([0, 0, 200]), np.array([180, 60, 255]))
        ]
        
        # Kalman filter for ball tracking
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        # Increased process noise for better adaptation
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32) * 0.1
        self.kalman_initialized = False
        
        # Ball trajectory prediction
        self.trajectory_points = []
        self.max_trajectory_points = 10
        
        # Goal post detection parameters
        self.net_detection_threshold = 0.4
        self.goal_line_threshold = 85  # Contrast threshold for goal line (higher value = more strict)
        self.ball_position_history = []
        self.goal_event_cooldown = 8.0  # Increased from 5.0 to 8.0 seconds between goal events
        self.last_goal_time = 0
        self.is_celebrating = False
        
        # Field lines detection parameters
        self.hough_threshold = 50
        self.min_line_length = 100
        self.max_line_gap = 30
        
        # Goal posts color range (white)
        self.goal_post_lower = np.array([0, 0, 200])
        self.goal_post_upper = np.array([180, 30, 255])
        
    def detect_ball(self, frame: np.ndarray) -> Optional[Ball]:
        """Detect soccer ball in the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Ball object with position, velocity and other properties
        """
        try:
            # Convert frame to HSV for better color segmentation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # White/yellow ball color range (more permissive)
            ball_lower1 = np.array([0, 0, 180])    # White
            ball_upper1 = np.array([30, 40, 255])
            
            ball_lower2 = np.array([20, 100, 180])  # Yellow
            ball_upper2 = np.array([35, 255, 255])
            
            # Create masks and combine
            mask1 = cv2.inRange(hsv, ball_lower1, ball_upper1)
            mask2 = cv2.inRange(hsv, ball_lower2, ball_upper2)
            ball_mask = cv2.bitwise_or(mask1, mask2)
            
            # Morphological operations to reduce noise
            kernel = np.ones((3, 3), np.uint8)
            ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
            ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If no contours found
            if not contours:
                return None
                
            # Filter contours by size, circularity
            max_ball_contour = None
            max_ball_radius = 0
            
            for cnt in contours:
                # Get area and perimeter
                area = cv2.contourArea(cnt)
                if area < 20:  # Minimum ball size
                    continue
                    
                # Check circularity
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Balls are circular
                if 0.5 < circularity:
                    # Find enclosing circle
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    
                    # Size constraints
                    if 3 < radius < 30:
                        if radius > max_ball_radius:
                            max_ball_radius = radius
                            max_ball_contour = cnt
            
            if max_ball_contour is not None:
                # Get ball properties
                (x, y), radius = cv2.minEnclosingCircle(max_ball_contour)
                
                # Update ball position history
                self.ball_position_history.append((int(x), int(y)))
                if len(self.ball_position_history) > self.max_history:
                    self.ball_position_history.pop(0)
                
                # Calculate ball velocity if enough history
                velocity_x, velocity_y = 0, 0
                if len(self.ball_position_history) > 5:
                    prev_x, prev_y = self.ball_position_history[-6]
                    curr_x, curr_y = self.ball_position_history[-1]
                    velocity_x = (curr_x - prev_x) / 5  # Pixels per frame
                    velocity_y = (curr_y - prev_y) / 5
                
                # Create and return Ball object
                return Ball(
                    position=(int(x), int(y)),
                    velocity=(velocity_x, velocity_y),
                    confidence=circularity,
                    box=[int(x - radius), int(y - radius), int(radius * 2), int(radius * 2)]
                )
                
            return None
        except Exception as e:
            print(f"Error detecting ball: {str(e)}")
            return None
    
    def detect_goal_posts(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect goal posts in the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of goal post bounding boxes [x, y, w, h]
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for white goal posts
            goal_post_mask = cv2.inRange(hsv, self.goal_post_lower, self.goal_post_upper)
            
            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            goal_post_mask = cv2.morphologyEx(goal_post_mask, cv2.MORPH_OPEN, kernel)
            goal_post_mask = cv2.morphologyEx(goal_post_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(goal_post_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            goal_posts = []
            for cnt in contours:
                # Filter by area
                area = cv2.contourArea(cnt)
                if area < 200:  # Min area for goal post
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Goal posts are typically vertical (height > width)
                if h > 2*w:
                    goal_posts.append((x, y, w, h))
            
            return goal_posts
        except Exception as e:
            print(f"Error detecting goal posts: {str(e)}")
            return []
            
    def _predict_ball_trajectory(self, ball: Ball, frame_shape: tuple) -> list:
        """Predict future ball trajectory based on current ball velocity.
        
        Args:
            ball: Ball object with current position and velocity
            frame_shape: Shape of the video frame
            
        Returns:
            List of predicted future points [(x1,y1), (x2,y2), ...]
        """
        if not hasattr(ball, 'position') or not hasattr(ball, 'velocity'):
            return []
        
        # Extract info
        pos_x, pos_y = ball.position
        vel_x, vel_y = ball.velocity
        height, width = frame_shape[:2]
        
        # Only predict if there's significant movement
        min_velocity = 3.0  # Minimum velocity magnitude to predict trajectory
        velocity_magnitude = np.sqrt(vel_x**2 + vel_y**2)
        
        if velocity_magnitude < min_velocity:
            return []
        
        # Number of future points to predict
        num_points = min(20, int(velocity_magnitude * 1.5))
        
        # Gravity factor for more realistic trajectory
        gravity = 0.5
        
        # Generate trajectory points
        trajectory = []
        for i in range(1, num_points + 1):
            # Predict position with gravity effect
            pred_x = pos_x + vel_x * i
            pred_y = pos_y + vel_y * i + 0.5 * gravity * i * i
            
            # Check if point is inside frame
            if 0 <= pred_x < width and 0 <= pred_y < height:
                trajectory.append((int(pred_x), int(pred_y)))
            else:
                break
            
        return trajectory

    def draw_ball(self, frame: np.ndarray, ball: Ball, highlight: bool = True) -> np.ndarray:
        """Draw enhanced ball visualization with effects.
        
        Args:
            frame: Input frame to draw on
            ball: Ball object with position and other info
            highlight: Whether to highlight the ball with effects
            
        Returns:
            Frame with ball visualization
        """
        if ball is None or not hasattr(ball, 'position'):
            return frame
        
        result = frame.copy()
        x, y = map(int, ball.position)
        
        # Get ball size
        ball_radius = 5
        if hasattr(ball, 'box') and len(ball.box) >= 3:
            ball_radius = max(5, int(min(ball.box[2], ball.box[3]) / 2))
        
        # Draw trajectory prediction if ball is moving
        if hasattr(ball, 'velocity'):
            trajectory = self._predict_ball_trajectory(ball, frame.shape)
            
            # Draw trajectory points with fading effect
            if trajectory:
                for i, (tx, ty) in enumerate(trajectory):
                    # Calculate alpha based on position in trajectory
                    alpha = 0.9 - (i / len(trajectory)) * 0.8
                    # Calculate radius - smaller for further points
                    t_radius = max(1, int(ball_radius * (1 - 0.7 * i / len(trajectory))))
                    # Calculate color - from white to red
                    g_val = int(255 * (1 - i / len(trajectory)))
                    color = (0, g_val, 255)
                    
                    overlay = result.copy()
                    cv2.circle(overlay, (tx, ty), t_radius, color, -1, cv2.LINE_AA)
                    cv2.addWeighted(overlay, alpha, result, 1-alpha, 0, result)
        
        # Enhanced ball visualization with highlight effect
        if highlight:
            # Outer glow effect
            for r in range(ball_radius + 6, ball_radius, -2):
                alpha = 0.2 - (r - ball_radius) * 0.03
                overlay = result.copy()
                cv2.circle(overlay, (x, y), r, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, result, 1-alpha, 0, result)
            
            # Main ball with white center
            cv2.circle(result, (x, y), ball_radius, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.circle(result, (x, y), ball_radius - 2, (255, 255, 255), -1, cv2.LINE_AA)
            
            # Add highlight reflection
            highlight_x = x - ball_radius // 3
            highlight_y = y - ball_radius // 3
            highlight_size = max(1, ball_radius // 4)
            cv2.circle(result, (highlight_x, highlight_y), highlight_size, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            # Simple ball drawing
            cv2.circle(result, (x, y), ball_radius, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.circle(result, (x, y), ball_radius - 1, (255, 255, 255), -1, cv2.LINE_AA)
        
        return result

    def detect_goal(self, frame: np.ndarray, ball: Optional[Ball] = None) -> Tuple[bool, np.ndarray]:
        """Detect if a goal has been scored with enhanced visualization.
        
        Args:
            frame: Input frame
            ball: Ball object from detect_ball
            
        Returns:
            Tuple of (is_goal, annotated_frame)
        """
        try:
            frame_copy = frame.copy()
            is_goal = False
            
            # Detect goal posts
            goal_posts = self.detect_goal_posts(frame)
            
            # Draw goal posts for visualization
            for gp in goal_posts:
                x, y, w, h = gp
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # If we have goal posts and a ball
            if len(goal_posts) >= 2 and ball is not None and hasattr(ball, 'position'):
                # Draw ball with effects
                frame_copy = self.draw_ball(frame_copy, ball, highlight=True)
                
                # Get ball position
                ball_x, ball_y = ball.position
                
                # Calculate goal boundaries
                goal_posts_sorted = sorted(goal_posts, key=lambda p: p[0])  # Sort by x coordinate
                
                # Determine if the goal is on left or right side
                frame_width = frame.shape[1]
                leftmost = min(goal_posts, key=lambda p: p[0])
                rightmost = max(goal_posts, key=lambda p: p[0] + p[2])
                
                # Check if goal is on left side
                if leftmost[0] < frame_width * 0.3:  # Left 30% of frame
                    left_goal = True
                    goal_x1 = leftmost[0]
                    goal_x2 = leftmost[0] + leftmost[2] + 20  # Add some margin
                    goal_y1 = min(leftmost[1], rightmost[1])
                    goal_y2 = max(leftmost[1] + leftmost[3], rightmost[1] + rightmost[3])
                # Check if goal is on right side
                elif rightmost[0] + rightmost[2] > frame_width * 0.7:  # Right 30% of frame
                    left_goal = False
                    goal_x1 = rightmost[0] - 20  # Add some margin
                    goal_x2 = rightmost[0] + rightmost[2]
                    goal_y1 = min(leftmost[1], rightmost[1])
                    goal_y2 = max(leftmost[1] + leftmost[3], rightmost[1] + rightmost[3])
                else:
                    return False, frame_copy
                    
                # Check if ball crossed the goal line
                ball_crossed = (left_goal and goal_x1 <= ball_x <= goal_x2 and goal_y1 <= ball_y <= goal_y2) or \
                              (not left_goal and goal_x1 <= ball_x <= goal_x2 and goal_y1 <= ball_y <= goal_y2)
                
                # Draw the goal area
                cv2.rectangle(frame_copy, (int(goal_x1), int(goal_y1)), 
                             (int(goal_x2), int(goal_y2)), (0, 0, 255), 2)
                
                # Check if enough time has passed since last goal
                current_time = time.time()
                time_since_last_goal = current_time - self.last_goal_time
                
                if ball_crossed and time_since_last_goal > self.goal_event_cooldown:
                    # Goal!
                    is_goal = True
                    self.is_celebrating = True
                    self.last_goal_time = current_time
                    
                    # Create radial celebration effect
                    overlay = frame_copy.copy()
                    center_x = int(frame.shape[1] / 2)
                    center_y = int(frame.shape[0] / 2)
                    
                    # Draw expanding circles
                    for r in range(10, 300, 30):
                        alpha = 0.7 - (r / 300)
                        cv2.circle(overlay, (center_x, center_y), r, (0, 0, 255), 3)
                        cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)
                    
                    # Add goal text with shadow effect
                    text = "GOAL!"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_size = cv2.getTextSize(text, font, 3, 5)[0]
                    text_x = center_x - text_size[0] // 2
                    text_y = center_y
                    
                    # Shadow
                    cv2.putText(frame_copy, text, (text_x + 5, text_y + 5),
                               font, 3, (0, 0, 0), 5)
                    # Text
                    cv2.putText(frame_copy, text, (text_x, text_y),
                               font, 3, (0, 0, 255), 5)
                
                # Continue celebration animation if needed
                elif self.is_celebrating:
                    celebration_elapsed = current_time - self.last_goal_time
                    if celebration_elapsed < 3.0:  # 3 seconds celebration
                        # Pulsing GOAL text with nicer visual effect
                        size = 2.0 + np.sin(celebration_elapsed * 5) * 0.5
                        text = "GOAL!"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_size = cv2.getTextSize(text, font, size, 5)[0]
                        text_x = frame_width // 2 - text_size[0] // 2
                        text_y = 100
                        
                        # Shadow effect
                        cv2.putText(frame_copy, text, (text_x + 3, text_y + 3),
                                   font, size, (0, 0, 0), 5)
                        # Main text with dynamic color
                        r = int(255 * (0.7 + 0.3 * np.sin(celebration_elapsed * 8)))
                        cv2.putText(frame_copy, text, (text_x, text_y),
                                   font, size, (0, 0, r), 5)
                    else:
                        self.is_celebrating = False
            
            return is_goal, frame_copy
        except Exception as e:
            print(f"Error detecting goal: {str(e)}")
            return False, frame
            
    def detect_field_lines(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect field lines on a soccer pitch.
        
        Args:
            frame: Input frame
            
        Returns:
            List of lines in format (x1, y1, x2, y2)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Detect lines using HoughLinesP
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                   threshold=self.hough_threshold,
                                   minLineLength=self.min_line_length,
                                   maxLineGap=self.max_line_gap)
            
            # Filter lines
            filtered_lines = []
            if lines is not None:
                # Get image dimensions
                h, w = frame.shape[:2]
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate line slope
                    if x2 != x1:  # Avoid division by zero
                        slope = abs((y2 - y1) / (x2 - x1))
                    else:
                        slope = float('inf')
                    
                    # Keep horizontal and vertical lines
                    if slope < 0.3 or slope > 5:  # Horizontal or vertical
                        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        
                        # Ignore short lines
                        if line_length > self.min_line_length:
                            filtered_lines.append((x1, y1, x2, y2))
            
            return filtered_lines
        except Exception as e:
            print(f"Error detecting field lines: {str(e)}")
            return []
            
    def detect_goal_lines(self, frame: np.ndarray) -> bool:
        """Enhanced goal line detection using multiple methods."""
        try:
            # Get field lines
            lines, _ = self.field_detector.detect_field_lines(frame)
            
            if not lines:
                return False
            
            frame_height, frame_width = frame.shape[:2]
            left_goal_candidates = []
            right_goal_candidates = []
            
            # First pass: detect vertical lines near edges
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check if line is vertical (within margin)
                if abs(x2 - x1) < 10:
                    # Calculate line height
                    height = abs(y2 - y1)
                    
                    # Filter by minimum height
                    if height < frame_height * 0.2:  # At least 20% of frame height
                        continue
                    
                    # Left goal line candidates
                    if x1 < frame_width * 0.2:  # Within 20% of left edge
                        left_goal_candidates.append((x1, y1, x2, y2, height))
                    # Right goal line candidates
                    elif x1 > frame_width * 0.8:  # Within 20% of right edge
                        right_goal_candidates.append((x1, y1, x2, y2, height))
            
            # Second pass: detect goal posts
            def find_goal_posts(candidates):
                if not candidates:
                    return None, None
                
                # Sort by height
                candidates.sort(key=lambda x: x[4], reverse=True)
                
                # Get the tallest line as main post
                main_post = candidates[0]
                
                # Look for parallel post
                parallel_post = None
                main_x = (main_post[0] + main_post[2]) / 2
                
                for candidate in candidates[1:]:
                    candidate_x = (candidate[0] + candidate[2]) / 2
                    
                    # Check if posts are roughly parallel
                    if abs(abs(candidate_x - main_x) - frame_width * 0.05) < 10:  # Expected goal width
                        parallel_post = candidate
                        break
                
                return main_post, parallel_post
            
            # Find posts for both goals
            left_main, left_parallel = find_goal_posts(left_goal_candidates)
            right_main, right_parallel = find_goal_posts(right_goal_candidates)
            
            # Update goal lines and posts
            if left_main:
                self.left_goal_line = left_main[:4]
                if left_parallel:
                    self.left_goal_posts = [(left_main[0], left_main[1]),
                                          (left_parallel[0], left_parallel[1])]
            
            if right_main:
                self.right_goal_line = right_main[:4]
                if right_parallel:
                    self.right_goal_posts = [(right_main[0], right_main[1]),
                                           (right_parallel[0], right_parallel[1])]
            
            return (self.left_goal_line is not None or 
                   self.right_goal_line is not None)
            
        except Exception as e:
            print(f"Error in goal line detection: {str(e)}")
            return False 