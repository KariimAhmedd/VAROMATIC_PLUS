import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum
from .field_detection import FieldLineDetector
from .player_tracking import PlayerTracker
from sklearn.cluster import KMeans
from collections import deque
import numpy.typing as npt
import time
import random

class Direction(Enum):
    LEFT = "left"
    RIGHT = "right"

@dataclass
class Player:
    """Class representing a detected player with enhanced attributes."""
    position: Tuple[float, float]  # (x, y) center position
    box: List[float]  # [x, y, w, h] bounding box
    team: int  # 0 or 1
    keypoints: Optional[np.ndarray] = None  # Optional pose keypoints
    track_id: Optional[int] = None  # Player tracking ID
    role_confidence: float = 0.0  # Confidence in role classification
    is_goalkeeper: bool = False  # Whether player is classified as goalkeeper
    advanced_position: Optional[Tuple[float, float]] = None  # Most forward body part position
    
    def __post_init__(self):
        """Ensure proper data types after initialization."""
        # Convert position to float tuple
        self.position = (float(self.position[0]), float(self.position[1]))
        
        # Convert box coordinates to float
        self.box = [float(x) for x in self.box]
        
        # Ensure team is integer
        self.team = int(self.team)
        
        # Convert keypoints to numpy array if provided
        if self.keypoints is not None and not isinstance(self.keypoints, np.ndarray):
            self.keypoints = np.array(self.keypoints)
            
        # Initialize advanced position with default center position
        if self.advanced_position is None:
            self.advanced_position = self.position

class OffsideDetector:
    def __init__(self):
        """Initialize the offside detector with enhanced components."""
        self.last_frame_players = []
        self.movement_threshold = 10  # pixels
        self.direction_cache = None
        self.direction_confidence = 0.0
        
        # Initialize enhanced components
        self.field_detector = FieldLineDetector()
        self.player_tracker = PlayerTracker()
        
        # Add temporal smoothing
        self.team_history = deque(maxlen=30)  # Store last 30 frames of team assignments
        self.kmeans = KMeans(n_clusters=2, random_state=42)
        
        self.last_offside_time = 0
        self.offside_cooldown = 1.0  # seconds
        self.offside_line_color = (255, 0, 0)  # Red
        self.line_thickness = 2
        
        # Perspective analysis components
        self.field_corners = None
        self.vanishing_point = None
        self.perspective_confident = False
        self.last_frame = None
        self.reference_points = []
        
        # Add memory for players' positions
        self.player_positions_history = deque(maxlen=30)
        
        # Field dimensions estimation
        self.field_width_estimate = 68.0  # meters (standard soccer field)
        self.field_height_estimate = 105.0  # meters (standard soccer field)
        self.pixels_per_meter = None
        
    def _assign_teams_with_clustering(self, players: List[Player]) -> List[int]:
        """Assign teams using enhanced clustering with color and position information.
        
        Args:
            players: List of detected players
            
        Returns:
            List of team assignments (0 or 1)
        """
        if len(players) < 2:
            return [0] * len(players)
            
        try:
            # Get player positions as 2D array
            positions = []
            player_centers = []
            
            for p in players:
                x, y = p.position
                positions.append([float(x), float(y)])  # Ensure float type
                player_centers.append((float(x), float(y)))
                
            positions = np.array(positions)
            
            # Ensure positions is 2D array with shape (n_samples, n_features)
            if len(positions.shape) == 1:
                positions = positions.reshape(-1, 2)
            
            # Analyze player distribution for better clustering
            field_width = self.last_frame.shape[1] if hasattr(self, 'last_frame') else 1280
            field_height = self.last_frame.shape[0] if hasattr(self, 'last_frame') else 720
            
            # First check if we have enough players to estimate formation
            if len(positions) >= 6:
                # Try to detect formation structure
                formation_clusters = {}
                formation_weights = {}
                
                # Store the best team assignments based on different features
                position_assignments = None
                color_assignments = None
                history_assignments = None
                
                # 1. Position-based clustering (standard KMeans)
                self.kmeans.fit(positions)
                position_assignments = self.kmeans.labels_.tolist()  # Convert to list
                
                # Get team centers for later use
                team_centers = self.kmeans.cluster_centers_
                team0_center = team_centers[0]
                team1_center = team_centers[1]
                
                # Ensure consistent ordering (team 0 on left, team 1 on right)
                if team_centers[0][0] > team_centers[1][0]:  # If first cluster is on the right
                    position_assignments = [1 - label for label in position_assignments]  # Flip labels
                    team0_center, team1_center = team1_center, team0_center
                
                # 2. Add color-based features if available
                if hasattr(self, 'last_frame') and self.last_frame is not None:
                    try:
                        # Extract color features from jersey regions
                        color_features = []
                        for player in players:
                            x, y, w, h = player.box
                            # Extract jersey region (upper half of player box)
                            jersey_y = int(y + h*0.2)
                            jersey_h = int(h*0.3)
                            
                            if jersey_y >= 0 and jersey_y + jersey_h < self.last_frame.shape[0] and \
                               x >= 0 and x + w < self.last_frame.shape[1]:
                                jersey_region = self.last_frame[jersey_y:jersey_y+jersey_h, int(x):int(x+w)]
                                
                                # Convert to HSV for better color discrimination
                                if jersey_region.size > 0:
                                    hsv_region = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
                                    # Get average hue and saturation as color features
                                    h_avg = np.mean(hsv_region[:,:,0])
                                    s_avg = np.mean(hsv_region[:,:,1])
                                    color_features.append([h_avg, s_avg*2])  # Weight saturation more
                                else:
                                    color_features.append([0, 0])  # Fallback
                            else:
                                color_features.append([0, 0])  # Fallback for out-of-bounds regions
                        
                        # Cluster based on color features
                        if len(color_features) > 1:
                            color_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                            color_features = np.array(color_features)
                            color_kmeans.fit(color_features)
                            color_assignments = color_kmeans.labels_.tolist()
                    except Exception as e:
                        print(f"Error in color-based team assignment: {str(e)}")
                
                # 3. Add temporal consistency using historical assignments
                if self.player_positions_history and len(self.player_positions_history) > 5:
                    try:
                        prev_positions = self.player_positions_history[-1]
                        history_assignments = [0] * len(players)
                        
                        # Match current players with previous frame players
                        for i, player in enumerate(players):
                            # Find closest player in previous frame
                            best_match = -1
                            min_dist = float('inf')
                            for j, prev_pos in enumerate(prev_positions):
                                if j < len(self.team_history[-1]):
                                    dist = np.sqrt((player.position[0] - prev_pos[0])**2 + 
                                                  (player.position[1] - prev_pos[1])**2)
                                    if dist < min_dist and dist < 50:  # Maximum distance threshold
                                        min_dist = dist
                                        best_match = j
                            
                            # If matched, use previous team assignment
                            if best_match >= 0 and best_match < len(self.team_history[-1]):
                                history_assignments[i] = self.team_history[-1][best_match]
                    except Exception as e:
                        print(f"Error in temporal team assignment: {str(e)}")
                
                # 4. Combine all assignments using weighted voting
                final_assignments = [0] * len(players)
                
                # Determine weights based on confidence in each method
                position_weight = 1.0  # Base weight
                color_weight = 0.8 if color_assignments else 0.0
                history_weight = 1.5 if history_assignments else 0.0
                
                # Calculate total weight
                total_weight = position_weight + color_weight + history_weight
                
                # Combine weighted votes
                for i in range(len(players)):
                    weighted_vote = 0
                    
                    # Add position-based vote
                    weighted_vote += position_assignments[i] * position_weight
                    
                    # Add color-based vote if available
                    if color_assignments and i < len(color_assignments):
                        weighted_vote += color_assignments[i] * color_weight
                    
                    # Add history-based vote if available
                    if history_assignments and i < len(history_assignments):
                        weighted_vote += history_assignments[i] * history_weight
                    
                    # Final assignment is 1 if weighted vote > 50% of total weight
                    final_assignments[i] = 1 if weighted_vote > (total_weight / 2) else 0
                
                # Store final assignments in history
                self.team_history.append(final_assignments.copy())
                
            else:
                # Not enough players for advanced clustering, fall back to simple KMeans
                self.kmeans.fit(positions)
                final_assignments = self.kmeans.labels_.tolist()
                
                # Ensure consistent team numbering (0 for left team, 1 for right team)
                team_centers = self.kmeans.cluster_centers_
                if team_centers[0][0] > team_centers[1][0]:  # If first cluster is on the right
                    final_assignments = [1 - label for label in final_assignments]  # Flip labels
                
                # Store for future reference
                self.team_history.append(final_assignments.copy())
            
            # Store positions for next frame
            self.player_positions_history.append([p.position for p in players])
                
            return final_assignments
            
        except Exception as e:
            print(f"Error in team assignment: {str(e)}")
            # In case of error, use previous assignments if available
            if self.team_history and len(self.team_history) > 0 and len(self.team_history[-1]) == len(players):
                return self.team_history[-1]
            
            # Otherwise fall back to default assignment
            return [0] * len(players)
        
    def determine_attack_direction(self, team_positions: Dict[int, List[Tuple[int, int]]]) -> Direction:
        """Determine attack direction based on team positions.
        
        Args:
            team_positions: Dictionary mapping team index to list of player positions
            
        Returns:
            Direction enum indicating attack direction
        """
        if not team_positions[0] or not team_positions[1]:
            return self.direction_cache or Direction.LEFT
            
        try:
            # Calculate average x-position for each team safely
            team0_x = [float(pos[0]) for pos in team_positions[0]]  # Convert to float
            team1_x = [float(pos[0]) for pos in team_positions[1]]  # Convert to float
            
            if team0_x and team1_x:  # Check if lists are not empty
                team0_avg_x = sum(team0_x) / len(team0_x)
                team1_avg_x = sum(team1_x) / len(team1_x)
                
                # Calculate confidence in direction
                total_players = len(team_positions[0]) + len(team_positions[1])
                confidence = abs(team0_avg_x - team1_avg_x) / total_players
                
                # Only update direction if confidence is higher than current
                if confidence > self.direction_confidence:
                    self.direction_confidence = confidence
                    self.direction_cache = Direction.RIGHT if team0_avg_x < team1_avg_x else Direction.LEFT
        except Exception as e:
            print(f"Error determining attack direction: {str(e)}")
            
        return self.direction_cache or Direction.LEFT
        
    def detect_offside(self, frame: np.ndarray, boxes: List[List[int]], 
                      team_assignments: List[int], keypoints: List[np.ndarray] = None) -> Tuple[bool, np.ndarray]:
        """Detect offside situation in the current frame with enhanced detection.
        
        Args:
            frame: Input frame
            boxes: List of player bounding boxes
            team_assignments: List of team assignments (0 or 1)
            keypoints: Optional list of pose keypoints for each player
            
        Returns:
            Tuple of (is_offside, annotated_frame)
        """
        try:
            # Store the frame for reference
            self.last_frame = frame.copy()
            
            if not boxes or not team_assignments or len(boxes) != len(team_assignments):
                return False, frame
                
            # Convert all box coordinates to float
            boxes = [[float(x) for x in box] for box in boxes]
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Analyze field for visual cues and perspective
            field_lines, self.field_corners = self.field_detector.detect_field(frame)
            
            # Calculate vanishing point if possible
            if field_lines and len(field_lines) >= 2:
                self._calculate_vanishing_point(field_lines)
            
            # Process player detections to get Player objects with advanced positions
            players = self._get_players(boxes, team_assignments, keypoints)
            
            # Calculate player velocities from tracking
            velocities = []
            for player in players:
                if player.track_id is not None:
                    vel = self.player_tracker.get_velocity(player.track_id)
                    velocities.append(vel if vel is not None else [0, 0])
                else:
                    velocities.append([0, 0])
                    
            # Separate players by team
            team_positions = {0: [], 1: []}
            for player, team in zip(players, team_assignments):
                team_positions[team].append(player.position)
            
            # Determine attack direction
            direction = self.determine_attack_direction(team_positions)
            
            # Create team lists
            team0 = [p for i, p in enumerate(players) if team_assignments[i] == 0]
            team1 = [p for i, p in enumerate(players) if team_assignments[i] == 1]
            
            # Determine defending and attacking teams based on direction
            if direction == Direction.LEFT:  # Team 0 attacking left
                defending_team = team1
                attacking_team = team0
            else:  # Team 0 attacking right
                defending_team = team0
                attacking_team = team1
                
            # Check for offside condition
            is_offside = False
            if defending_team and attacking_team:
                # Find the last defender (second to last including GK)
                last_defender, second_defender = self._find_last_defenders(defending_team, direction)
                
                # Find the forward-most attacker
                forward_attacker = self._find_forward_attacker(attacking_team, direction)
                
                if last_defender and forward_attacker:
                    # Calculate perpendicular line through last defender position
                    if self.vanishing_point and self.perspective_confident:
                        # Use perspective-aware method
                        defender_line = self._calculate_line_params(last_defender.position, self.vanishing_point)
                        attacker_line = self._calculate_line_params(forward_attacker.advanced_position, self.vanishing_point)
                        
                        # Check if attacker is ahead of the defender line
                        is_offside = self._check_offside_enhanced(
                            last_defender, forward_attacker, defender_line, attacker_line, direction, velocities
                        )
                    else:
                        # Fall back to simple 2D comparison
                        defender_x = last_defender.position[0]
                        attacker_x = forward_attacker.advanced_position[0]
                        
                        if direction == Direction.LEFT:
                            is_offside = attacker_x < defender_x
                        else:
                            is_offside = attacker_x > defender_x
                    
                    # Draw the offside visualization
                    self._draw_offside_visualization(frame, last_defender, forward_attacker, is_offside, direction)
                    
                    # Add offside line and annotation
                    if is_offside:
                        # Draw the offside line with 3D perspective if possible
                        if self.perspective_confident and self.vanishing_point:
                            defender_line = self._calculate_line_params(last_defender.position, self.vanishing_point)
                            self._draw_optimized_reference_line(frame, defender_line, self.offside_line_color)
                        else:
                            # Fall back to vertical line
                            offside_line_x = int(last_defender.position[0])
                            cv2.line(frame, 
                                    (offside_line_x, 0),
                                    (offside_line_x, frame_height),
                                    self.offside_line_color,
                                    self.line_thickness)
                        
                        # Add "OFFSIDE" text with animation effect
                        text_color = (0, 0, 255)  # Red
                        text_size = cv2.getTextSize("OFFSIDE", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                        
                        # Create background for text
                        bg_margin = 10
                        cv2.rectangle(frame, 
                                     (frame_width - text_size[0] - bg_margin*2, 20), 
                                     (frame_width - bg_margin, 20 + text_size[1] + bg_margin*2),
                                     (0, 0, 0), -1)
                        
                        # Draw animated text if new offside detection
                        current_time = time.time()
                        if current_time - self.last_offside_time > self.offside_cooldown:
                            self.last_offside_time = current_time
                            
                        # Make text blink based on time
                        if (current_time - self.last_offside_time) % 1.0 < 0.5:
                            text_color = (0, 0, 255)  # Red
                        else:
                            text_color = (255, 255, 255)  # White
                        
                        cv2.putText(frame, "OFFSIDE",
                                   (frame_width - text_size[0] - bg_margin, 20 + text_size[1]),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   1.5, text_color, 3)
            
            return is_offside, frame
            
        except Exception as e:
            print(f"Error in offside detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, frame
            
    def _calculate_vanishing_point(self, lines):
        """Calculate the vanishing point from field lines."""
        try:
            # We need at least 2 lines to calculate vanishing point
            if len(lines) < 2:
                return
            
            # Extract line parameters from field lines
            params = []
            for line in lines:
                # Check if the line is in the expected format
                if isinstance(line, np.ndarray) and line.shape[0] >= 1 and len(line[0]) >= 2:
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    # Convert to y = mx + b format
                    if abs(b) > 0.01:  # Avoid division by zero
                        m = -a / b
                        c = y0 - m * x0
                        params.append((m, c))
                # Handle the case where lines are directly returned as [x1,y1,x2,y2] format
                elif isinstance(line, (list, tuple)) and len(line) >= 4:
                    x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
                    # Check if the line is not a point
                    if x2 != x1:
                        m = (y2 - y1) / (x2 - x1)
                        c = y1 - m * x1
                        params.append((m, c))
                    else:
                        # Vertical line has infinite slope
                        params.append((float('inf'), x1, 90.0))
            
            # Calculate intersections of all line pairs
            intersections = []
            for i in range(len(params)):
                for j in range(i+1, len(params)):
                    try:
                        # Skip if either is not a valid format
                        if len(params[i]) < 2 or len(params[j]) < 2:
                            continue
                            
                        # Handle different parameter formats
                        if len(params[i]) == 3:  # Vertical line format
                            m1, x1, _ = params[i]
                            if len(params[j]) == 3:  # Two vertical lines don't intersect
                                continue
                            else:
                                m2, c2 = params[j]
                                x = x1
                                y = m2 * x + c2
                        elif len(params[j]) == 3:  # Vertical line format
                            m2, x2, _ = params[j]
                            m1, c1 = params[i]
                            x = x2
                            y = m1 * x + c1
                        else:
                            m1, c1 = params[i]
                            m2, c2 = params[j]
                            
                            # Skip parallel lines
                            if abs(m1 - m2) < 0.01:
                                continue
                                
                            # Calculate intersection
                            x = (c2 - c1) / (m1 - m2)
                            y = m1 * x + c1
                        
                        # Check if intersection is reasonable
                        frame_height, frame_width = self.last_frame.shape[:2]
                        expanded_frame = (-frame_width*2, -frame_height*2, frame_width*3, frame_height*3)
                        if (expanded_frame[0] < x < expanded_frame[2] and 
                            expanded_frame[1] < y < expanded_frame[3]):
                            intersections.append((x, y))
                    except Exception as e:
                        print(f"Error calculating intersection: {str(e)}")
                        continue
            
            # Calculate median position if we have enough intersections
            if len(intersections) >= 3:
                x_vals = [p[0] for p in intersections]
                y_vals = [p[1] for p in intersections]
                
                # Use median to be robust to outliers
                median_x = np.median(x_vals)
                median_y = np.median(y_vals)
                
                self.vanishing_point = (median_x, median_y)
                self.perspective_confident = True
                
                # Calculate approximate pixels per meter
                if self.field_corners and len(self.field_corners) == 4:
                    # Estimate field width in pixels
                    field_width_px = max(corner[0] for corner in self.field_corners) - min(corner[0] for corner in self.field_corners)
                    self.pixels_per_meter = field_width_px / self.field_width_estimate
        except Exception as e:
            print(f"Error calculating vanishing point: {str(e)}")
            self.perspective_confident = False
        
    def _get_players(self, boxes: List[List[int]], team_assignments: List[int],
                    keypoints: List[np.ndarray] = None, 
                    track_ids: List[int] = None) -> List[Player]:
        """Convert detection data to Player objects with enhanced role classification."""
        players = []
        
        # Define frame dimensions for calculations
        frame_width = self.last_frame.shape[1] if hasattr(self, 'last_frame') else 1920
        frame_height = self.last_frame.shape[0] if hasattr(self, 'last_frame') else 1080
        
        # Define key body parts indices for advanced position calculation
        # COCO keypoints: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
        #                 left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
        #                 left_knee, right_knee, left_ankle, right_ankle]
        HEAD_PARTS = [0, 1, 2, 3, 4]  # nose, eyes, ears
        FOOT_PARTS = [15, 16]  # ankles
        ARM_PARTS = [9, 10]  # wrists
        
        for i, (box, team) in enumerate(zip(boxes, team_assignments)):
            try:
                x, y, w, h = box
                center = (float(x + w/2), float(y + h))  # Bottom center for better position
                kpts = keypoints[i] if keypoints and i < len(keypoints) else None
                track_id = track_ids[i] if track_ids and i < len(track_ids) else None
                
                # Calculate GK probability based on multiple factors
                role_conf = 0.0
                is_gk = False
                
                # Factor 1: Check position relative to goal line (stronger weight)
                goal_line_proximity = 0.0
                if x < frame_width * 0.1:  # Near left goal
                    goal_line_proximity = 1.0 - (x / (frame_width * 0.1))
                    is_gk = True if goal_line_proximity > 0.7 else False
                elif x > frame_width * 0.9:  # Near right goal
                    goal_line_proximity = (x - frame_width * 0.9) / (frame_width * 0.1)
                    is_gk = True if goal_line_proximity > 0.7 else False
                
                role_conf += min(goal_line_proximity * 0.6, 0.6)  # Up to 60% weight
                
                # Factor 2: Check historical position if tracked (more confidence)
                historical_gk_conf = 0.0
                if track_id is not None:
                    track_history = self.player_tracker.get_track_history(track_id)
                    if track_history and len(track_history) > 10:
                        avg_x = np.mean([pos[0] for pos in track_history])
                        if avg_x < frame_width * 0.15 or avg_x > frame_width * 0.85:
                            historical_gk_conf = 0.4
                            is_gk = is_gk or (historical_gk_conf > 0.3)
                
                role_conf += historical_gk_conf
                
                # Factor 3: Check position relative to teammates
                team_players = [j for j, t in enumerate(team_assignments) if t == team and j != i]
                if team_players:
                    team_x_positions = [boxes[j][0] + boxes[j][2]/2 for j in team_players]
                    team_avg_x = np.mean(team_x_positions) if team_x_positions else x + w/2
                    
                    # If player is significantly more extreme than teammates
                    player_x = x + w/2
                    is_extreme_position = False
                    
                    if team_avg_x < frame_width/2:  # Team average on left side
                        if player_x < team_avg_x * 0.8:  # Player much further left
                            is_extreme_position = True
                    else:  # Team average on right side
                        if player_x > team_avg_x * 1.2:  # Player much further right
                            is_extreme_position = True
                            
                    if is_extreme_position:
                        role_conf += 0.2
                        is_gk = is_gk or (role_conf > 0.7)
                
                # Calculate advanced position (most forward body part) with better keypoint analysis
                advanced_pos = center
                if kpts is not None:
                    # Filter only keypoints with good confidence
                    confident_keypoints = kpts[kpts[:, 2] > 0.35]  # Increased threshold for accuracy
                    
                    if len(confident_keypoints) > 0:
                        # Try to use feet and hands as most advanced body parts for attackers
                        forward_parts = []
                        
                        # Add foot keypoints if detected with high confidence
                        for foot_idx in FOOT_PARTS:
                            if foot_idx < len(kpts) and kpts[foot_idx, 2] > 0.5:  # High confidence feet
                                forward_parts.append((kpts[foot_idx, 0], kpts[foot_idx, 1]))
                                
                        # Add arm/hand keypoints with slightly lower confidence requirement
                        for arm_idx in ARM_PARTS:
                            if arm_idx < len(kpts) and kpts[arm_idx, 2] > 0.4:  # Medium confidence arms
                                forward_parts.append((kpts[arm_idx, 0], kpts[arm_idx, 1]))
                                
                        # If we have forward parts, use the most extreme one based on team position
                        if forward_parts:
                            if team_avg_x < frame_width/2:  # Team on left side, attacking right
                                # Most right-side body part
                                advanced_pos = max(forward_parts, key=lambda p: p[0])
                            else:  # Team on right side, attacking left
                                # Most left-side body part
                                advanced_pos = min(forward_parts, key=lambda p: p[0])
                        else:
                            # Fall back to simple box position if no good body parts detected
                            if team_avg_x < frame_width/2:  # Team on left, attacking right
                                advanced_pos = (x + w, y + h/2)  # Right edge of box
                            else:  # Team on right, attacking left
                                advanced_pos = (x, y + h/2)  # Left edge of box
                    
                # Create Player object with enhanced attributes
                player = Player(
                    position=center,
                    box=box,
                    team=team,
                    keypoints=kpts,
                    track_id=track_id,
                    role_confidence=min(role_conf, 1.0),  # Cap at 1.0
                    is_goalkeeper=is_gk,
                    advanced_position=advanced_pos
                )
                players.append(player)
                
            except Exception as e:
                print(f"Error creating player {i}: {str(e)}")
                # Create minimal player object on error
                player = Player(
                    position=(float(box[0] + box[2]/2), float(box[1] + box[3])),
                    box=box,
                    team=team
                )
                players.append(player)
                
        return players
        
    def _find_forward_attacker(self, team: List[Player], 
                             direction: Direction) -> Optional[Player]:
        """Find the most forward attacking player, considering pose and movement."""
        if not team:
            return None
            
        # Use advanced position that accounts for body parts
        if direction == Direction.LEFT:
            return min(team, key=lambda p: p.advanced_position[0])
        return max(team, key=lambda p: p.advanced_position[0])
        
    def _check_offside_enhanced(self, defender: Player, attacker: Player,
                              defender_line: Tuple[float, float, float],
                              attacker_line: Tuple[float, float, float],
                              direction: Direction,
                              velocities: List[List[float]]) -> bool:
        """Enhanced offside check using pose and movement information."""
        # Basic position check with advanced position
        is_offside = self._check_offside_position_optimized(
            defender.position, attacker.advanced_position,
            defender_line, attacker_line, direction
        )
        
        if not is_offside:
            return False
        
        # Check if attacker is moving away from goal (passive offside exception)
        if attacker.track_id is not None and attacker.track_id < len(velocities):
            velocity = velocities[attacker.track_id]
            # If attacker is moving away from goal at significant speed, not offside
            if (direction == Direction.LEFT and velocity[0] > 3) or \
               (direction == Direction.RIGHT and velocity[0] < -3):
                return False
        
        # Use pose information if available for even more precise check
        if attacker.keypoints is not None and defender.keypoints is not None:
            # Get relevant body parts for offside check
            # We'll check shoulders, hips, knees, and feet
            attacker_parts = {
                'shoulders': attacker.keypoints[[5, 6]],  # Left and right shoulder
                'hips': attacker.keypoints[[11, 12]],     # Left and right hip
                'knees': attacker.keypoints[[13, 14]],    # Left and right knee
                'feet': attacker.keypoints[[15, 16]]      # Left and right foot/ankle
            }
            
            defender_parts = {
                'shoulders': defender.keypoints[[5, 6]],
                'hips': defender.keypoints[[11, 12]],
                'knees': defender.keypoints[[13, 14]],
                'feet': defender.keypoints[[15, 16]]
            }
            
            # Initialize counters for confident detections
            total_valid_comparisons = 0
            offside_comparisons = 0
            
            # Weight factors for different body parts (feet and shoulders are most important)
            weights = {
                'shoulders': 0.3,
                'hips': 0.2,
                'knees': 0.2,
                'feet': 0.3
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for part_name in attacker_parts:
                att_parts = attacker_parts[part_name]
                def_parts = defender_parts[part_name]
                weight = weights[part_name]
                
                valid_comparisons = 0
                offside_count = 0
                
                for att_part, def_part in zip(att_parts, def_parts):
                    # Only compare if both keypoints are confident
                    if att_part[2] > 0.5 and def_part[2] > 0.5:
                        valid_comparisons += 1
                        if direction == Direction.LEFT:
                            if att_part[0] < def_part[0]:
                                offside_count += 1
                        else:
                            if att_part[0] > def_part[0]:
                                offside_count += 1
                            
                if valid_comparisons > 0:
                    part_score = offside_count / valid_comparisons
                    weighted_score += part_score * weight
                    total_weight += weight
                    
            # If we have valid comparisons, use weighted average
            if total_weight > 0:
                final_score = weighted_score / total_weight
                return final_score > 0.6  # Slightly higher threshold for confidence
                
        # Fall back to simple position check if no valid keypoints
        return True
        
    def _draw_offside_visualization(self, frame, last_defender, forward_attacker, is_offside, direction):
        """Draw enhanced offside visualization with 3D perspective."""
        # Draw the offside line first (under players)
        h, w = frame.shape[:2]
        line_color = (0, 0, 255) if is_offside else (0, 255, 0)  # Red if offside, green if not
        
        # Draw 3D perspective-aware line if possible
        if self.perspective_confident and self.vanishing_point:
            defender_line = self._calculate_line_params(last_defender.position, self.vanishing_point)
            self._draw_optimized_reference_line(frame, defender_line, line_color)
            
            # Add distance indicator if we know pixels per meter
            if self.pixels_per_meter and self.pixels_per_meter > 0:
                if direction == Direction.LEFT:
                    distance_px = last_defender.position[0] - forward_attacker.advanced_position[0] 
                else:
                    distance_px = forward_attacker.advanced_position[0] - last_defender.position[0]
                    
                distance_meters = abs(distance_px / self.pixels_per_meter)
                
                # Only show if meaningful distance
                if distance_meters > 0.1:
                    # Draw distance text at midpoint between players
                    mid_x = (last_defender.position[0] + forward_attacker.advanced_position[0]) / 2
                    mid_y = (last_defender.position[1] + forward_attacker.advanced_position[1]) / 2 - 30
                    
                    distance_text = f"{distance_meters:.2f}m"
                    cv2.putText(frame, distance_text,
                              (int(mid_x), int(mid_y)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                              (255, 255, 255), 2)
        else:
            # Fall back to simple vertical line
            line_x = int(last_defender.position[0])
            cv2.line(frame, (line_x, 0), (line_x, h), line_color, 2)
            
        # Highlight forward attacker position
        marker_color = (0, 0, 255) if is_offside else (0, 255, 0)  # Red if offside, green if not
        cv2.circle(frame, 
                 (int(forward_attacker.advanced_position[0]), int(forward_attacker.advanced_position[1])),
                 8, marker_color, -1)
        cv2.circle(frame, 
                 (int(forward_attacker.advanced_position[0]), int(forward_attacker.advanced_position[1])),
                 10, (255, 255, 255), 2)
        
        # Draw line connecting advanced position to player center
        if forward_attacker.advanced_position != forward_attacker.position:
            cv2.line(frame,
                    (int(forward_attacker.position[0]), int(forward_attacker.position[1])),
                    (int(forward_attacker.advanced_position[0]), int(forward_attacker.advanced_position[1])),
                    marker_color, 2)
                
    def _draw_pose(self, frame: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int]) -> None:
        """Draw pose keypoints and connections."""
        connections = [
            (5, 7), (7, 9),   # Left arm
            (6, 8), (8, 10),  # Right arm
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
            (5, 6), (11, 12)     # Shoulders and hips
        ]
        
        # Draw keypoints
        for kpt in keypoints:
            if kpt[2] > 0.5:  # Confidence threshold
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, color, -1)
        
        # Draw connections
        for p1_idx, p2_idx in connections:
            if (keypoints[p1_idx][2] > 0.5 and keypoints[p2_idx][2] > 0.5):
                pt1 = (int(keypoints[p1_idx][0]), int(keypoints[p1_idx][1]))
                pt2 = (int(keypoints[p2_idx][0]), int(keypoints[p2_idx][1]))
                cv2.line(frame, pt1, pt2, color, 2)
                
    def _find_last_defenders(self, team: List[Player], 
                           direction: Direction) -> Tuple[Optional[Player], Optional[Player]]:
        """Find the last and second last defenders based on direction and role."""
        if not team:
            return None, None
            
        # First check for goalkeeper with high confidence
        goalkeeper = None
        for player in team:
            if player.is_goalkeeper and player.role_confidence > 0.6:  # Increased confidence threshold
                goalkeeper = player
                break
            
        # Sort defenders by their x position
        sorted_defenders = sorted(team, key=lambda p: p.position[0], 
                                reverse=(direction == Direction.RIGHT))
        
        # Remove goalkeeper from sorted list if found
        if goalkeeper:
            sorted_defenders = [p for p in sorted_defenders if p != goalkeeper]
            
        if goalkeeper:
            if len(sorted_defenders) >= 1:
                # Return goalkeeper and last outfield defender
                return goalkeeper, sorted_defenders[0]
            return goalkeeper, None
        elif len(sorted_defenders) >= 2:
            return sorted_defenders[0], sorted_defenders[1]
        elif len(sorted_defenders) == 1:
            return sorted_defenders[0], None
        return None, None
        
    def _is_behind(self, pos1: Tuple[int, int], pos2: Tuple[int, int], 
                  direction: Direction) -> bool:
        """Check if pos1 is behind pos2 based on direction."""
        if direction == Direction.LEFT:
            return pos1[0] < pos2[0]
        return pos1[0] > pos2[0]
            
    def _calculate_line_params(self, point: Tuple[int, int], 
                             vanishing_point: Tuple[int, int]) -> Tuple[float, float, float]:
        """Calculate line parameters (slope, intercept, angle).
        
        Args:
            point: Starting point of line
            vanishing_point: Vanishing point coordinates
            
        Returns:
            Tuple of (slope, intercept, angle)
        """
        if point[0] == vanishing_point[0]:
            return (float('inf'), point[0], 90.0)
            
        slope = (point[1] - vanishing_point[1]) / (point[0] - vanishing_point[0])
        intercept = point[1] - slope * point[0]
        angle = np.arctan2(point[1] - vanishing_point[1], point[0] - vanishing_point[0])
        
        return (slope, intercept, angle)
        
    def _draw_optimized_reference_line(self, frame: np.ndarray, 
                                     line_params: Tuple[float, float, float],
                                     color: Tuple[int, int, int]) -> None:
        """Draw reference line more efficiently using pre-calculated parameters.
        
        Args:
            frame: Frame to draw on
            line_params: Tuple of (slope, intercept, angle)
            color: Line color
        """
        slope, intercept, angle = line_params
        h, w = frame.shape[:2]
        
        if slope == float('inf'):
            x = int(intercept)
            # Draw vertical line with dashed pattern
            dash_length = 20
            for y in range(0, h, dash_length * 2):
                y2 = min(y + dash_length, h)
                cv2.line(frame, (x, y), (x, y2), color, 2, cv2.LINE_AA)
            return
            
        # Calculate line endpoints
        if abs(slope) > 1:
            # Line is more vertical, use y coordinates
            y1, y2 = 0, h
            x1 = int((y1 - intercept) / slope) if slope != 0 else 0
            x2 = int((y2 - intercept) / slope) if slope != 0 else w
        else:
            # Line is more horizontal, use x coordinates
            x1, x2 = 0, w
            y1 = int(slope * x1 + intercept)
            y2 = int(slope * x2 + intercept)
            
        # Clip line endpoints to frame boundaries
        x1 = max(0, min(w-1, x1))
        x2 = max(0, min(w-1, x2))
        y1 = max(0, min(h-1, y1))
        y2 = max(0, min(h-1, y2))
        
        # Draw dashed line with anti-aliasing
        dash_length = 20
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < 1:  # Avoid division by zero
            return
            
        dx = dx / dist * dash_length
        dy = dy / dist * dash_length
        
        # Draw dashed segments
        curr_x, curr_y = x1, y1
        while (curr_x-x1)*dx + (curr_y-y1)*dy < dist*dash_length:
            next_x = min(x2, curr_x + dx)
            next_y = min(y2, curr_y + dy)
            cv2.line(frame, (int(curr_x), int(curr_y)), 
                    (int(next_x), int(next_y)), color, 2, cv2.LINE_AA)
            curr_x = curr_x + 2*dx
            curr_y = curr_y + 2*dy
            
        # Add arrow at the end of the line
        arrow_length = 20
        angle_rad = np.arctan2(y2-y1, x2-x1)
        arrow_angle = np.pi/6  # 30 degrees
        
        # Calculate arrow points
        p1_x = x2 - arrow_length * np.cos(angle_rad + arrow_angle)
        p1_y = y2 - arrow_length * np.sin(angle_rad + arrow_angle)
        p2_x = x2 - arrow_length * np.cos(angle_rad - arrow_angle)
        p2_y = y2 - arrow_length * np.sin(angle_rad - arrow_angle)
        
        # Draw arrow
        cv2.line(frame, (int(x2), int(y2)), (int(p1_x), int(p1_y)), color, 2, cv2.LINE_AA)
        cv2.line(frame, (int(x2), int(y2)), (int(p2_x), int(p2_y)), color, 2, cv2.LINE_AA)
        
    def _check_offside_position_optimized(self, defender_pos: Tuple[int, int],
                                        attacker_pos: Tuple[int, int],
                                        defender_params: Tuple[float, float, float],
                                        attacker_params: Tuple[float, float, float],
                                        direction: Direction) -> bool:
        """Check if attacker is in offside position using pre-calculated parameters."""
        defender_slope = defender_params[0]
        defender_intercept = defender_params[1]
        
        if direction == Direction.LEFT:
            if defender_slope == float('inf'):
                return attacker_pos[0] < defender_pos[0]
            y_at_attacker = defender_slope * attacker_pos[0] + defender_intercept
            return attacker_pos[1] < y_at_attacker
        else:
            if defender_slope == float('inf'):
                return attacker_pos[0] > defender_pos[0]
            y_at_attacker = defender_slope * attacker_pos[0] + defender_intercept
            return attacker_pos[1] > y_at_attacker
            
    def _find_last_player(self, team: List[Player], 
                         direction: Direction) -> Optional[Player]:
        """Find the last player of a team based on direction."""
        if not team:
            return None
            
        if direction == Direction.LEFT:
            return min(team, key=lambda p: p.position[0])
        return max(team, key=lambda p: p.position[0]) 