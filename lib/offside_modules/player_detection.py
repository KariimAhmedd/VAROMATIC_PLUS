from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Tuple, Optional
import traceback

class PlayerDetector:
    def __init__(self, confidence_threshold: float = 0.25):  # Increased from 0.15 to 0.25
        """Initialize the player detector with enhanced parameters."""
        self.model = YOLO('yolov8x-pose.pt')  # Using the largest model for best accuracy
        
        # Enable GPU acceleration explicitly
        if torch.backends.mps.is_available():  # For Apple Silicon (M1/M2)
            print("Using MPS (Metal Performance Shaders) acceleration for Apple Silicon")
            self.device = torch.device("mps")
        elif torch.cuda.is_available():  # For NVIDIA GPUs
            print("Using CUDA acceleration")
            self.device = torch.device("cuda")
        else:
            print("Using CPU for inference (no GPU acceleration available)")
            self.device = torch.device("cpu")
            
        # Try to move model to the chosen device
        try:
            self.model.to(self.device)
        except Exception as e:
            print(f"Failed to move model to {self.device}: {e}")
            print("Falling back to default device")
            
        self.confidence_threshold = confidence_threshold
        
        # More strict filtering parameters for better player detection
        self.min_height = 20  # Increased minimum height to filter out distant spectators
        self.max_height = 600  # Reduced maximum height to filter out very close spectators
        self.min_aspect_ratio = 0.15  # More strict ratio for player detection
        self.max_aspect_ratio = 1.5  # More strict ratio for player detection
        
        # Multi-angle detection parameters
        self.pose_confidence_threshold = 0.15  # Increased threshold for better pose detection
        self.min_visible_keypoints = 4  # Increased required keypoints for better accuracy
        
        # Field position filtering
        self.field_margin = 0.1  # 10% margin from field edges
        self.field_center_x = None
        self.field_center_y = None
        self.field_width = None
        self.field_height = None
        
        # Temporal smoothing
        self.last_detections = []
        self.smoothing_frames = 3
        self.last_frame_shape = None
        
        # Role classification parameters
        self.player_roles = {
            'player': {'color': (0, 255, 0)},    # Green
            'goalkeeper': {'color': (0, 165, 255)},  # Orange
            'referee': {'color': (255, 255, 255)}    # White
        }
        
        # Previous frame info for tracking
        self.prev_boxes = []
        self.prev_roles = []
        
        # Detection enhancement parameters
        self.detection_history = []  # For temporal consistency
        self.detection_history_max = 5
        self.iou_threshold = 0.3  # Lower IOU threshold for better duplicate detection
        
        print(f"Player detector initialized with confidence_threshold={confidence_threshold}, "
              f"min_height={self.min_height}, max_height={self.max_height}, "
              f"min_aspect_ratio={self.min_aspect_ratio}, max_aspect_ratio={self.max_aspect_ratio}")
        
    def _is_in_field(self, box: List[float], frame_shape: Tuple[int, int]) -> bool:
        """Check if detection is within the field boundaries."""
        if self.field_center_x is None:
            # Initialize field boundaries if not set
            h, w = frame_shape
            self.field_center_x = w // 2
            self.field_center_y = h // 2
            self.field_width = w
            self.field_height = h
        
        x, y, w, h = box
        center_x = x + w/2
        center_y = y + h/2
        
        # Calculate field boundaries with margin
        margin_x = self.field_width * self.field_margin
        margin_y = self.field_height * self.field_margin
        
        # Check if detection is within field boundaries
        return (margin_x <= center_x <= self.field_width - margin_x and
                margin_y <= center_y <= self.field_height - margin_y)

    def filter_detections(self, boxes: List[List[float]], keypoints: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[List[List[float]], List[int]]:
        """Enhanced filtering of player detections with spectator filtering."""
        kept_boxes = []
        kept_indices = []
        
        if not boxes:
            return [], []
            
        for i, box in enumerate(boxes):
            try:
                x, y, w, h = map(float, box)  # Ensure all values are float
                
                # Skip invalid boxes
                if w <= 0.0 or h <= 0.0:
                    continue
                    
                aspect_ratio = float(w) / float(h)
                
                # Strict size and ratio checks
                if float(h) < self.min_height or float(h) > self.max_height:
                    continue
                if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                    continue
                
                # Check if detection is within field boundaries
                if not self._is_in_field([x, y, w, h], frame_shape):
                    continue
                    
                # Improved pose-based filtering
                if keypoints is not None and i < len(keypoints):
                    kpts = keypoints[i]
                    visible_keypoints = np.sum(kpts[:, 2] > self.pose_confidence_threshold)
                    
                    # Accept detection if enough keypoints or back-facing
                    if visible_keypoints >= self.min_visible_keypoints or self._is_back_facing_player(kpts):
                        kept_boxes.append([float(x), float(y), float(w), float(h)])
                        kept_indices.append(i)
                        continue
                        
                # If no keypoints available or keypoint check failed, accept based on box criteria
                kept_boxes.append([float(x), float(y), float(w), float(h)])
                kept_indices.append(i)
                
            except Exception as e:
                print(f"Error filtering detection {i}: {str(e)}")
                continue
                
        return kept_boxes, kept_indices
        
    def _is_back_facing_player(self, keypoints: np.ndarray) -> bool:
        """Improved back-facing player detection with stricter criteria."""
        try:
            # Key points for back detection
            NOSE = 0
            L_SHOULDER, R_SHOULDER = 5, 6
            L_HIP, R_HIP = 11, 12
            
            # Check if back keypoints are more visible than front
            back_points = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
            back_confidences = [keypoints[i][2] for i in back_points if i < len(keypoints)]
            
            if not back_confidences:  # No valid back points
                return False
                
            back_confidence = np.mean(back_confidences)
            
            # Check if nose is less visible (indicating back view)
            nose_confidence = keypoints[NOSE][2] if NOSE < len(keypoints) else 0
            
            # Check shoulder width for back view
            if L_SHOULDER < len(keypoints) and R_SHOULDER < len(keypoints):
                shoulder_width = np.abs(keypoints[L_SHOULDER][0] - keypoints[R_SHOULDER][0])
                if shoulder_width > 20:  # Increased minimum shoulder width
                    return back_confidence > nose_confidence and back_confidence > self.pose_confidence_threshold
                    
            return back_confidence > self.pose_confidence_threshold * 1.2  # Increased threshold for back view
            
        except Exception as e:
            print(f"Error in back-facing detection: {str(e)}")
            return False
        
    def detect_players(self, frame: np.ndarray) -> Tuple[List[List[int]], List[int], List[np.ndarray], List[str]]:
        """Detect players in the frame with improved accuracy and filtering.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (all_boxes, kept_indices, keypoints, player_roles)
        """
        try:
            frame_shape = frame.shape[:2]
            
            # Optimize inference with fixed model settings
            results = self.model.predict(
                source=frame, 
                conf=self.confidence_threshold,
                classes=0,  # Only detect people
                verbose=False,
                device=self.device,
                augment=False,  # No augmentation for speed
                retina_masks=False
            )
            
            boxes = []
            keypoints = []
            
            if len(results) > 0 and hasattr(results[0], 'boxes'):
                # Get person detections
                for box in results[0].boxes.data:
                    x1, y1, x2, y2, conf, cls = box
                    if conf >= self.confidence_threshold:
                        # Convert to [x, y, w, h] format
                        w = x2 - x1
                        h = y2 - y1
                        boxes.append([float(x1), float(y1), float(w), float(h)])
                
                # Get keypoints if available
                if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                    for kpt in results[0].keypoints.data:
                        keypoints.append(kpt.cpu().numpy())
                
                # Check if we have valid detections
                if not boxes:
                    print("No valid detections found in frame")
                    return [], [], [], []
                
                try:
                    # Apply filtering with frame shape information
                    kept_boxes, kept_indices = self.filter_detections(boxes, keypoints, frame_shape)
                    
                    # Validate indices after filtering
                    valid_indices = [idx for idx in kept_indices if idx < len(boxes)]
                    if len(valid_indices) != len(kept_indices):
                        print(f"Warning: Some indices were out of range after filtering. Fixing...")
                        kept_indices = valid_indices
                        
                    if not kept_indices:
                        print("No detections remained after filtering")
                        return boxes, [], keypoints, []
                    
                    # Apply temporal smoothing if enabled
                    if self.last_frame_shape == frame_shape and self.last_detections:
                        try:
                            kept_boxes, kept_indices = self._smooth_detections(
                                kept_boxes, kept_indices, keypoints, self.last_detections
                            )
                            
                            # Check if any detections remain
                            if not kept_indices:
                                print("No detections remained after smoothing")
                                return boxes, [], keypoints, []
                                
                        except Exception as e:
                            print(f"Error during temporal smoothing: {str(e)}")
                            # Continue with unsmoothed detections
                    
                    # Apply non-maximum suppression to remove overlapping detections
                    if len(kept_boxes) > 1:
                        try:
                            # Use more permissive IoU threshold to catch more duplicates
                            nms_boxes, nms_indices = self._apply_nms(kept_boxes, self.iou_threshold)
                            # Map the NMS indices back to the original kept_indices
                            if nms_indices:
                                kept_indices = [kept_indices[i] for i in nms_indices if i < len(kept_indices)]
                                kept_boxes = nms_boxes
                            
                            # Check if any detections remain
                            if not kept_indices:
                                print("No detections remained after NMS")
                                return boxes, [], keypoints, []
                                
                        except Exception as e:
                            print(f"Error during NMS: {str(e)}")
                            # Continue with unsuppressed detections
                
                    # Store for next frame
                    self.last_detections = kept_boxes.copy()
                    self.last_frame_shape = frame_shape
                    
                    # Final validation of indices
                    for idx in kept_indices:
                        if idx >= len(boxes):
                            print(f"Warning: Index {idx} is out of range for boxes list of length {len(boxes)}")
                            kept_indices = [i for i in kept_indices if i < len(boxes)]
                            break
                    
                    # Determine player roles (placeholder - will be updated with team assignments)
                    player_roles = ['player'] * len(kept_indices)
                    
                    print(f"Drawing {len(kept_indices)} boxes from {len(boxes)} total boxes")
                    return boxes, kept_indices, keypoints, player_roles
                    
                except Exception as e:
                    print(f"Error during detection processing: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return [], [], [], []
            
        except Exception as e:
            print(f"Error in player detection: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Return empty results on failure
        return [], [], [], []
            
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
            
    def _preprocess_frame(self, frame: np.ndarray, high_quality: bool = False) -> np.ndarray:
        """Optimized preprocessing to maintain video quality while enhancing detection.
        
        Args:
            frame: Input frame
            high_quality: If True, use higher quality processing (potentially slower)
        """
        try:
            # For M1/M2 Macs, use minimal preprocessing to maintain quality
            if hasattr(self, 'device') and self.device.type == "mps":
                # Use subtle enhancements that preserve quality
                if high_quality:
                    # Make a deep copy to avoid modifying original
                    frame_copy = frame.copy()
                    
                    # For high quality, use CLAHE for better contrast while preserving details
                    lab = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    
                    # Apply gentle CLAHE to L channel
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    
                    # Merge channels back
                    enhanced_lab = cv2.merge([cl, a, b])
                    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                    
                    # Very mild contrast enhancement
                    alpha = 1.05  # Very slight contrast increase
                    beta = 3      # Minimal brightness boost
                    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
                    
                    return enhanced
                else:
                    # Just do a minimal contrast enhancement
                    alpha = 1.1  # Reduced contrast increase
                    beta = 5     # Reduced brightness boost
                    enhanced = cv2.convertScaleAbs(frame.copy(), alpha=alpha, beta=beta)
                    return enhanced
            
            # For other systems, use more targeted preprocessing based on quality setting
            # Make a copy to avoid in-place modifications
            frame_copy = frame.copy()
            
            # Convert to LAB color space for better color processing
            lab = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            if high_quality:
                # Apply precise CLAHE to L channel for better contrast with quality preservation
                clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8))  # More gentle
            else:
                # Standard CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                
            cl = clahe.apply(l)
            
            # Merge channels back
            enhanced_lab = cv2.merge([cl, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            if high_quality:
                # Use higher quality denoising
                enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 3, 7)
                
                # Apply very gentle contrast enhancement
                alpha = 1.08  # Minimal contrast
                beta = 2      # Minimal brightness
            else:
                # Use lighter denoising - less quality loss
                enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 3, 9)
                
                # Apply gentler contrast enhancement
                alpha = 1.15  # Reduced contrast
                beta = 3      # Reduced brightness
                
            enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
            
            return enhanced
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            # Return the original frame if preprocessing fails
            return frame.copy()
            
    def _smooth_detections(self, current_boxes: List[List[float]], kept_indices: List[int], 
                         keypoints: List[np.ndarray], prev_boxes: List[List[float]]) -> Tuple[List[List[float]], List[int]]:
        """Apply temporal smoothing to detected player boxes for more stable tracking.
        
        Args:
            current_boxes: Current frame's detection boxes
            kept_indices: Indices of valid detections
            keypoints: List of pose keypoints for each player
            prev_boxes: Previous frame's detection boxes
            
        Returns:
            Tuple of (smoothed_boxes, updated_indices)
        """
        if not current_boxes or not kept_indices or not prev_boxes:
            return current_boxes, kept_indices
            
        # Validate that all indices are within range
        valid_kept_indices = [idx for idx in kept_indices if idx < len(current_boxes)]
        if len(valid_kept_indices) != len(kept_indices):
            print(f"Warning in smooth_detections: Some indices were out of range. Original length: {len(kept_indices)}, Valid length: {len(valid_kept_indices)}")
            kept_indices = valid_kept_indices
            
        if not kept_indices:  # All indices were invalid
            return current_boxes, []
            
        smoothed_boxes = current_boxes.copy()
        updated_indices = kept_indices.copy()
        
        # Match current detections with previous ones
        for i, idx in enumerate(kept_indices):
            if idx >= len(current_boxes):
                continue
                
            current_box = current_boxes[idx]
            current_center = (current_box[0] + current_box[2]/2, current_box[1] + current_box[3]/2)
            
            # Find closest previous box
            closest_dist = float('inf')
            closest_box = None
            
            for prev_box in prev_boxes:
                prev_center = (prev_box[0] + prev_box[2]/2, prev_box[1] + prev_box[3]/2)
                dist = ((current_center[0] - prev_center[0])**2 + 
                       (current_center[1] - prev_center[1])**2)**0.5
                       
                if dist < closest_dist:
                    closest_dist = dist
                    closest_box = prev_box
            
            # If found a close previous box, apply smoothing
            if closest_box and closest_dist < 50:  # Threshold for matching
                # Apply temporal smoothing with 0.7 weight for current position, 0.3 for previous
                alpha = 0.7  # Weight for current box
                smoothed_box = [
                    alpha * current_box[0] + (1-alpha) * closest_box[0],
                    alpha * current_box[1] + (1-alpha) * closest_box[1],
                    alpha * current_box[2] + (1-alpha) * closest_box[2],
                    alpha * current_box[3] + (1-alpha) * closest_box[3]
                ]
                smoothed_boxes[idx] = smoothed_box
                
        return smoothed_boxes, updated_indices
        
    def draw_detections(self, frame, boxes, kept_indices, keypoints=None, team_colors=None, roles=None):
        """Draw detection boxes and poses with improved visibility and role differentiation."""
        try:
            frame_copy = frame.copy()
            
            if not boxes or not kept_indices:
                print("No boxes or indices to draw")
                return frame_copy
                
            print(f"Drawing {len(kept_indices)} boxes from {len(boxes)} total boxes")
            
            # Loop through the kept indices
            for i, idx in enumerate(kept_indices):
                if idx < 0 or idx >= len(boxes):
                    print(f"Invalid box index: {idx} (max: {len(boxes)-1})")
                    continue
                    
                box = boxes[idx]
                x, y, w, h = map(int, box)
                
                # Determine color based on role and team
                role = roles[i] if roles and i < len(roles) else 'player'
                
                if role == 'referee':
                    # Always use white for referees regardless of team
                    color = self.player_roles['referee']['color']  # White
                elif role == 'goalkeeper':
                    # Use bright goalkeeper color (orange)
                    # Slight variation based on team
                    gk_color = list(self.player_roles['goalkeeper']['color'])
                    if team_colors and i < len(team_colors):
                        # Add a hint of team color
                        for j in range(3):
                            gk_color[j] = min(255, max(0, int(gk_color[j] * 0.7 + team_colors[i][j] * 0.3)))
                    color = tuple(gk_color)
                else:
                    # Regular player - use team color
                    color = team_colors[i] if team_colors and i < len(team_colors) else (0, 255, 0)
                
                # Draw double border for better visibility
                cv2.rectangle(frame_copy, (x-1, y-1), (x+w+1, y+h+1), (255, 255, 255), 3)
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), color, 2)
                
                # Draw player role and number
                role_text = f"{role[0].upper()}#{idx+1}"  # First letter of role + number
                cv2.putText(frame_copy, role_text, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame_copy, role_text, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw pose keypoints if available
                kpt_idx = idx if keypoints and idx < len(keypoints) else None
                if kpt_idx is not None and keypoints[kpt_idx] is not None:
                    self.draw_pose(frame_copy, keypoints[kpt_idx], color)
            
            return frame_copy
            
        except Exception as e:
            print(f"Error drawing detections: {str(e)}")
            traceback.print_exc()
            return frame
            
    def draw_pose(self, frame, keypoints, color):
        """Draw pose keypoints and connections with improved visibility."""
        try:
            # Draw keypoints
            for kpt in keypoints:
                x, y = map(int, kpt[:2])
                conf = kpt[2]
                if conf > self.pose_confidence_threshold:
                    cv2.circle(frame, (x, y), 3, color, -1)
                    cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)
            
            # Draw connections (simplified for better performance)
            connections = [
                (5, 7), (7, 9),   # Right arm
                (6, 8), (8, 10),  # Left arm
                (11, 13), (13, 15),  # Right leg
                (12, 14), (14, 16),  # Left leg
                (5, 6), (11, 12)     # Shoulders and hips
            ]
            
            for p1, p2 in connections:
                if (keypoints[p1][2] > self.pose_confidence_threshold and 
                    keypoints[p2][2] > self.pose_confidence_threshold):
                    pt1 = tuple(map(int, keypoints[p1][:2]))
                    pt2 = tuple(map(int, keypoints[p2][:2]))
                    cv2.line(frame, pt1, pt2, color, 2)
                    
        except Exception as e:
            print(f"Error drawing pose: {str(e)}")
    
    def _is_valid_player_box_vectorized(self, wh: np.ndarray, frame_shape: tuple) -> np.ndarray:
        """Vectorized check if detected boxes have valid player proportions.
        
        Args:
            wh: Array of width and height values
            frame_shape: Shape of the input frame
            
        Returns:
            Boolean mask of valid boxes
        """
        frame_h, frame_w = frame_shape[:2]
        
        # Very relaxed size constraints to detect all people on field
        min_h = frame_h * 0.005  # Reduced minimum height
        max_h = frame_h * 0.8    # Increased maximum height
        min_w = frame_w * 0.002  # Reduced minimum width
        max_w = frame_w * 0.3    # Increased maximum width
        
        valid_h = (wh[:, 1] >= min_h) & (wh[:, 1] <= max_h)
        valid_w = (wh[:, 0] >= min_w) & (wh[:, 0] <= max_w)
        
        return valid_h & valid_w
    
    def _classify_role(self, box: List[float], frame_shape: Tuple[int, int], keypoints: Optional[np.ndarray] = None, frame=None) -> Tuple[str, float]:
        """Classify player role (player, goalkeeper, referee) with enhanced detection.
        
        Args:
            box: Player bounding box [x, y, w, h]
            frame_shape: Shape of the frame (h, w)
            keypoints: Optional player keypoints
            frame: Optional frame for color-based analysis
            
        Returns:
            Tuple of (role, confidence)
        """
        try:
            x, y, w, h = box
            frame_h, frame_w = frame_shape[:2]
            
            # Default role and confidence
            role = "player"
            confidence = 0.0
            
            # Initialize position-based features
            is_near_goal_line = False
            is_center_position = False
            
            # Position-based classification
            # Check if player is near goal line
            goal_line_margin = frame_w * 0.08  # 8% of frame width
            if x < goal_line_margin or (x + w) > (frame_w - goal_line_margin):
                is_near_goal_line = True
                confidence += 0.3
            
            # Check if player is in a central position (referees often are)
            center_margin_x = frame_w * 0.4  # 40% of center width
            center_margin_y = frame_h * 0.3  # 30% of center height
            
            is_center_x = (x + w/2) > (frame_w/2 - center_margin_x/2) and (x + w/2) < (frame_w/2 + center_margin_x/2)
            is_center_y = (y + h/2) > (frame_h/2 - center_margin_y/2) and (y + h/2) < (frame_h/2 + center_margin_y/2)
            
            if is_center_x and is_center_y:
                is_center_position = True
                confidence += 0.2
            
            # Color-based classification if frame is available
            if frame is not None:
                # Ensure valid coordinates
                x_start = max(0, int(x))
                y_start = max(0, int(y))
                x_end = min(frame_w, int(x + w))
                y_end = min(frame_h, int(y + h))
                
                if x_end > x_start and y_end > y_start:
                    player_img = frame[y_start:y_end, x_start:x_end]
                    
                    if player_img.size > 0:
                        # Convert to HSV for better color segmentation
                        hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
                        
                        # Goalkeeper detection - look for bright gloves and distinctive uniform colors
                        
                        # Detect bright gloves (typically white/light colored)
                        glove_lower = np.array([0, 0, 180])
                        glove_upper = np.array([180, 30, 255])
                        glove_mask = cv2.inRange(hsv, glove_lower, glove_upper)
                        glove_ratio = np.sum(glove_mask > 0) / float(player_img.size / 3)
                        
                        # Detect common goalkeeper colors (bright green, bright yellow, bright orange)
                        gk_color1_lower = np.array([35, 100, 150])  # Bright green
                        gk_color1_upper = np.array([85, 255, 255])
                        
                        gk_color2_lower = np.array([20, 100, 150])  # Bright yellow/orange
                        gk_color2_upper = np.array([35, 255, 255])
                        
                        gk_mask1 = cv2.inRange(hsv, gk_color1_lower, gk_color1_upper)
                        gk_mask2 = cv2.inRange(hsv, gk_color2_lower, gk_color2_upper)
                        gk_mask = cv2.bitwise_or(gk_mask1, gk_mask2)
                        gk_color_ratio = np.sum(gk_mask > 0) / float(player_img.size / 3)
                        
                        # Referee detection - look for black/dark uniform or bright yellow
                        ref_black_lower = np.array([0, 0, 0])
                        ref_black_upper = np.array([180, 30, 80])
                        
                        ref_yellow_lower = np.array([20, 100, 150])
                        ref_yellow_upper = np.array([35, 255, 255])
                        
                        ref_mask1 = cv2.inRange(hsv, ref_black_lower, ref_black_upper)
                        ref_mask2 = cv2.inRange(hsv, ref_yellow_lower, ref_yellow_upper)
                        
                        ref_black_ratio = np.sum(ref_mask1 > 0) / float(player_img.size / 3)
                        ref_yellow_ratio = np.sum(ref_mask2 > 0) / float(player_img.size / 3)
                        
                        # Combined decision logic
                        if is_near_goal_line:
                            if glove_ratio > 0.15 or gk_color_ratio > 0.3:
                                role = "goalkeeper"
                                confidence += 0.4
                        
                        # Referee typically has black or bright yellow/green uniform and is often in central position
                        if (ref_black_ratio > 0.6 or ref_yellow_ratio > 0.4) and is_center_position:
                            role = "referee"
                            confidence += 0.5
            
            # Use pose information if available
            if keypoints is not None and len(keypoints) > 0:
                try:
                    # Referee often has hands raised or in specific positions
                    RIGHT_WRIST, LEFT_WRIST = 10, 9
                    NOSE = 0
                    
                    # Check if arms are raised (common referee gesture)
                    if (RIGHT_WRIST < len(keypoints) and LEFT_WRIST < len(keypoints) and 
                        NOSE < len(keypoints)):
                        
                        right_wrist_y = keypoints[RIGHT_WRIST][1]
                        left_wrist_y = keypoints[LEFT_WRIST][1]
                        nose_y = keypoints[NOSE][1]
                        
                        # Check if either wrist is above the nose (arm raised)
                        if (keypoints[RIGHT_WRIST][2] > 0.3 and 
                            right_wrist_y < nose_y) or (
                            keypoints[LEFT_WRIST][2] > 0.3 and 
                            left_wrist_y < nose_y):
                            
                            if is_center_position:  # If in center position, likely referee
                                role = "referee"
                                confidence += 0.3
                                
                except Exception as e:
                    print(f"Error in pose-based role classification: {str(e)}")
            
            # Normalize confidence to [0,1]
            confidence = min(1.0, confidence)
            
            return role, confidence
        except Exception as e:
            print(f"Error in role classification: {str(e)}")
            return "player", 0.0

    def _apply_temporal_consistency(self, boxes, kept_indices, keypoints, roles, frame_shape):
        """Apply temporal consistency to detections using previous frames."""
        if not self.detection_history:
            return boxes, kept_indices, keypoints, roles
            
        try:
            # If no previous detections, return current ones
            if not boxes or not kept_indices:
                return boxes, kept_indices, keypoints, roles
                
            # If previous frames exist, check for consistency
            last_boxes, last_indices, last_kpts, last_roles = self.detection_history[-1] if self.detection_history else ([], [], [], [])
            
            if not last_boxes:
                return boxes, kept_indices, keypoints, roles
                
            # Try to match current detections with previous frame
            matched_indices = []
            matched_roles = []
            
            # For each current detection, find best match in previous frame
            for i, box in enumerate(boxes):
                best_match = -1
                best_iou = 0.3  # Minimum IOU to consider a match
                
                x1, y1, w, h = box
                
                # Check against previous boxes
                for j, prev_box in enumerate(last_boxes):
                    prev_x, prev_y, prev_w, prev_h = prev_box
                    
                    # Calculate IOU
                    iou = self._calculate_iou(
                        [x1, y1, x1 + w, y1 + h],
                        [prev_x, prev_y, prev_x + prev_w, prev_y + prev_h]
                    )
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = j
                
                # If found a match, use role from previous frame for consistency
                if best_match >= 0 and best_match < len(last_roles):
                    # If current role is referee but previous was player, use player
                    # to address the over-detection of referees
                    if roles[i] == 'referee' and last_roles[best_match] == 'player':
                        matched_roles.append('player')
                    # For goalkeeper, be more consistent - once detected, maintain
                    elif last_roles[best_match] == 'goalkeeper' or roles[i] == 'goalkeeper':
                        matched_roles.append('goalkeeper')
                    else:
                        # Otherwise keep current role
                        matched_roles.append(roles[i])
                else:
                    # No match found, use current role but limit referees
                    if roles[i] == 'referee':
                        # Count how many referees we already have
                        referee_count = sum(1 for r in matched_roles if r == 'referee')
                        if referee_count >= 2:  # Limit to 2 referees max
                            matched_roles.append('player')
                        else:
                            matched_roles.append('referee')
                    else:
                        matched_roles.append(roles[i])
                
                matched_indices.append(i)
            
            # Return temporally consistent detections
            consistent_boxes = [boxes[i] for i in matched_indices]
            consistent_kept = list(range(len(matched_indices)))
            consistent_keypoints = [keypoints[i] for i in matched_indices] if keypoints else []
            
            return consistent_boxes, consistent_kept, consistent_keypoints, matched_roles
            
        except Exception as e:
            print(f"Error applying temporal consistency: {str(e)}")
            return boxes, kept_indices, keypoints, roles 

    def detect_players_multi_scale(self, image: np.ndarray) -> Tuple[List[List[int]], List[np.ndarray]]:
        """Detect players with multi-scale detection to improve accuracy.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (bounding boxes, keypoints)
        """
        # Original image dimensions
        original_h, original_w = image.shape[:2]
        
        # Define scales based on image resolution
        if original_h >= 1080:  # Full HD or higher
            scales = [0.6, 0.8, 1.0, 1.2]  # Add smaller scale for higher resolution
        elif original_h >= 720:  # HD
            scales = [0.8, 1.0, 1.2]
        else:  # SD
            scales = [1.0, 1.2, 1.4]  # Use larger scales for smaller resolutions
            
        all_boxes = []
        all_keypoints = []
        
        # Process each scale
        for scale in scales:
            # Resize image according to scale
            if scale != 1.0:
                h, w = int(original_h * scale), int(original_w * scale)
                resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                resized = image
                h, w = original_h, original_w
                
            # Detect at this scale
            boxes, keypoints = self._detect_single_scale(resized)
            
            # Adjust coordinates back to original scale if needed
            if scale != 1.0:
                scale_factor = 1.0 / scale
                for i, box in enumerate(boxes):
                    # Scale box coordinates
                    boxes[i][0] = int(box[0] * scale_factor)
                    boxes[i][1] = int(box[1] * scale_factor)
                    boxes[i][2] = int(box[2] * scale_factor)
                    boxes[i][3] = int(box[3] * scale_factor)
                    
                    # Scale keypoint coordinates if we have them
                    if keypoints and i < len(keypoints):
                        for j in range(len(keypoints[i])):
                            if keypoints[i][j][2] > 0:  # Only scale valid keypoints
                                keypoints[i][j][0] *= scale_factor
                                keypoints[i][j][1] *= scale_factor
            
            all_boxes.extend(boxes)
            if keypoints:
                all_keypoints.extend(keypoints)
                
        # Apply non-maximum suppression to merge overlapping boxes
        if len(all_boxes) > 0:
            all_boxes, indices = self._apply_nms(all_boxes, 0.5)
            if all_keypoints:
                all_keypoints = [all_keypoints[i] for i in indices]
        
        return all_boxes, all_keypoints if all_keypoints else [] 

    def _detect_single_scale(self, frame: np.ndarray) -> Tuple[List[List[int]], List[np.ndarray]]:
        """Detect players in a single scale frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (bounding boxes, keypoints)
        """
        try:
            # Preprocess frame for better detection
            processed_frame = self._preprocess_frame(frame)
            
            # Run YOLOv8 inference for person detection (class 0)
            results = self.model(processed_frame, classes=0, conf=self.confidence_threshold)
            
            if not results or len(results) == 0:
                return [], []
                
            boxes = []
            keypoints = []
            
            # Process all detections
            for result in results:
                if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                    continue
                    
                # Extract bounding boxes
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    # Convert to [x, y, w, h] format
                    w = x2 - x1
                    h = y2 - y1
                    boxes.append([float(x1), float(y1), float(w), float(h)])
                    
                    # Extract keypoints if available
                    if hasattr(result, 'keypoints') and i < len(result.keypoints):
                        kpt = result.keypoints[i].data[0].cpu().numpy()
                        keypoints.append(kpt)
                    else:
                        keypoints.append(None)
            
            return boxes, keypoints
            
        except Exception as e:
            print(f"Error in single scale detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], []

    def _filter_detections(self, boxes: List[List[int]], keypoints: List[np.ndarray],
                       frame_shape: Tuple[int, int]) -> Tuple[List[List[int]], List[int], List[np.ndarray], List[str]]:
        """Filter detections based on size, aspect ratio, and other criteria.
        
        Args:
            boxes: List of bounding boxes [x, y, w, h]
            keypoints: List of pose keypoints
            frame_shape: Shape of the frame (h, w)
            
        Returns:
            Tuple of (filtered boxes, indices of kept boxes, filtered keypoints, roles)
        """
        filtered_boxes = []
        kept_indices = []
        filtered_keypoints = []
        roles = []
        
        frame_height, frame_width = frame_shape[:2]
        
        # Scale thresholds based on frame size
        scale_factor = max(1.0, min(frame_height, frame_width) / 500.0)
        min_height = self.min_height * scale_factor
        max_height = self.max_height * scale_factor
        
        # Filter boxes
        for i, box in enumerate(boxes):
            x, y, w, h = box
            
            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter based on dimensions
            if (min_height <= h <= max_height and
                self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                
                # Check if box is not too close to the edge (borders often have artifacts)
                edge_margin = min(frame_width, frame_height) * 0.01
                if (edge_margin < x < frame_width - edge_margin and 
                    edge_margin < y < frame_height - edge_margin):
                    
                    # Calculate role
                    kpt = keypoints[i] if keypoints and i < len(keypoints) else None
                    role, confidence = self._classify_role(box, frame_shape, kpt)
                    
                    filtered_boxes.append(box)
                    kept_idx = len(filtered_boxes) - 1
                    kept_indices.append(kept_idx)
                    roles.append(role)
                    
                    if kpt is not None:
                        filtered_keypoints.append(kpt)
                    elif keypoints:
                        filtered_keypoints.append(None)
        
        return filtered_boxes, kept_indices, filtered_keypoints, roles
        
    def _apply_nms(self, boxes: List[List[int]], iou_threshold: float) -> Tuple[List[List[int]], List[int]]:
        """Apply non-maximum suppression to remove overlapping boxes.
        
        Args:
            boxes: List of bounding boxes [x, y, w, h]
            iou_threshold: IoU threshold for overlap
            
        Returns:
            Tuple of (filtered boxes, indices of kept boxes)
        """
        if not boxes:
            return [], []
            
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        boxes_xyxy = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes]
        
        # Calculate areas
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes_xyxy]
        
        # Sort boxes by bottom-right y-coordinate (better for players)
        order = sorted(range(len(boxes_xyxy)), key=lambda i: boxes_xyxy[i][3], reverse=True)
        
        kept_indices = []
        while order:
            i = order.pop(0)
            kept_indices.append(i)
            
            # Remove overlapping boxes
            new_order = []
            for j in order:
                # Calculate intersection
                xx1 = max(boxes_xyxy[i][0], boxes_xyxy[j][0])
                yy1 = max(boxes_xyxy[i][1], boxes_xyxy[j][1])
                xx2 = min(boxes_xyxy[i][2], boxes_xyxy[j][2])
                yy2 = min(boxes_xyxy[i][3], boxes_xyxy[j][3])
                
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                intersection = w * h
                
                # Calculate IoU
                iou = intersection / (areas[i] + areas[j] - intersection)
                
                if iou <= iou_threshold:
                    new_order.append(j)
                    
            order = new_order
            
        # Return filtered boxes
        filtered_boxes = [boxes[i] for i in kept_indices]
        
        return filtered_boxes, kept_indices 