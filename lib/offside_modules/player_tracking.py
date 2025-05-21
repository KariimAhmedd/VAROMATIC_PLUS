import cv2
import numpy as np
from typing import List, Tuple, Dict
import torch
from collections import defaultdict

class PlayerTracker:
    def __init__(self):
        """Initialize the player tracker."""
        self.tracks = defaultdict(list)  # Track history for each player
        self.next_id = 0
        self.max_track_length = 30  # Maximum number of frames to track
        
    def update(self, boxes: List[List[int]], team_assignments: List[int]) -> Tuple[List[int], List[List[float]]]:
        """Update tracks with new detections.
        
        Args:
            boxes: List of bounding boxes [x, y, w, h]
            team_assignments: List of team assignments (0 or 1)
            
        Returns:
            Tuple containing:
            - List of track IDs
            - List of velocity vectors [vx, vy]
        """
        if not boxes:
            return [], []
            
        # Convert boxes to center format
        centers = []
        for box in boxes:
            x, y, w, h = box
            centers.append([x + w/2, y + h/2])
            
        # Match new detections to existing tracks
        track_ids = []
        velocities = []
        
        if not self.tracks:
            # First frame, create new tracks
            for i, (center, team) in enumerate(zip(centers, team_assignments)):
                self.tracks[self.next_id] = [(center, team)]
                track_ids.append(self.next_id)
                velocities.append([0, 0])
                self.next_id += 1
        else:
            # Match detections to tracks using Hungarian algorithm
            cost_matrix = np.zeros((len(centers), len(self.tracks)))
            
            for i, center in enumerate(centers):
                for j, track_id in enumerate(self.tracks.keys()):
                    last_pos = self.tracks[track_id][-1][0]
                    cost_matrix[i,j] = np.linalg.norm(np.array(center) - np.array(last_pos))
                    
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Update matched tracks and create new ones
            matched_detections = set(row_ind)
            matched_tracks = set(col_ind)
            
            track_list = list(self.tracks.keys())
            
            for i, center in enumerate(centers):
                if i in matched_detections:
                    # Update existing track
                    track_idx = col_ind[list(row_ind).index(i)]
                    track_id = track_list[track_idx]
                    self.tracks[track_id].append((center, team_assignments[i]))
                    
                    # Calculate velocity
                    if len(self.tracks[track_id]) >= 2:
                        prev_pos = np.array(self.tracks[track_id][-2][0])
                        curr_pos = np.array(center)
                        velocity = curr_pos - prev_pos
                    else:
                        velocity = np.array([0, 0])
                        
                    track_ids.append(track_id)
                    velocities.append(velocity.tolist())
                else:
                    # Create new track
                    self.tracks[self.next_id] = [(center, team_assignments[i])]
                    track_ids.append(self.next_id)
                    velocities.append([0, 0])
                    self.next_id += 1
                    
            # Remove unmatched tracks
            for i, track_id in enumerate(track_list):
                if i not in matched_tracks:
                    del self.tracks[track_id]
                    
        # Limit track history
        for track_id in self.tracks:
            if len(self.tracks[track_id]) > self.max_track_length:
                self.tracks[track_id] = self.tracks[track_id][-self.max_track_length:]
                
        return track_ids, velocities
        
    def get_track_history(self, track_id: int) -> List[Tuple[List[float], int]]:
        """Get position history for a track.
        
        Args:
            track_id: ID of track to get history for
            
        Returns:
            List of (position, team) tuples
        """
        return self.tracks.get(track_id, [])
        
    def predict_position(self, track_id: int, frames_ahead: int = 5) -> Tuple[List[float], float]:
        """Predict future position of a player.
        
        Args:
            track_id: ID of track to predict
            frames_ahead: Number of frames to predict ahead
            
        Returns:
            Tuple of:
            - Predicted position [x, y]
            - Confidence score
        """
        history = self.tracks.get(track_id, [])
        if len(history) < 2:
            return None, 0.0
            
        # Get recent positions
        positions = np.array([pos for pos, _ in history[-10:]])
        
        # Fit linear motion model
        timestamps = np.arange(len(positions))
        A = np.vstack([timestamps, np.ones(len(timestamps))]).T
        m_x, b_x = np.linalg.lstsq(A, positions[:,0], rcond=None)[0]
        m_y, b_y = np.linalg.lstsq(A, positions[:,1], rcond=None)[0]
        
        # Predict future position
        future_time = len(positions) + frames_ahead
        pred_x = m_x * future_time + b_x
        pred_y = m_y * future_time + b_y
        
        # Calculate prediction confidence based on fit error
        error_x = np.mean(np.abs(positions[:,0] - (m_x * timestamps + b_x)))
        error_y = np.mean(np.abs(positions[:,1] - (m_y * timestamps + b_y)))
        confidence = 1.0 / (1.0 + error_x + error_y)
        
        return [pred_x, pred_y], confidence
        
    def draw_tracks(self, frame: np.ndarray, track_ids: List[int], 
                   team_assignments: List[int]) -> np.ndarray:
        """Draw track histories on frame.
        
        Args:
            frame: Input frame
            track_ids: List of track IDs to draw
            team_assignments: List of team assignments
            
        Returns:
            Frame with tracks drawn
        """
        output = frame.copy()
        
        colors = [(0, 0, 255), (255, 0, 0)]  # Red for team 0, Blue for team 1
        
        for track_id, team in zip(track_ids, team_assignments):
            history = self.tracks.get(track_id, [])
            if len(history) < 2:
                continue
                
            # Draw track line
            points = np.array([pos for pos, _ in history], dtype=np.int32)
            cv2.polylines(output, [points], False, colors[team], 2)
            
            # Draw predicted position
            pred_pos, conf = self.predict_position(track_id)
            if pred_pos is not None and conf > 0.5:
                cv2.circle(output, (int(pred_pos[0]), int(pred_pos[1])), 
                          5, colors[team], -1)
                
        return output 