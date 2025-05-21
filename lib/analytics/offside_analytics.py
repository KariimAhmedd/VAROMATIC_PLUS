"""
Offside Analytics Module for VAROMATIC+
Collects and analyzes offside events during a match
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import os
import json
from datetime import datetime

@dataclass
class OffsideEvent:
    """Represents a single offside event in a match"""
    timestamp: float  # Time in video (seconds)
    frame_number: int  # Frame where offside occurred
    player_id: Optional[int] = None  # ID of the player caught offside (if known)
    player_position: Tuple[float, float] = (0, 0)  # (x, y) coordinates
    offside_line_position: float = 0  # X position of offside line
    team: int = 0  # 0 or 1, representing which team was offside
    zone: Optional[str] = None  # Field zone (e.g., "left flank", "center")
    match_time: Optional[str] = None  # Match time (e.g., "12:34")
    confidence: float = 1.0  # Confidence score of offside detection

class OffsideAnalytics:
    """Collects and analyzes offside events during a match"""
    
    def __init__(self, pitch_dimensions: Tuple[int, int] = (105, 68)):
        """
        Initialize the analytics engine
        
        Args:
            pitch_dimensions: (length, width) in meters of the pitch
        """
        self.pitch_dimensions = pitch_dimensions
        self.offside_events: List[OffsideEvent] = []
        self.reset_analytics()
        
        # Define pitch zones
        self._define_zones()
        
        # Create output directories
        self.output_dir = os.path.join(os.getcwd(), "analysis_results", f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def reset_analytics(self):
        """Reset all analytics data"""
        self.offside_events = []
        self.team0_events = []  # Events for team 0
        self.team1_events = []  # Events for team 1
        self.player_stats = {}  # Player-specific offside statistics
        self.zone_stats = {}    # Zone-specific offside statistics
        self.frame_to_events = {}  # Mapping from frame number to events
        
        # For heatmap generation
        self.heatmap_data = np.zeros((20, 20))  # 20x20 grid for the pitch
        
        # For timeline
        self.timeline_events = []
        
        # Additional event types
        self.goal_events = []
        self.key_pass_events = []
        
        # Team scores
        self.team0_score = 0
        self.team1_score = 0
        
        # Performance metrics
        self.attacking_efficiency = {0: 0.0, 1: 0.0}  # Ratio of goals to offsides
    
    def _define_zones(self):
        """Define zones on the pitch for zone-based analysis"""
        pitch_length, pitch_width = self.pitch_dimensions
        
        # Define thirds of the pitch (length-wise)
        self.defensive_third = (0, pitch_length/3)
        self.middle_third = (pitch_length/3, 2*pitch_length/3)
        self.attacking_third = (2*pitch_length/3, pitch_length)
        
        # Define channels (width-wise)
        self.left_channel = (0, pitch_width/3)
        self.central_channel = (pitch_width/3, 2*pitch_width/3)
        self.right_channel = (2*pitch_width/3, pitch_width)
        
        # Names of the 9 zones
        self.zones = {
            "defensive_left": (self.defensive_third, self.left_channel),
            "defensive_center": (self.defensive_third, self.central_channel),
            "defensive_right": (self.defensive_third, self.right_channel),
            "middle_left": (self.middle_third, self.left_channel),
            "middle_center": (self.middle_third, self.central_channel),
            "middle_right": (self.middle_third, self.right_channel),
            "attacking_left": (self.attacking_third, self.left_channel),
            "attacking_center": (self.attacking_third, self.central_channel),
            "attacking_right": (self.attacking_third, self.right_channel),
        }
    
    def get_zone(self, position: Tuple[float, float]) -> str:
        """
        Determine which zone a position is in
        
        Args:
            position: (x, y) coordinates on the pitch
            
        Returns:
            Name of the zone
        """
        x, y = position
        
        # Normalize to pitch dimensions
        x_norm = x / self.pitch_dimensions[0]
        y_norm = y / self.pitch_dimensions[1]
        
        # Get the zones
        for zone_name, ((x_min, x_max), (y_min, y_max)) in self.zones.items():
            if (x_min <= x_norm * self.pitch_dimensions[0] <= x_max and 
                y_min <= y_norm * self.pitch_dimensions[1] <= y_max):
                return zone_name
        
        return "unknown"
    
    def add_offside_event(self, event: OffsideEvent):
        """
        Add an offside event to the analytics
        
        Args:
            event: The offside event to add
        """
        # Add to all events
        self.offside_events.append(event)
        
        # Add to team-specific events
        if event.team == 0:
            self.team0_events.append(event)
        else:
            self.team1_events.append(event)
        
        # Update player statistics
        if event.player_id is not None:
            if event.player_id not in self.player_stats:
                self.player_stats[event.player_id] = {
                    'count': 0,
                    'positions': [],
                    'team': event.team
                }
            self.player_stats[event.player_id]['count'] += 1
            self.player_stats[event.player_id]['positions'].append(event.player_position)
        
        # Update zone statistics
        zone = event.zone or self.get_zone(event.player_position)
        if zone not in self.zone_stats:
            self.zone_stats[zone] = {
                'team0_count': 0,
                'team1_count': 0,
                'total_count': 0
            }
        
        self.zone_stats[zone]['total_count'] += 1
        if event.team == 0:
            self.zone_stats[zone]['team0_count'] += 1
        else:
            self.zone_stats[zone]['team1_count'] += 1
        
        # Update heatmap data
        x, y = event.player_position
        x_idx = min(19, max(0, int(x / self.pitch_dimensions[0] * 20)))
        y_idx = min(19, max(0, int(y / self.pitch_dimensions[1] * 20)))
        self.heatmap_data[y_idx, x_idx] += 1
        
        # Add to frame mapping
        if event.frame_number not in self.frame_to_events:
            self.frame_to_events[event.frame_number] = []
        self.frame_to_events[event.frame_number].append(event)
        
        # Add to timeline
        self.timeline_events.append({
            'type': 'offside',
            'timestamp': event.timestamp,
            'frame': event.frame_number,
            'team': event.team,
            'position': event.player_position,
            'player_id': event.player_id,
            'match_time': event.match_time or self.format_time(event.timestamp)
        })
        
        # Update efficiency metrics
        self._update_efficiency_metrics()
    
    def add_goal_event(self, frame_number: int, timestamp: float, team: int, 
                     position: Tuple[float, float] = (0, 0), 
                     match_time: Optional[str] = None):
        """
        Add a goal event to the timeline
        
        Args:
            frame_number: Frame where the goal occurred
            timestamp: Time in video (seconds)
            team: Team that scored (0 or 1)
            position: (x, y) coordinates of the goal
            match_time: Match time string (e.g., "12:34")
        """
        # Update score
        if team == 0:
            self.team0_score += 1
        else:
            self.team1_score += 1
        
        # Add to goal events
        goal_event = {
            'frame': frame_number,
            'timestamp': timestamp,
            'team': team,
            'position': position,
            'match_time': match_time or self.format_time(timestamp)
        }
        self.goal_events.append(goal_event)
        
        # Add to timeline
        self.timeline_events.append({
            'type': 'goal',
            'timestamp': timestamp,
            'frame': frame_number,
            'team': team,
            'position': position,
            'match_time': match_time or self.format_time(timestamp)
        })
        
        # Update efficiency metrics
        self._update_efficiency_metrics()
    
    def add_key_pass_event(self, frame_number: int, timestamp: float, team: int, 
                         position: Tuple[float, float] = (0, 0), 
                         match_time: Optional[str] = None):
        """
        Add a key pass event to the timeline
        
        Args:
            frame_number: Frame where the key pass occurred
            timestamp: Time in video (seconds)
            team: Team that made the pass (0 or 1)
            position: (x, y) coordinates of the pass
            match_time: Match time string (e.g., "12:34")
        """
        # Add to key pass events
        key_pass_event = {
            'frame': frame_number,
            'timestamp': timestamp,
            'team': team,
            'position': position,
            'match_time': match_time or self.format_time(timestamp)
        }
        self.key_pass_events.append(key_pass_event)
        
        # Add to timeline
        self.timeline_events.append({
            'type': 'key_pass',
            'timestamp': timestamp,
            'frame': frame_number,
            'team': team,
            'position': position,
            'match_time': match_time or self.format_time(timestamp)
        })
    
    def _update_efficiency_metrics(self):
        """Update efficiency metrics based on current events"""
        for team in [0, 1]:
            goals = self.team0_score if team == 0 else self.team1_score
            offsides = len(self.team0_events) if team == 0 else len(self.team1_events)
            
            # Avoid division by zero
            if offsides > 0:
                self.attacking_efficiency[team] = goals / offsides
            else:
                self.attacking_efficiency[team] = 0.0
    
    def format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS"""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes:02d}:{secs:02d}"
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert offside events to a pandas DataFrame
        
        Returns:
            DataFrame containing all offside events
        """
        data = []
        for event in self.offside_events:
            data.append({
                'timestamp': event.timestamp,
                'frame': event.frame_number,
                'player_id': event.player_id,
                'position_x': event.player_position[0],
                'position_y': event.player_position[1],
                'offside_line': event.offside_line_position,
                'team': event.team,
                'zone': event.zone or self.get_zone(event.player_position),
                'match_time': event.match_time or self.format_time(event.timestamp),
                'confidence': event.confidence
            })
        
        return pd.DataFrame(data)
    
    def get_efficiency_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get efficiency statistics
        
        Returns:
            Dictionary with team efficiency stats
        """
        stats = {
            'team0': {
                'goals': self.team0_score,
                'offsides': len(self.team0_events),
                'efficiency': self.attacking_efficiency[0],
                'key_passes': len([e for e in self.key_pass_events if e['team'] == 0])
            },
            'team1': {
                'goals': self.team1_score,
                'offsides': len(self.team1_events),
                'efficiency': self.attacking_efficiency[1],
                'key_passes': len([e for e in self.key_pass_events if e['team'] == 1])
            }
        }
        return stats
    
    def save_analytics(self, filename: str = "offside_analytics.json"):
        """
        Save analytics data to a JSON file
        
        Args:
            filename: Name of the file to save to
        """
        data = {
            'total_offsides': len(self.offside_events),
            'team0_offsides': len(self.team0_events),
            'team1_offsides': len(self.team1_events),
            'team0_goals': self.team0_score,
            'team1_goals': self.team1_score,
            'team0_key_passes': len([e for e in self.key_pass_events if e['team'] == 0]),
            'team1_key_passes': len([e for e in self.key_pass_events if e['team'] == 1]),
            'player_stats': self.player_stats,
            'zone_stats': self.zone_stats,
            'efficiency_stats': {
                'team0': self.attacking_efficiency[0],
                'team1': self.attacking_efficiency[1]
            },
            'offside_events': [
                {
                    'timestamp': event.timestamp,
                    'frame': event.frame_number,
                    'player_id': event.player_id,
                    'position_x': event.player_position[0],
                    'position_y': event.player_position[1],
                    'offside_line': event.offside_line_position,
                    'team': event.team,
                    'zone': event.zone or self.get_zone(event.player_position),
                    'match_time': event.match_time or self.format_time(event.timestamp),
                    'confidence': event.confidence
                }
                for event in self.offside_events
            ],
            'goal_events': self.goal_events,
            'key_pass_events': self.key_pass_events
        }
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Offside analytics saved to {filepath}")
        
        # Also save as CSV for easier data analysis
        csv_path = os.path.join(self.output_dir, "offside_events.csv")
        self.to_dataframe().to_csv(csv_path, index=False)
        print(f"Offside events saved to CSV: {csv_path}")
    
    def get_player_offsides(self, player_id: int) -> int:
        """
        Get the number of offsides for a specific player
        
        Args:
            player_id: ID of the player
            
        Returns:
            Number of offsides for that player
        """
        if player_id in self.player_stats:
            return self.player_stats[player_id]['count']
        return 0
    
    def get_zone_offsides(self, zone: str) -> Dict[str, int]:
        """
        Get offside statistics for a specific zone
        
        Args:
            zone: Name of the zone
            
        Returns:
            Dictionary with team0_count, team1_count, and total_count
        """
        if zone in self.zone_stats:
            return self.zone_stats[zone]
        return {'team0_count': 0, 'team1_count': 0, 'total_count': 0}
    
    def get_team_offsides(self, team: int) -> int:
        """
        Get the number of offsides for a specific team
        
        Args:
            team: Team number (0 or 1)
            
        Returns:
            Number of offsides for that team
        """
        if team == 0:
            return len(self.team0_events)
        else:
            return len(self.team1_events)
    
    def get_team_goals(self, team: int) -> int:
        """
        Get the number of goals for a specific team
        
        Args:
            team: Team number (0 or 1)
            
        Returns:
            Number of goals for that team
        """
        return self.team0_score if team == 0 else self.team1_score
    
    def get_events_by_timeframe(self, start_time: float, end_time: float) -> Dict[str, List]:
        """
        Get events within a specific timeframe
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Dictionary with events by type
        """
        offsides = [e for e in self.offside_events 
                   if start_time <= e.timestamp <= end_time]
        
        goals = [e for e in self.goal_events 
                if start_time <= e['timestamp'] <= end_time]
        
        key_passes = [e for e in self.key_pass_events 
                     if start_time <= e['timestamp'] <= end_time]
        
        return {
            'offsides': offsides,
            'goals': goals,
            'key_passes': key_passes
        } 