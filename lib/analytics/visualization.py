"""
Visualization Module for VAROMATIC+ Analytics
Generates visualizations for offside analytics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
import seaborn as sns
import os
from typing import List, Dict, Tuple, Optional, Any
import io
from datetime import datetime
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import cv2
from .offside_analytics import OffsideAnalytics, OffsideEvent

class AnalyticsVisualizer:
    """Generates visualizations for offside analytics"""
    
    def __init__(self, analytics: OffsideAnalytics):
        """
        Initialize the visualizer
        
        Args:
            analytics: OffsideAnalytics instance with collected data
        """
        self.analytics = analytics
        self.output_dir = analytics.output_dir
        
        # Create visualizations directory
        self.viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('ggplot')
        
        # Team colors
        self.team_colors = ['#1E88E5', '#FFC107']  # Blue for team 0, Yellow for team 1
        
        # Custom colormap for heatmaps
        self.heatmap_cmap = LinearSegmentedColormap.from_list(
            'offside_cmap', ['#f7fbff', '#2171b5', '#08306b']
        )
        
        # Customized color maps for each team
        self.team0_cmap = LinearSegmentedColormap.from_list(
            'team0_cmap', ['#f7fbff', '#90caf9', '#1565c0']
        )
        self.team1_cmap = LinearSegmentedColormap.from_list(
            'team1_cmap', ['#fffde7', '#ffecb3', '#ff6f00']
        )
        
        # Define a pitch outline style
        self.pitch_outline = {
            'linewidth': 2,
            'color': '#666666',
            'zorder': 2
        }
    
    def draw_pitch(self, ax: plt.Axes, pitch_color='#1a782f', line_color='white'):
        """
        Draw a football pitch on the given axes
        
        Args:
            ax: Matplotlib axes to draw on
            pitch_color: Color of the pitch
            line_color: Color of the lines
        """
        # Pitch dimensions
        pitch_length, pitch_width = self.analytics.pitch_dimensions
        
        # Set limits
        ax.set_xlim(0, pitch_length)
        ax.set_ylim(0, pitch_width)
        
        # Draw pitch background
        ax.add_patch(patches.Rectangle((0, 0), pitch_length, pitch_width, 
                                      facecolor=pitch_color, edgecolor=line_color, zorder=1))
        
        # Draw halfway line
        ax.plot([pitch_length/2, pitch_length/2], [0, pitch_width], color=line_color, zorder=2)
        
        # Draw center circle
        center_circle = patches.Circle((pitch_length/2, pitch_width/2), 9.15, 
                                       fill=False, color=line_color, zorder=2)
        ax.add_patch(center_circle)
        
        # Draw center spot
        center_spot = patches.Circle((pitch_length/2, pitch_width/2), 0.3, 
                                    color=line_color, zorder=2)
        ax.add_patch(center_spot)
        
        # Draw penalty areas
        penalty_area_left = patches.Rectangle((0, pitch_width/2 - 20.16), 16.5, 40.32, 
                                             fill=False, color=line_color, zorder=2)
        penalty_area_right = patches.Rectangle((pitch_length - 16.5, pitch_width/2 - 20.16), 
                                              16.5, 40.32, fill=False, color=line_color, zorder=2)
        ax.add_patch(penalty_area_left)
        ax.add_patch(penalty_area_right)
        
        # Draw goal areas
        goal_area_left = patches.Rectangle((0, pitch_width/2 - 9.16), 5.5, 18.32, 
                                          fill=False, color=line_color, zorder=2)
        goal_area_right = patches.Rectangle((pitch_length - 5.5, pitch_width/2 - 9.16), 
                                           5.5, 18.32, fill=False, color=line_color, zorder=2)
        ax.add_patch(goal_area_left)
        ax.add_patch(goal_area_right)
        
        # Draw penalty spots
        penalty_spot_left = patches.Circle((11, pitch_width/2), 0.3, color=line_color, zorder=2)
        penalty_spot_right = patches.Circle((pitch_length - 11, pitch_width/2), 0.3, 
                                           color=line_color, zorder=2)
        ax.add_patch(penalty_spot_left)
        ax.add_patch(penalty_spot_right)
        
        # Remove axes ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        return ax
    
    def generate_offside_heatmap(self, 
                               title: str = "Offside Heatmap", 
                               save_filename: str = "offside_heatmap.png") -> Figure:
        """
        Generate a heatmap showing offside locations
        
        Args:
            title: Title of the plot
            save_filename: Filename to save the plot to
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Draw pitch
        self.draw_pitch(ax)
        
        # Create heatmap for all offsides
        if len(self.analytics.offside_events) > 0:
            positions = np.array([event.player_position for event in self.analytics.offside_events])
            x = positions[:, 0]
            y = positions[:, 1]
            
            # Create smoothed heatmap using kernel density estimation
            if len(x) >= 3:  # Need at least 3 points for KDE
                # Create a meshgrid covering the pitch
                pitch_length, pitch_width = self.analytics.pitch_dimensions
                x_grid = np.linspace(0, pitch_length, 100)
                y_grid = np.linspace(0, pitch_width, 100)
                xx, yy = np.meshgrid(x_grid, y_grid)
                
                try:
                    # Try to compute kernel density estimate
                    from scipy.stats import gaussian_kde
                    k = gaussian_kde(np.vstack([x, y]))
                    grid_density = k(np.vstack([xx.flatten(), yy.flatten()]))
                    grid_density = grid_density.reshape(xx.shape)
                    
                    # Plot the heatmap with contours
                    contour = ax.contourf(
                        xx, yy, grid_density, 
                        levels=20, 
                        cmap=self.heatmap_cmap, 
                        alpha=0.7,
                        zorder=1
                    )
                    
                    # Add a colorbar
                    cbar = plt.colorbar(contour, ax=ax, pad=0.01)
                    cbar.set_label('Offside Density', rotation=270, labelpad=20)
                    
                except Exception as e:
                    print(f"KDE failed, falling back to scatter: {e}")
                    # Fallback to standard scatter
                    ax.scatter(
                        x, y, 
                        c='#1976D2', s=150, alpha=0.6, 
                        edgecolor='white', linewidth=1,
                        zorder=3
                    )
            else:
                # Just use scatter for a few points
                ax.scatter(
                    x, y, 
                    c='#1976D2', s=150, alpha=0.6, 
                    edgecolor='white', linewidth=1,
                    zorder=3
                )
                
            # Add team-specific plots
            team0_positions = np.array([
                event.player_position for event in self.analytics.offside_events
                if event.team == 0
            ])
            
            team1_positions = np.array([
                event.player_position for event in self.analytics.offside_events
                if event.team == 1
            ])
            
            # Add team markers
            if len(team0_positions) > 0:
                ax.scatter(
                    team0_positions[:, 0], team0_positions[:, 1],
                    marker='o', s=120, color=self.team_colors[0],
                    edgecolor='white', linewidth=1.5, alpha=0.8,
                    label='Team 1 Offsides', zorder=4
                )
                
            if len(team1_positions) > 0:
                ax.scatter(
                    team1_positions[:, 0], team1_positions[:, 1],
                    marker='s', s=120, color=self.team_colors[1],
                    edgecolor='white', linewidth=1.5, alpha=0.8,
                    label='Team 2 Offsides', zorder=4
                )
        
        # Add title and legend
        ax.set_title(title, fontsize=18, pad=20)
        if len(self.analytics.offside_events) > 0:
            ax.legend(loc='upper right', fontsize=12)
        
        # Add context information
        team0_count = self.analytics.get_team_offsides(0)
        team1_count = self.analytics.get_team_offsides(1)
        info_text = (
            f"Total Offsides: {len(self.analytics.offside_events)}\n"
            f"Team 1: {team0_count} offsides\n"
            f"Team 2: {team1_count} offsides"
        )
        
        # Add info text box
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(
            0.02, 0.02, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(self.viz_dir, save_filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        print(f"Offside heatmap saved to: {filepath}")
        
        return fig
    
    def generate_player_offside_chart(self, 
                                     title: str = "Player Offside Frequency", 
                                     save_filename: str = "player_offsides.png",
                                     max_players: int = 10) -> Figure:
        """
        Generate a bar chart showing offsides per player
        
        Args:
            title: Title of the plot
            save_filename: Filename to save the plot to
            max_players: Maximum number of players to include
            
        Returns:
            Matplotlib Figure object
        """
        # Get player stats
        player_stats = self.analytics.player_stats
        
        # Sort players by offside count
        sorted_players = sorted(player_stats.items(), 
                               key=lambda x: x[1]['count'], reverse=True)
        
        # Take top players
        top_players = sorted_players[:max_players]
        
        # Create lists for plotting
        player_ids = [f"Player {p[0]}" for p in top_players]
        offside_counts = [p[1]['count'] for p in top_players]
        player_teams = [p[1]['team'] for p in top_players]
        colors = [self.team_colors[team] for team in player_teams]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot horizontal bar chart
        bars = ax.barh(player_ids, offside_counts, color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f"{width:.0f}", ha='left', va='center', fontsize=12)
        
        # Add title and labels
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Number of Offsides', fontsize=14)
        ax.set_ylabel('Player', fontsize=14)
        
        # Add team legend
        from matplotlib.lines import Line2D
        team_legend = [
            Line2D([0], [0], color=self.team_colors[0], lw=4, label='Team 1'),
            Line2D([0], [0], color=self.team_colors[1], lw=4, label='Team 2')
        ]
        ax.legend(handles=team_legend, loc='lower right')
        
        # Add average line if we have enough data
        if len(offside_counts) >= 3:
            avg = np.mean(offside_counts)
            ax.axvline(x=avg, color='gray', linestyle='--', alpha=0.7)
            ax.text(avg + 0.1, -0.5, f'Average: {avg:.1f}', 
                  color='gray', fontsize=10, ha='left', va='center')
        
        # Add a subtle grid
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # Add contextual information
        if offside_counts:
            # Calculate average offsides per player for each team
            team0_players = [p for p in sorted_players if p[1]['team'] == 0]
            team1_players = [p for p in sorted_players if p[1]['team'] == 1]
            
            team0_avg = np.mean([p[1]['count'] for p in team0_players]) if team0_players else 0
            team1_avg = np.mean([p[1]['count'] for p in team1_players]) if team1_players else 0
            
            info_text = (
                f"Total Players with Offsides: {len(sorted_players)}\n"
                f"Team 1 Average: {team0_avg:.1f} offsides per player\n"
                f"Team 2 Average: {team1_avg:.1f} offsides per player"
            )
            
            # Add info text box
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(
                0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props
            )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(self.viz_dir, save_filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Player offside chart saved to: {filepath}")
        
        return fig
    
    def generate_zone_team_behavior(self, 
                                  title: str = "Zone-Based Team Behavior", 
                                  save_filename: str = "zone_behavior.png") -> Figure:
        """
        Generate a visualization of team behavior by zone
        
        Args:
            title: Title of the plot
            save_filename: Filename to save the plot to
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure with two subplots (one for each team)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Draw pitches
        self.draw_pitch(ax1, pitch_color='#1a782f', line_color='white')
        self.draw_pitch(ax2, pitch_color='#1a782f', line_color='white')
        
        # Set titles for each subplot
        ax1.set_title('Team 1 Offside Distribution', fontsize=16)
        ax2.set_title('Team 2 Offside Distribution', fontsize=16)
        
        # Get zone stats
        zone_stats = self.analytics.zone_stats
        
        # Define zone centers based on the analytics zone definitions
        zone_centers = {}
        pitch_length, pitch_width = self.analytics.pitch_dimensions
        
        # Calculate the center of each zone
        for zone_name, ((x_min, x_max), (y_min, y_max)) in self.analytics.zones.items():
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            zone_centers[zone_name] = (center_x, center_y)
            
            # Draw zone rectangles with light colors and labels
            rect1 = patches.Rectangle(
                (x_min, y_min), 
                x_max - x_min, 
                y_max - y_min, 
                linewidth=1, 
                edgecolor='gray', 
                facecolor='none', 
                alpha=0.5,
                zorder=1
            )
            
            rect2 = patches.Rectangle(
                (x_min, y_min), 
                x_max - x_min, 
                y_max - y_min, 
                linewidth=1, 
                edgecolor='gray', 
                facecolor='none', 
                alpha=0.5,
                zorder=1
            )
            
            ax1.add_patch(rect1)
            ax2.add_patch(rect2)
            
            # Add zone labels
            readable_name = zone_name.replace('_', ' ').title()
            ax1.text(center_x, center_y, readable_name, 
                     ha='center', va='center', fontsize=8, 
                     color='white', alpha=0.7, zorder=2)
            ax2.text(center_x, center_y, readable_name, 
                     ha='center', va='center', fontsize=8, 
                     color='white', alpha=0.7, zorder=2)
        
        # Find the max count to normalize circle sizes
        team0_max = max([stats['team0_count'] for stats in zone_stats.values()], default=1)
        team1_max = max([stats['team1_count'] for stats in zone_stats.values()], default=1)
        max_count = max(team0_max, team1_max, 1)  # Avoid division by zero
        
        # Draw circles for each zone
        for zone_name, stats in zone_stats.items():
            if zone_name not in zone_centers:
                continue
                
            center_x, center_y = zone_centers[zone_name]
            team0_count = stats['team0_count']
            team1_count = stats['team1_count']
            
            # Calculate size based on percentage of max count
            team0_size = max(500 * (team0_count / max_count), 50) if team0_count > 0 else 0
            team1_size = max(500 * (team1_count / max_count), 50) if team1_count > 0 else 0
            
            # Add circles to represent number of offsides
            if team0_count > 0:
                team0_circle = plt.Circle(
                    (center_x, center_y), 
                    radius=np.sqrt(team0_size / np.pi), 
                    color=self.team_colors[0], 
                    alpha=0.7,
                    zorder=3
                )
                ax1.add_patch(team0_circle)
                ax1.text(center_x, center_y, str(team0_count), 
                        ha='center', va='center', fontsize=11, fontweight='bold',
                        zorder=4)
            
            if team1_count > 0:
                team1_circle = plt.Circle(
                    (center_x, center_y), 
                    radius=np.sqrt(team1_size / np.pi), 
                    color=self.team_colors[1], 
                    alpha=0.7,
                    zorder=3
                )
                ax2.add_patch(team1_circle)
                ax2.text(center_x, center_y, str(team1_count), 
                        ha='center', va='center', fontsize=11, fontweight='bold',
                        zorder=4)
        
        # Add context information - top zones for each team
        if zone_stats:
            # Team 1 top zones
            team0_zones = {zone: stats['team0_count'] for zone, stats in zone_stats.items() if stats['team0_count'] > 0}
            team0_sorted = sorted(team0_zones.items(), key=lambda x: x[1], reverse=True)
            team0_top_zones = team0_sorted[:3] if len(team0_sorted) >= 3 else team0_sorted
            
            team0_info = "Top Offside Zones - Team 1:\n"
            for zone, count in team0_top_zones:
                readable_zone = zone.replace('_', ' ').title()
                team0_info += f"• {readable_zone}: {count} offsides\n"
            
            # Team 2 top zones
            team1_zones = {zone: stats['team1_count'] for zone, stats in zone_stats.items() if stats['team1_count'] > 0}
            team1_sorted = sorted(team1_zones.items(), key=lambda x: x[1], reverse=True)
            team1_top_zones = team1_sorted[:3] if len(team1_sorted) >= 3 else team1_sorted
            
            team1_info = "Top Offside Zones - Team 2:\n"
            for zone, count in team1_top_zones:
                readable_zone = zone.replace('_', ' ').title()
                team1_info += f"• {readable_zone}: {count} offsides\n"
            
            # Add info text boxes
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax1.text(
                0.02, 0.02, team0_info, transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props
            )
            
            ax2.text(
                0.02, 0.02, team1_info, transform=ax2.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props
            )
            
            # Add a comprehensive summary title
            team0_total = sum(team0_zones.values())
            team1_total = sum(team1_zones.values())
            
            fig.suptitle(
                f"{title} - Team 1: {team0_total} offsides, Team 2: {team1_total} offsides",
                fontsize=18, y=0.98
            )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(self.viz_dir, save_filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Zone behavior visualization saved to: {filepath}")
        
        return fig
    
    def generate_match_timeline(self, 
                               title: str = "Match Timeline", 
                               save_filename: str = "match_timeline.png") -> Figure:
        """
        Generate a timeline visualization of match events
        
        Args:
            title: Title of the plot
            save_filename: Filename to save the plot to
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Get events
        events = sorted(self.analytics.timeline_events, key=lambda x: x['timestamp'])
        
        if not events:
            ax.text(0.5, 0.5, "No events to display", fontsize=14, ha='center', va='center')
            ax.set_title(title, fontsize=16, pad=20)
            
            # Save figure
            filepath = os.path.join(self.viz_dir, save_filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Match timeline saved to: {filepath}")
            return fig
        
        # Time axis setup
        min_time = min(event['timestamp'] for event in events)
        max_time = max(event['timestamp'] for event in events)
        
        # Add padding to timeline
        time_padding = (max_time - min_time) * 0.1 if max_time > min_time else 10
        ax.set_xlim(min_time - time_padding, max_time + time_padding)
        
        # Create a background grid for time periods
        for t in range(int(min_time) - int(min_time) % 60, int(max_time) + 60, 60):
            ax.axvline(x=t, linestyle='-', color='#e0e0e0', alpha=0.5, zorder=1)
            # Add minute markers
            if t >= 0:
                minutes = t // 60
                ax.text(t, -0.5, f"{minutes}'", ha='center', va='top', 
                       fontsize=10, color='#666666', zorder=2)
        
        # Generate event positions with lanes for different event types
        # Use different y positions for clarity
        event_lanes = {'offside': 0, 'goal': 1, 'key_pass': 2}
        team_offset = {0: 0.2, 1: -0.2}  # Team 1 above, Team 2 below
        
        # Colors, markers and sizes for events
        event_colors = {'offside': self.team_colors, 'goal': ['#e53935', '#d32f2f'], 'key_pass': ['#43a047', '#2e7d32']}
        event_markers = {'offside': 'o', 'goal': '*', 'key_pass': 's'}
        event_sizes = {'offside': 100, 'goal': 300, 'key_pass': 100}
        
        # Setup the y-axis for lanes
        ax.set_ylim(-1, len(event_lanes))
        
        # Draw events
        offside_times = []
        goal_times = []
        keypass_times = []
        
        # First pass to collect event times by type
        for event in events:
            event_type = event['type']
            if event_type == 'offside':
                offside_times.append(event['timestamp'])
            elif event_type == 'goal':
                goal_times.append(event['timestamp'])
            elif event_type == 'key_pass':
                keypass_times.append(event['timestamp'])
        
        # Draw semi-transparent bands for event clusters
        def draw_event_bands(event_times, y_pos, color, alpha=0.15):
            """Draw background bands for clusters of similar events"""
            if len(event_times) < 2:
                return
                
            # Sort times
            sorted_times = sorted(event_times)
            
            # Find clusters (events within 30 seconds of each other)
            clusters = []
            current_cluster = [sorted_times[0]]
            
            for i in range(1, len(sorted_times)):
                if sorted_times[i] - sorted_times[i-1] < 30:
                    current_cluster.append(sorted_times[i])
                else:
                    if len(current_cluster) >= 2:
                        clusters.append(current_cluster)
                    current_cluster = [sorted_times[i]]
            
            # Add the last cluster if needed
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)
            
            # Draw bands for clusters
            height = 0.7
            for cluster in clusters:
                start_time = min(cluster) - 5
                end_time = max(cluster) + 5
                width = end_time - start_time
                rect = patches.Rectangle(
                    (start_time, y_pos - height/2), 
                    width, height, 
                    facecolor=color, alpha=alpha, 
                    edgecolor='none', zorder=1
                )
                ax.add_patch(rect)
        
        # Draw background bands
        draw_event_bands(offside_times, event_lanes['offside'], '#90caf9')
        draw_event_bands(goal_times, event_lanes['goal'], '#ef9a9a')  
        draw_event_bands(keypass_times, event_lanes['key_pass'], '#a5d6a7')
        
        # Draw horizontal separators between lanes
        for y in range(len(event_lanes)):
            ax.axhline(y=y+0.5, linestyle='-', color='#e0e0e0', alpha=0.7, zorder=1)
        
        # Second pass to plot the actual events
        for event in events:
            event_type = event['type']
            team = event.get('team', 0)
            timestamp = event['timestamp']
            
            # Calculate y position (lane + team offset)
            y_pos = event_lanes[event_type] + team_offset[team]
            
            # Draw marker
            ax.scatter(
                timestamp, y_pos,
                marker=event_markers[event_type],
                s=event_sizes[event_type],
                color=event_colors[event_type][team],
                edgecolor='white',
                linewidth=1.5,
                alpha=0.9,
                zorder=4
            )
            
            # Add match time label for goals
            if event_type == 'goal':
                match_time = event.get('match_time', self.format_time(timestamp))
                ax.text(
                    timestamp, y_pos + 0.3,
                    f"GOAL! {match_time}",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold',
                    color=event_colors[event_type][team],
                    zorder=5
                )
            
            # Add position indicator line for offsides
            if event_type == 'offside' and 'position' in event:
                # Show small annotation for offside position
                position_str = f"({event['position'][0]:.1f}, {event['position'][1]:.1f})"
                ax.text(
                    timestamp, y_pos + 0.2,
                    position_str,
                    ha='center', va='bottom',
                    fontsize=8, color='#666666',
                    zorder=5
                )
        
        # Setup y-axis labels
        ax.set_yticks([lane for lane in event_lanes.values()])
        ax.set_yticklabels([label.replace('_', ' ').title() for label in event_lanes.keys()])
        
        # Add event count labels to y-axis
        for event_type, lane in event_lanes.items():
            count = len([e for e in events if e['type'] == event_type])
            if count > 0:
                ax.text(
                    ax.get_xlim()[0] - time_padding * 0.2, lane,
                    f"({count})",
                    ha='right', va='center',
                    fontsize=10, color='#666666'
                )
        
        # Add title and labels
        ax.set_title(title, fontsize=18, pad=20)
        ax.set_xlabel('Time (seconds)', fontsize=14)
        
        # Add legend for teams
        from matplotlib.lines import Line2D
        team_legend = [
            Line2D([0], [0], color=self.team_colors[0], marker='o', linestyle='none',
                  markersize=10, label='Team 1'),
            Line2D([0], [0], color=self.team_colors[1], marker='o', linestyle='none',
                  markersize=10, label='Team 2')
        ]
        ax.legend(handles=team_legend, loc='upper right')
        
        # Add context summary box
        team0_offsides = len([e for e in events if e['type'] == 'offside' and e['team'] == 0])
        team1_offsides = len([e for e in events if e['type'] == 'offside' and e['team'] == 1])
        team0_goals = len([e for e in events if e['type'] == 'goal' and e['team'] == 0])
        team1_goals = len([e for e in events if e['type'] == 'goal' and e['team'] == 1])
        
        summary_text = (
            f"Score: Team 1 {team0_goals} - {team1_goals} Team 2\n"
            f"Offsides: Team 1 {team0_offsides}, Team 2 {team1_offsides}\n"
            f"Duration: {self.format_time(max_time)}"
        )
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(
            0.02, 0.02, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(self.viz_dir, save_filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Match timeline saved to: {filepath}")
        
        return fig
    
    def format_time(self, seconds):
        """Format seconds as MM:SS"""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes:02d}:{secs:02d}"
    
    def generate_team_comparison(self, 
                               title: str = "Team Offside Comparison", 
                               save_filename: str = "team_comparison.png") -> Figure:
        """
        Generate a comparison of offside statistics between teams
        
        Args:
            title: Title of the plot
            save_filename: Filename to save the plot to
            
        Returns:
            Matplotlib Figure object
        """
        # Get team offside counts
        team0_count = self.analytics.get_team_offsides(0)
        team1_count = self.analytics.get_team_offsides(1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bars
        team_labels = ['Team 1', 'Team 2']
        offside_counts = [team0_count, team1_count]
        bars = ax.bar(team_labels, offside_counts, color=self.team_colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                   f"{height:.0f}", ha='center', va='bottom', fontsize=12)
        
        # Add title and labels
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_ylabel('Number of Offsides', fontsize=14)
        
        # Adjust y-axis to start from 0
        ax.set_ylim(0, max(offside_counts) * 1.2)
        
        # Add grid
        ax.grid(True, axis='y', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(self.viz_dir, save_filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Team comparison chart saved to: {filepath}")
        
        return fig
    
    def generate_animated_movement_trails(self, 
                                        event_id: int = 0, 
                                        save_filename: str = "movement_trail.gif"):
        """
        Generate an animated visualization of player movement before an offside
        
        Args:
            event_id: Index of the offside event to visualize
            save_filename: Filename to save the animation to
            
        Returns:
            Path to the saved GIF file
        """
        # This is a placeholder for the animated movement trails feature
        # In a real implementation, this would use player tracking data to create an animation
        
        print("Animated movement trails feature requires player tracking data.")
        print("This is a placeholder for future implementation.")
        
        # Return the filepath that would be used
        return os.path.join(self.viz_dir, save_filename)
    
    def generate_dashboard(self, save_filename: str = "dashboard.png") -> Figure:
        """
        Generate a comprehensive dashboard with all visualizations
        
        Args:
            save_filename: Filename to save the dashboard to
            
        Returns:
            Matplotlib Figure object
        """
        # Create a large figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Define grid layout
        gs = fig.add_gridspec(3, 3)
        
        # Heatmap (larger, spans two columns)
        ax_heatmap = fig.add_subplot(gs[0, :2])
        self.draw_pitch(ax_heatmap)
        
        # Create a smoother heatmap by upsampling
        heatmap_smooth = cv2.resize(self.analytics.heatmap_data, (100, 65), 
                                   interpolation=cv2.INTER_CUBIC)
        
        # Plot heatmap
        im = ax_heatmap.imshow(heatmap_smooth, cmap=self.heatmap_cmap, alpha=0.7,
                              extent=[0, self.analytics.pitch_dimensions[0], 
                                     0, self.analytics.pitch_dimensions[1]],
                              origin='lower', zorder=3)
        ax_heatmap.set_title('Offside Heatmap', fontsize=14)
        
        # Player offsides bar chart
        ax_player = fig.add_subplot(gs[0, 2])
        
        # Get player stats
        player_stats = self.analytics.player_stats
        
        # Sort players by offside count
        sorted_players = sorted(player_stats.items(), 
                               key=lambda x: x[1]['count'], reverse=True)
        
        # Take top players
        top_players = sorted_players[:5]  # Fewer players for the dashboard
        
        # Create lists for plotting
        player_ids = [f"Player {p[0]}" for p in top_players]
        offside_counts = [p[1]['count'] for p in top_players]
        player_teams = [p[1]['team'] for p in top_players]
        colors = [self.team_colors[team] for team in player_teams]
        
        # Plot horizontal bar chart
        bars = ax_player.barh(player_ids, offside_counts, color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax_player.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                          f"{width:.0f}", ha='left', va='center', fontsize=10)
        
        ax_player.set_title('Player Offside Frequency', fontsize=14)
        
        # Team comparison pie chart
        ax_team = fig.add_subplot(gs[1, 0])
        
        # Get team offside counts
        team0_count = self.analytics.get_team_offsides(0)
        team1_count = self.analytics.get_team_offsides(1)
        
        # Plot pie chart
        team_labels = ['Team 1', 'Team 2']
        offside_counts = [team0_count, team1_count]
        ax_team.pie(offside_counts, labels=team_labels, colors=self.team_colors,
                   autopct='%1.1f%%', startangle=90)
        ax_team.set_title('Team Offside Distribution', fontsize=14)
        
        # Zone behavior
        ax_zone = fig.add_subplot(gs[1, 1:])
        self.draw_pitch(ax_zone)
        
        # Get zone stats
        zone_stats = self.analytics.zone_stats
        
        # Define zone centers based on the analytics zone definitions
        zone_centers = {}
        pitch_length, pitch_width = self.analytics.pitch_dimensions
        
        # Calculate the center of each zone
        for zone_name, ((x_min, x_max), (y_min, y_max)) in self.analytics.zones.items():
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            zone_centers[zone_name] = (center_x, center_y)
        
        # Plot circles for each zone, with size proportional to offside count
        max_count = max([stats['total_count'] for stats in zone_stats.values()], default=1)
        
        for zone_name, stats in zone_stats.items():
            if zone_name in zone_centers:
                center_x, center_y = zone_centers[zone_name]
                
                # Calculate circle size based on offside count
                total_count = stats['total_count']
                radius = 3 * (total_count / max_count) + 1  # Min size of 1
                
                # Calculate team proportions
                team0_prop = stats['team0_count'] / total_count if total_count > 0 else 0
                team1_prop = stats['team1_count'] / total_count if total_count > 0 else 0
                
                # Draw team 0 portion (as a pie chart)
                team0_circle = patches.Wedge(
                    (center_x, center_y), radius, 0, 360 * team0_prop,
                    facecolor=self.team_colors[0], alpha=0.7, zorder=3
                )
                ax_zone.add_patch(team0_circle)
                
                # Draw team 1 portion
                team1_circle = patches.Wedge(
                    (center_x, center_y), radius, 360 * team0_prop, 360,
                    facecolor=self.team_colors[1], alpha=0.7, zorder=3
                )
                ax_zone.add_patch(team1_circle)
                
                # Add count text if significant
                if total_count > max_count / 10:
                    ax_zone.text(center_x, center_y, str(total_count),
                                ha='center', va='center', fontsize=8, 
                                fontweight='bold', color='white', zorder=4)
        
        ax_zone.set_title('Zone-Based Team Behavior', fontsize=14)
        
        # Timeline
        ax_timeline = fig.add_subplot(gs[2, :])
        
        # Sort events by timestamp
        events = sorted(self.analytics.timeline_events, key=lambda x: x['timestamp'])
        
        if events:
            # Extract timestamps
            timestamps = [event['timestamp'] for event in events]
            
            # Create y-positions for events (stagger them for visibility)
            y_positions = []
            y_base = {'offside': 1, 'goal': 2, 'key_pass': 0}
            for event in events:
                y_positions.append(y_base[event['type']] + (0.1 if event['team'] == 1 else -0.1))
            
            # Create markers and colors for different event types
            markers = {'offside': 'o', 'goal': '*', 'key_pass': 's'}
            sizes = {'offside': 60, 'goal': 150, 'key_pass': 60}
            
            # Plot events
            for i, event in enumerate(events):
                ax_timeline.scatter(event['timestamp'], y_positions[i], 
                                   marker=markers[event['type']], 
                                   s=sizes[event['type']], 
                                   color=self.team_colors[event['team']],
                                   alpha=0.7, 
                                   edgecolor='black',
                                   linewidth=1,
                                   zorder=3)
            
            # Add horizontal lines to separate event types
            ax_timeline.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, zorder=1)
            ax_timeline.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5, zorder=1)
            
            # Set y-axis labels and ticks
            ax_timeline.set_yticks([0, 1, 2])
            ax_timeline.set_yticklabels(['Key Passes', 'Offsides', 'Goals'])
            
            # Set x-axis labels
            ax_timeline.set_xlabel('Time (seconds)', fontsize=12)
            
            # Set x limits with some padding
            if timestamps:
                ax_timeline.set_xlim(min(timestamps) - 5, max(timestamps) + 5)
        
        ax_timeline.set_title('Match Timeline', fontsize=14)
        
        # Add main title
        fig.suptitle('Offside Analysis Dashboard', fontsize=20, y=0.98)
        
        # Adjust spacing
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        filepath = os.path.join(self.viz_dir, save_filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Dashboard saved to: {filepath}")
        
        return fig
    
    def generate_all_visualizations(self):
        """Generate all available visualizations and save them"""
        # Generate individual visualizations
        self.generate_offside_heatmap()
        self.generate_player_offside_chart()
        self.generate_zone_team_behavior()
        self.generate_match_timeline()
        self.generate_team_comparison()
        
        # Generate comprehensive dashboard
        self.generate_dashboard()
        
        print(f"All visualizations saved to: {self.viz_dir}")
    
    def export_report_pdf(self, filename: str = "offside_report.pdf"):
        """
        Export a PDF report with all visualizations and analysis
        
        Args:
            filename: Name of the PDF file to save
        """
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            
            # Generate all visualizations first
            self.generate_all_visualizations()
            
            # Create PDF document
            pdf_path = os.path.join(self.output_dir, filename)
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = styles['Title']
            heading_style = styles['Heading1']
            normal_style = styles['Normal']
            
            # Create content
            content = []
            
            # Title
            content.append(Paragraph("Offside Analysis Report", title_style))
            content.append(Spacer(1, 0.25*inch))
            
            # Date
            content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
            content.append(Spacer(1, 0.25*inch))
            
            # Summary statistics
            content.append(Paragraph("Summary Statistics", heading_style))
            content.append(Spacer(1, 0.1*inch))
            
            # Create summary table
            data = [
                ["Total Offsides", str(len(self.analytics.offside_events))],
                ["Team 1 Offsides", str(self.analytics.get_team_offsides(0))],
                ["Team 2 Offsides", str(self.analytics.get_team_offsides(1))]
            ]
            
            summary_table = Table(data, colWidths=[2*inch, 1*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(summary_table)
            content.append(Spacer(1, 0.25*inch))
            
            # Add visualizations
            viz_files = [
                ('offside_heatmap.png', 'Offside Heatmap'),
                ('player_offsides.png', 'Player Offside Frequency'),
                ('zone_behavior.png', 'Zone-Based Team Behavior'),
                ('match_timeline.png', 'Match Timeline'),
                ('team_comparison.png', 'Team Offside Comparison'),
                ('dashboard.png', 'Analysis Dashboard')
            ]
            
            for viz_file, viz_title in viz_files:
                file_path = os.path.join(self.viz_dir, viz_file)
                if os.path.exists(file_path):
                    content.append(Paragraph(viz_title, heading_style))
                    content.append(Spacer(1, 0.1*inch))
                    
                    # Add image, scaled to fit the page width
                    img = Image(file_path, width=6*inch, height=4*inch)
                    content.append(img)
                    content.append(Spacer(1, 0.25*inch))
            
            # Build PDF
            doc.build(content)
            
            print(f"PDF report saved to: {pdf_path}")
            return pdf_path
            
        except ImportError:
            print("ReportLab not installed. Please install it to generate PDF reports.")
            print("You can install it with: pip install reportlab")
            return None
    
    def generate_efficiency_chart(self, 
                                title: str = "Attacking Efficiency Analysis", 
                                save_filename: str = "efficiency_chart.png") -> Figure:
        """
        Generate a chart showing the relationship between offsides and goals
        
        Args:
            title: Title of the plot
            save_filename: Filename to save the plot to
            
        Returns:
            Matplotlib Figure object
        """
        # Get efficiency stats
        stats = self.analytics.get_efficiency_stats()
        
        # Create figure with grid for multiple plots
        fig = plt.figure(figsize=(14, 10))
        gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[2, 1])
        
        # 1. Bar chart comparing offsides to goals
        ax_bars = fig.add_subplot(gs[0, 0])
        
        # Data for bar chart
        teams = ['Team 1', 'Team 2']
        offside_counts = [stats['team0']['offsides'], stats['team1']['offsides']]
        goal_counts = [stats['team0']['goals'], stats['team1']['goals']]
        key_pass_counts = [stats['team0']['key_passes'], stats['team1']['key_passes']]
        
        # Set up bar positions
        bar_width = 0.25
        pos1 = np.arange(len(teams))
        pos2 = [p + bar_width for p in pos1]
        pos3 = [p + bar_width for p in pos2]
        
        # Plot bars
        ax_bars.bar(pos1, offside_counts, bar_width, color='#e57373', label='Offsides')
        ax_bars.bar(pos2, goal_counts, bar_width, color='#81c784', label='Goals')
        ax_bars.bar(pos3, key_pass_counts, bar_width, color='#64b5f6', label='Key Passes')
        
        # Add labels and title
        ax_bars.set_title('Offsides vs Goals vs Key Passes', fontsize=14)
        ax_bars.set_xticks([p + bar_width for p in range(len(teams))])
        ax_bars.set_xticklabels(teams)
        ax_bars.legend()
        
        # Add value labels to bars
        for i, count in enumerate(offside_counts):
            ax_bars.text(pos1[i], count + 0.1, str(count), ha='center', va='bottom')
        for i, count in enumerate(goal_counts):
            ax_bars.text(pos2[i], count + 0.1, str(count), ha='center', va='bottom')
        for i, count in enumerate(key_pass_counts):
            ax_bars.text(pos3[i], count + 0.1, str(count), ha='center', va='bottom')
        
        # 2. Efficiency ratio chart
        ax_ratio = fig.add_subplot(gs[0, 1])
        
        # Calculate efficiency (goals per offside)
        efficiencies = [stats['team0']['efficiency'], stats['team1']['efficiency']]
        
        # Create a colormap
        cmap = plt.cm.get_cmap('RdYlGn')
        colors = [cmap(e) for e in efficiencies]
        
        # Create horizontal bar chart
        bars = ax_ratio.barh(teams, efficiencies, color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax_ratio.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                         f"{width:.2f}", ha='left', va='center', fontsize=12)
        
        # Set title and labels
        ax_ratio.set_title('Goals per Offside Ratio', fontsize=14)
        ax_ratio.set_xlabel('Efficiency Ratio')
        
        # 3. Scatter plot of offsides vs goals over time
        ax_scatter = fig.add_subplot(gs[1, :])
        
        # Get offside events
        offside_events = sorted(self.analytics.offside_events, key=lambda x: x.timestamp)
        team0_offsides = [e for e in offside_events if e.team == 0]
        team1_offsides = [e for e in offside_events if e.team == 1]
        
        # Get goal events
        goal_events = sorted(self.analytics.goal_events, key=lambda x: x['timestamp'])
        team0_goals = [g for g in goal_events if g['team'] == 0]
        team1_goals = [g for g in goal_events if g['team'] == 1]
        
        # Plot offside events
        if team0_offsides:
            ax_scatter.scatter(
                [e.timestamp for e in team0_offsides],
                [1] * len(team0_offsides),
                marker='o',
                color=self.team_colors[0],
                alpha=0.6,
                s=80,
                label='Team 1 Offsides'
            )
        
        if team1_offsides:
            ax_scatter.scatter(
                [e.timestamp for e in team1_offsides],
                [0.9] * len(team1_offsides),
                marker='o',
                color=self.team_colors[1],
                alpha=0.6,
                s=80,
                label='Team 2 Offsides'
            )
        
        # Plot goal events
        if team0_goals:
            ax_scatter.scatter(
                [g['timestamp'] for g in team0_goals],
                [0.7] * len(team0_goals),
                marker='*',
                color=self.team_colors[0],
                s=200,
                label='Team 1 Goals'
            )
        
        if team1_goals:
            ax_scatter.scatter(
                [g['timestamp'] for g in team1_goals],
                [0.6] * len(team1_goals),
                marker='*',
                color=self.team_colors[1],
                s=200,
                label='Team 2 Goals'
            )
        
        # Set up the timeline
        if offside_events or goal_events:
            all_timestamps = []
            if offside_events:
                all_timestamps.extend([e.timestamp for e in offside_events])
            if goal_events:
                all_timestamps.extend([g['timestamp'] for g in goal_events])
            
            min_time = min(all_timestamps)
            max_time = max(all_timestamps)
            
            # Add some padding
            time_padding = (max_time - min_time) * 0.05
            ax_scatter.set_xlim(min_time - time_padding, max_time + time_padding)
        
        # Set up y-axis
        ax_scatter.set_yticks([0.65, 0.75, 0.95])
        ax_scatter.set_yticklabels(['Goals', '', 'Offsides'])
        ax_scatter.set_ylim(0.5, 1.1)
        
        # Add vertical lines for goals
        if goal_events:
            for goal in goal_events:
                color = self.team_colors[goal['team']]
                ax_scatter.axvline(
                    x=goal['timestamp'],
                    color=color,
                    linestyle='--',
                    alpha=0.4,
                    zorder=1
                )
                # Add goal annotation
                team_name = "Team 1" if goal['team'] == 0 else "Team 2"
                ax_scatter.text(
                    goal['timestamp'],
                    1.05,
                    f"Goal: {team_name}\n{goal['match_time']}",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    rotation=90,
                    color=color
                )
        
        # Add title and labels
        ax_scatter.set_title('Offside & Goal Timeline', fontsize=14)
        ax_scatter.set_xlabel('Time (seconds)')
        ax_scatter.legend(loc='upper right')
        
        # Add game score overlay
        team0_score = self.analytics.get_team_goals(0)
        team1_score = self.analytics.get_team_goals(1)
        score_text = f"Score: Team 1 {team0_score} - {team1_score} Team 2"
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax_scatter.text(
            0.5, 0.05, score_text, transform=ax_scatter.transAxes,
            fontsize=12, ha='center', va='center', bbox=props
        )
        
        # Add summary insights
        summary_text = self._generate_efficiency_insights()
        ax_bars.text(
            -0.1, -0.3, summary_text, transform=ax_bars.transAxes,
            fontsize=10, ha='left', va='top', bbox=props
        )
        
        # Add main title
        fig.suptitle(title, fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        filepath = os.path.join(self.viz_dir, save_filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Efficiency chart saved to: {filepath}")
        
        return fig
    
    def _generate_efficiency_insights(self) -> str:
        """Generate textual insights about efficiency metrics"""
        stats = self.analytics.get_efficiency_stats()
        team0_eff = stats['team0']['efficiency']
        team1_eff = stats['team1']['efficiency']
        
        insights = "Efficiency Analysis:\n"
        
        # Compare efficiencies
        if team0_eff > 0 and team1_eff > 0:
            if team0_eff > team1_eff * 1.5:
                insights += "• Team 1 is significantly more efficient at converting chances to goals.\n"
            elif team1_eff > team0_eff * 1.5:
                insights += "• Team 2 is significantly more efficient at converting chances to goals.\n"
            else:
                insights += "• Both teams show similar efficiency in converting chances.\n"
        
        # Key passes to goals ratio
        team0_key_passes = stats['team0']['key_passes']
        team1_key_passes = stats['team0']['key_passes']
        team0_goals = stats['team0']['goals']
        team1_goals = stats['team1']['goals']
        
        if team0_key_passes > 0 and team0_goals > 0:
            insights += f"• Team 1 creates {team0_key_passes/team0_goals:.1f} key passes per goal.\n"
        
        if team1_key_passes > 0 and team1_goals > 0:
            insights += f"• Team 2 creates {team1_key_passes/team1_goals:.1f} key passes per goal.\n"
        
        # High offside count but low goals
        if stats['team0']['offsides'] > 3 and stats['team0']['goals'] == 0:
            insights += "• Team 1 has several offsides but no goals, suggesting poor finishing or timing issues.\n"
        
        if stats['team1']['offsides'] > 3 and stats['team1']['goals'] == 0:
            insights += "• Team 2 has several offsides but no goals, suggesting poor finishing or timing issues.\n"
        
        return insights 