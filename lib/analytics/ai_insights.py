"""
AI Insights Module for VAROMATIC+
Provides automated insights and suggestions based on analytics data
"""

from typing import List, Dict, Tuple, Optional, Any
from .offside_analytics import OffsideAnalytics

class InsightGenerator:
    """Generates AI-powered insights from offside analytics data"""
    
    def __init__(self, analytics: OffsideAnalytics):
        """
        Initialize the insight generator
        
        Args:
            analytics: OffsideAnalytics instance with collected data
        """
        self.analytics = analytics
    
    def generate_insights(self, min_offsides: int = 2) -> List[str]:
        """
        Generate insights based on the analytics data
        
        Args:
            min_offsides: Minimum number of offsides required for certain insights
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Add team-level insights
        team_insights = self._generate_team_insights(min_offsides)
        insights.extend(team_insights)
        
        # Add player-level insights
        player_insights = self._generate_player_insights(min_offsides)
        insights.extend(player_insights)
        
        # Add zone-based insights
        zone_insights = self._generate_zone_insights(min_offsides)
        insights.extend(zone_insights)
        
        # Add timing-based insights
        timing_insights = self._generate_timing_insights(min_offsides)
        insights.extend(timing_insights)
        
        # Add tactical insights
        tactical_insights = self._generate_tactical_insights(min_offsides)
        insights.extend(tactical_insights)
        
        # Add defensive line insights
        defensive_insights = self._generate_defensive_insights(min_offsides)
        insights.extend(defensive_insights)
        
        return insights
    
    def _generate_team_insights(self, min_offsides: int) -> List[str]:
        """Generate insights related to team performance"""
        insights = []
        
        # Team comparison insights
        team0_count = self.analytics.get_team_offsides(0)
        team1_count = self.analytics.get_team_offsides(1)
        
        if team0_count >= min_offsides or team1_count >= min_offsides:
            # Compare team offside frequencies
            if team0_count > team1_count * 1.5 and team0_count >= min_offsides:
                insights.append(f"Team 1 has significantly more offsides ({team0_count}) than Team 2 ({team1_count}), suggesting they may be playing a more aggressive attacking style but need to work on timing their forward runs better.")
            elif team1_count > team0_count * 1.5 and team1_count >= min_offsides:
                insights.append(f"Team 2 has significantly more offsides ({team1_count}) than Team 1 ({team0_count}), suggesting they may be playing a more aggressive attacking style but need to work on timing their forward runs better.")
            
            # Total offside insights
            total_offsides = team0_count + team1_count
            if total_offsides >= min_offsides * 3:
                insights.append(f"High number of offsides detected ({total_offsides}) in this match, suggesting either aggressive attacking tactics from both teams or potentially highly effective defensive offside traps.")
            
            # Team with few offsides
            if team0_count <= 1 and team1_count >= min_offsides * 2:
                insights.append(f"Team 1 has very few offsides ({team0_count}), which could indicate either conservative attacking play or excellent timing on forward runs.")
            elif team1_count <= 1 and team0_count >= min_offsides * 2:
                insights.append(f"Team 2 has very few offsides ({team1_count}), which could indicate either conservative attacking play or excellent timing on forward runs.")
        
        return insights
    
    def _generate_player_insights(self, min_offsides: int) -> List[str]:
        """Generate insights related to individual players"""
        insights = []
        
        # Player with most offsides
        player_stats = self.analytics.player_stats
        if player_stats:
            sorted_players = sorted(player_stats.items(), key=lambda x: x[1]['count'], reverse=True)
            
            # Identify the players with the most offsides
            if sorted_players and sorted_players[0][1]['count'] >= min_offsides:
                top_player_id, top_player_stats = sorted_players[0]
                team_name = "Team 1" if top_player_stats['team'] == 0 else "Team 2"
                insights.append(f"Player {top_player_id} ({team_name}) is caught offside most frequently ({top_player_stats['count']} times), suggesting they may need to work on timing their runs better or be more aware of the defensive line.")
            
            # Compare top offside players by team
            team0_players = [p for p in sorted_players if p[1]['team'] == 0]
            team1_players = [p for p in sorted_players if p[1]['team'] == 1]
            
            if team0_players and team0_players[0][1]['count'] >= min_offsides:
                top_team0_player = team0_players[0]
                insights.append(f"For Team 1, Player {top_team0_player[0]} has the most offsides ({top_team0_player[1]['count']}), suggesting they might be the primary forward or making the most attacking runs.")
            
            if team1_players and team1_players[0][1]['count'] >= min_offsides:
                top_team1_player = team1_players[0]
                insights.append(f"For Team 2, Player {top_team1_player[0]} has the most offsides ({top_team1_player[1]['count']}), suggesting they might be the primary forward or making the most attacking runs.")
            
            # Analyze players who are repeatedly caught offside
            repeat_offenders = [p for p in sorted_players if p[1]['count'] >= min_offsides * 2]
            if repeat_offenders:
                player_id, player_data = repeat_offenders[0]
                team_name = "Team 1" if player_data['team'] == 0 else "Team 2"
                insights.append(f"Player {player_id} ({team_name}) is repeatedly caught offside ({player_data['count']} times), suggesting they may need coaching on positioning or need to adjust their attacking approach.")
            
            # Analyze player positions for offside patterns
            if len(sorted_players) >= 2:
                # Collect all positions from players with multiple offsides
                offside_positions = []
                for player_id, stats in sorted_players:
                    if stats['count'] >= min_offsides:
                        offside_positions.extend(stats['positions'])
                
                if offside_positions:
                    # Analyze typical offside positions
                    avg_x = sum(pos[0] for pos in offside_positions) / len(offside_positions)
                    avg_y = sum(pos[1] for pos in offside_positions) / len(offside_positions)
                    
                    # Determine field location
                    pitch_length, pitch_width = self.analytics.pitch_dimensions
                    
                    # Determine attacking third/flank based on coordinates
                    third_desc = ""
                    if avg_x < pitch_length * 0.33:
                        third_desc = "defensive third"
                    elif avg_x < pitch_length * 0.66:
                        third_desc = "middle third"
                    else:
                        third_desc = "attacking third"
                    
                    flank_desc = ""
                    if avg_y < pitch_width * 0.33:
                        flank_desc = "left flank"
                    elif avg_y < pitch_width * 0.66:
                        flank_desc = "central area"
                    else:
                        flank_desc = "right flank"
                    
                    insights.append(f"Most offsides occur in the {third_desc} on the {flank_desc} of the pitch, suggesting tactical adjustments may be needed in this area. Players should be more conscious of the defensive line when attacking through this zone.")
        
        return insights
    
    def _generate_zone_insights(self, min_offsides: int) -> List[str]:
        """Generate insights related to field zones"""
        insights = []
        
        # Zone with most offsides
        zone_stats = self.analytics.zone_stats
        if zone_stats:
            sorted_zones = sorted(zone_stats.items(), key=lambda x: x[1]['total_count'], reverse=True)
            
            # Analyze top offside zone
            if sorted_zones and sorted_zones[0][1]['total_count'] >= min_offsides:
                top_zone, top_zone_stats = sorted_zones[0]
                
                # Determine which team has more offsides in this zone
                team0_count = top_zone_stats['team0_count']
                team1_count = top_zone_stats['team1_count']
                
                zone_name = top_zone.replace('_', ' ').title()
                
                if team0_count > team1_count * 1.5 and team0_count >= min_offsides:
                    insights.append(f"Team 1 is frequently caught offside in the {zone_name} zone ({team0_count} times). This could indicate either a weakness in their attacking strategy or an effective offside trap being set by Team 2 in this area.")
                elif team1_count > team0_count * 1.5 and team1_count >= min_offsides:
                    insights.append(f"Team 2 is frequently caught offside in the {zone_name} zone ({team1_count} times). This could indicate either a weakness in their attacking strategy or an effective offside trap being set by Team 1 in this area.")
                elif team0_count >= min_offsides and team1_count >= min_offsides:
                    insights.append(f"Both teams are frequently caught offside in the {zone_name} zone, suggesting this area might be tactically challenging for maintaining the offside line. Defenders might be effectively implementing offside traps in this region.")
            
            # Analyze patterns in zone distribution
            if len(sorted_zones) >= 3:
                top_3_zones = [zone for zone, stats in sorted_zones[:3] if stats['total_count'] >= min_offsides]
                
                # Check for patterns in zone locations
                attacking_zones = sum(1 for zone in top_3_zones if 'attacking' in zone)
                middle_zones = sum(1 for zone in top_3_zones if 'middle' in zone)
                defensive_zones = sum(1 for zone in top_3_zones if 'defensive' in zone)
                
                left_zones = sum(1 for zone in top_3_zones if 'left' in zone)
                center_zones = sum(1 for zone in top_3_zones if 'center' in zone)
                right_zones = sum(1 for zone in top_3_zones if 'right' in zone)
                
                # Generate insights based on zone patterns
                if attacking_zones >= 2:
                    insights.append("Most offsides occur in the attacking third, suggesting teams need to be more aware of the defensive line when making final runs into the box. Attackers should focus on staying level with defenders until the pass is played.")
                
                if middle_zones >= 2:
                    insights.append("A significant number of offsides occur in the middle third, which could indicate issues with through-ball timing or counter-attacking transitions. Better communication between passers and runners could improve this.")
                
                # Flank analysis
                if left_zones >= 2:
                    insights.append("The left flank shows a high concentration of offsides, suggesting attacking patterns or defensive line positioning issues on this side. Attackers on this flank should work on their timing or consider adjusting their runs.")
                elif right_zones >= 2:
                    insights.append("The right flank shows a high concentration of offsides, suggesting attacking patterns or defensive line positioning issues on this side. Attackers on this flank should work on their timing or consider adjusting their runs.")
                elif center_zones >= 2:
                    insights.append("Central areas have the most offsides, indicating that teams are trying to play through the middle but struggling with timing. More diagonal runs from wide positions might be more effective.")
        
        return insights
    
    def _generate_timing_insights(self, min_offsides: int) -> List[str]:
        """Generate insights related to timing of offsides"""
        insights = []
        
        # Only proceed if we have enough events
        events = sorted(self.analytics.timeline_events, key=lambda x: x['timestamp'])
        offside_events = [e for e in events if e['type'] == 'offside']
        
        if len(offside_events) >= min_offsides:
            # Check for clusters of offsides
            offside_timestamps = [e['timestamp'] for e in offside_events]
            clusters = []
            
            if offside_timestamps:
                current_cluster = [offside_timestamps[0]]
                
                # Identify clusters of offsides occurring within 3 minutes of each other
                for i in range(1, len(offside_timestamps)):
                    if offside_timestamps[i] - offside_timestamps[i-1] < 180:  # 3 minutes in seconds
                        current_cluster.append(offside_timestamps[i])
                    else:
                        if len(current_cluster) >= 2:
                            clusters.append(current_cluster.copy())
                        current_cluster = [offside_timestamps[i]]
                
                # Add the last cluster if it has at least 2 events
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
            
            # Generate insights based on clusters
            if clusters:
                largest_cluster = max(clusters, key=len)
                if len(largest_cluster) >= 3:
                    # Calculate average time of the cluster
                    avg_time = sum(largest_cluster) / len(largest_cluster)
                    # Convert to minutes for readability
                    minutes = int(avg_time / 60)
                    
                    cluster_team_counts = self._get_team_counts_for_timestamps(largest_cluster)
                    
                    if cluster_team_counts[0] > cluster_team_counts[1] * 2:
                        team_str = "Team 1"
                    elif cluster_team_counts[1] > cluster_team_counts[0] * 2:
                        team_str = "Team 2"
                    else:
                        team_str = "Both teams"
                    
                    insights.append(f"A significant cluster of {len(largest_cluster)} offsides was detected around the {minutes}-minute mark, primarily from {team_str}. This suggests a period of high attacking pressure or possibly a tactical adjustment to the defensive line that attackers haven't adapted to yet.")
            
            # Analyze match periods
            if len(offside_events) >= 4:
                # Split into first and second half
                mid_point = offside_events[len(offside_events) // 2]['timestamp']
                first_half = [e for e in offside_events if e['timestamp'] < mid_point]
                second_half = [e for e in offside_events if e['timestamp'] >= mid_point]
                
                # Compare first and second half frequencies
                if len(first_half) > len(second_half) * 2:
                    insights.append(f"Offsides are much more frequent in the first part of the match ({len(first_half)} vs {len(second_half)}), suggesting teams may be adjusting their tactics as the game progresses.")
                elif len(second_half) > len(first_half) * 2:
                    insights.append(f"Offsides are increasing significantly in the latter part of the match ({len(second_half)} vs {len(first_half)}), possibly due to fatigue affecting positioning discipline or more aggressive attacking as the game progresses.")
            
            # Check for offsides after key events like goals
            goal_events = [e for e in events if e['type'] == 'goal']
            
            for goal in goal_events:
                # Look for offsides within 5 minutes after a goal
                post_goal_offsides = [
                    e for e in offside_events 
                    if goal['timestamp'] < e['timestamp'] < goal['timestamp'] + 300
                ]
                
                if len(post_goal_offsides) >= 2:
                    scoring_team = "Team 1" if goal['team'] == 0 else "Team 2"
                    
                    # Check which team has more offsides after the goal
                    post_goal_team_counts = self._get_team_counts_for_events(post_goal_offsides)
                    
                    if post_goal_team_counts[goal['team']] > post_goal_team_counts[1 - goal['team']]:
                        # Scoring team has more offsides
                        insights.append(f"After {scoring_team}'s goal, they were caught offside {post_goal_team_counts[goal['team']]} times, suggesting they may be pushing forward more aggressively with their lead.")
                    else:
                        # Conceding team has more offsides
                        conceding_team = "Team 2" if goal['team'] == 0 else "Team 1"
                        insights.append(f"After conceding to {scoring_team}, {conceding_team} was caught offside {post_goal_team_counts[1 - goal['team']]} times, suggesting they may be pushing forward more urgently to equalize.")
        
        return insights
    
    def _generate_tactical_insights(self, min_offsides: int) -> List[str]:
        """Generate insights about tactical patterns in offside events"""
        insights = []
        
        team0_count = self.analytics.get_team_offsides(0)
        team1_count = self.analytics.get_team_offsides(1)
        total_offsides = team0_count + team1_count
        
        if total_offsides >= min_offsides:
            # Analyze offside positions relative to pitch dimensions
            team0_positions = []
            team1_positions = []
            
            for event in self.analytics.offside_events:
                if event.team == 0:
                    team0_positions.append(event.player_position)
                else:
                    team1_positions.append(event.player_position)
            
            pitch_length, pitch_width = self.analytics.pitch_dimensions
            
            # Check for high offside line (defensive tactics)
            if team0_positions:
                team0_avg_x = sum(pos[0] for pos in team0_positions) / len(team0_positions)
                if team0_avg_x > pitch_length * 0.75 and len(team0_positions) >= min_offsides:
                    insights.append("Team 1's offsides are occurring very deep in the attacking third, suggesting Team 2 is maintaining a high defensive line. Team 1 attackers should look for diagonal runs or quick through balls to exploit this.")
            
            if team1_positions:
                team1_avg_x = sum(pos[0] for pos in team1_positions) / len(team1_positions)
                if team1_avg_x < pitch_length * 0.25 and len(team1_positions) >= min_offsides:
                    insights.append("Team 2's offsides are occurring very deep in the attacking third, suggesting Team 1 is maintaining a high defensive line. Team 2 attackers should look for diagonal runs or quick through balls to exploit this.")
            
            # Check for offside trap patterns
            if len(self.analytics.offside_events) >= 3:
                consecutive_same_team = 0
                last_team = None
                
                for event in sorted(self.analytics.offside_events, key=lambda x: x.timestamp):
                    if last_team is None:
                        last_team = event.team
                        consecutive_same_team = 1
                    elif event.team == last_team:
                        consecutive_same_team += 1
                    else:
                        if consecutive_same_team >= 3:
                            team_name = "Team 1" if last_team == 0 else "Team 2"
                            insights.append(f"{team_name} was caught offside {consecutive_same_team} times in succession, suggesting their opponents may be effectively implementing an offside trap and they need to adjust their attacking approach.")
                        last_team = event.team
                        consecutive_same_team = 1
                
                # Check the last sequence too
                if consecutive_same_team >= 3:
                    team_name = "Team 1" if last_team == 0 else "Team 2"
                    insights.append(f"{team_name} was caught offside {consecutive_same_team} times in succession, suggesting their opponents may be effectively implementing an offside trap and they need to adjust their attacking approach.")
        
        return insights
    
    def _generate_defensive_insights(self, min_offsides: int) -> List[str]:
        """Generate insights about defensive line positioning"""
        insights = []
        
        offside_events = self.analytics.offside_events
        if len(offside_events) >= min_offsides:
            # Analyze offside line positions
            team0_lines = []
            team1_lines = []
            
            for event in offside_events:
                if event.offside_line_position > 0:  # Only consider if we have valid line data
                    if event.team == 0:
                        team0_lines.append(event.offside_line_position)
                    else:
                        team1_lines.append(event.offside_line_position)
            
            pitch_length = self.analytics.pitch_dimensions[0]
            
            # Analyze Team 2's defensive line (for Team 1's offsides)
            if len(team0_lines) >= min_offsides:
                avg_line = sum(team0_lines) / len(team0_lines)
                if avg_line > pitch_length * 0.7:
                    insights.append("Team 2's defensive line is positioned very high up the pitch (average position at 70% of pitch length), suggesting they're using an aggressive offside trap. Team 1 should consider using pace to get behind this high line.")
                elif avg_line < pitch_length * 0.4:
                    insights.append("Team 2's defensive line is sitting quite deep (average position at 40% of pitch length), making it challenging for Team 1 to stay onside. Team 1 should consider more patient build-up play or shots from distance.")
            
            # Analyze Team 1's defensive line (for Team 2's offsides)
            if len(team1_lines) >= min_offsides:
                avg_line = sum(team1_lines) / len(team1_lines)
                if avg_line < pitch_length * 0.3:
                    insights.append("Team 1's defensive line is positioned very high up the pitch, suggesting they're using an aggressive offside trap. Team 2 should consider using pace to get behind this high line.")
                elif avg_line > pitch_length * 0.6:
                    insights.append("Team 1's defensive line is sitting quite deep, making it challenging for Team 2 to stay onside. Team 2 should consider more patient build-up play or shots from distance.")
        
        return insights
    
    def _get_team_counts_for_timestamps(self, timestamps):
        """Helper to get team counts for a list of timestamps"""
        team_counts = {0: 0, 1: 0}
        
        for event in self.analytics.timeline_events:
            if event['type'] == 'offside' and event['timestamp'] in timestamps:
                team_counts[event['team']] += 1
                
        return team_counts
    
    def _get_team_counts_for_events(self, events):
        """Helper to get team counts from a list of events"""
        team_counts = {0: 0, 1: 0}
        
        for event in events:
            team_counts[event['team']] += 1
                
        return team_counts
    
    def get_top_insights(self, max_insights: int = 5) -> List[str]:
        """
        Get the top insights, limited to a maximum number
        
        Args:
            max_insights: Maximum number of insights to return
            
        Returns:
            List of the top insight strings
        """
        all_insights = self.generate_insights()
        
        # Sort insights by potential value (length is a simple proxy for detail/complexity)
        sorted_insights = sorted(all_insights, key=len, reverse=True)
        
        # Return top N insights
        return sorted_insights[:max_insights]
    
    def generate_recommendation(self) -> str:
        """
        Generate a single, most important recommendation based on the insights
        
        Returns:
            A recommendation string
        """
        insights = self.generate_insights()
        
        if not insights:
            return "Insufficient data to generate a recommendation. More match data is needed."
        
        # If we have multiple insights, prioritize the most significant one
        team0_count = self.analytics.get_team_offsides(0)
        team1_count = self.analytics.get_team_offsides(1)
        
        # Check which insight is most important
        if team0_count > team1_count * 2:
            return f"Team 1 should work on timing their attacking runs better and be more aware of the defensive line. They were caught offside {team0_count} times, which is significantly higher than Team 2's {team1_count}."
        elif team1_count > team0_count * 2:
            return f"Team 2 should work on timing their attacking runs better and be more aware of the defensive line. They were caught offside {team1_count} times, which is significantly higher than Team 1's {team0_count}."
        
        # If team comparison isn't significant, check for player insights
        player_stats = self.analytics.player_stats
        if player_stats:
            sorted_players = sorted(player_stats.items(), key=lambda x: x[1]['count'], reverse=True)
            if sorted_players and sorted_players[0][1]['count'] >= 3:
                player_id, stats = sorted_players[0]
                team_name = "Team 1" if stats['team'] == 0 else "Team 2"
                return f"Player {player_id} ({team_name}) should focus on improving the timing of their runs, as they were caught offside {stats['count']} times, more than any other player."
        
        # If no specific issue stands out, give a general recommendation
        return "Both teams should focus on improving the coordination between passers and runners to reduce offside occurrences. Consider video analysis sessions focusing on defensive line awareness and timing of forward runs." 