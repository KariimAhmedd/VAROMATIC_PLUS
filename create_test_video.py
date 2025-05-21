import cv2
import numpy as np
import os

# Create test_videos directory if it doesn't exist
os.makedirs('test_videos', exist_ok=True)

# Video properties
width = 1280
height = 720
fps = 30
duration = 15  # Increased duration to 15 seconds
total_frames = fps * duration

# Create video writer
if os.path.exists('test_videos/test_football.mp4'):
    os.remove('test_videos/test_football.mp4')

# Try different codecs
try:
    # Try MJPG codec first
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('test_videos/test_football.avi', fourcc, fps, (width, height))
    if not out.isOpened():
        raise Exception("Failed to open video writer with MJPG codec")
except:
    try:
        # Try MP4V codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_videos/test_football.mp4', fourcc, fps, (width, height))
        if not out.isOpened():
            raise Exception("Failed to open video writer with MP4V codec")
    except:
        # Fallback to raw video format
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('test_videos/test_football.avi', fourcc, fps, (width, height))

if not out.isOpened():
    raise Exception("Could not create video writer with any codec")

print("Creating test video...")

# Create moving objects (simulating players)
class Player:
    def __init__(self, x, y, vx, vy, color, team=1, role=""):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.team = team
        self.role = role
        self.height = 80  # Increased player height
        self.width = 40   # Increased player width
        self.running_cycle = 0  # For running animation

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.running_cycle = (self.running_cycle + 1) % 8  # Update running cycle
        
        # Bounce off walls with some margin
        margin = 50
        if self.x < margin or self.x > width - margin:
            self.vx *= -1
        if self.y < margin or self.y > height - margin:
            self.vy *= -1

    def draw(self, frame):
        # Draw a more human-like figure
        # Head
        head_radius = 10
        head_y = int(self.y - self.height//2)
        cv2.circle(frame, (int(self.x), head_y), head_radius, self.color, -1)
        
        # Body
        body_top = (int(self.x), head_y + head_radius)
        body_bottom = (int(self.x), int(self.y + self.height//2))
        cv2.line(frame, body_top, body_bottom, self.color, 4)
        
        # Arms with running animation
        arm_y = head_y + head_radius + 15
        arm_angle = np.sin(self.running_cycle * 0.5) * 30  # Swing arms while running
        arm_dx = int(15 * np.cos(np.radians(arm_angle)))
        arm_dy = int(10 * np.sin(np.radians(arm_angle)))
        
        # Left arm
        cv2.line(frame, (int(self.x), arm_y), 
                 (int(self.x - arm_dx), arm_y + arm_dy), 
                 self.color, 4)
        # Right arm
        cv2.line(frame, (int(self.x), arm_y), 
                 (int(self.x + arm_dx), arm_y - arm_dy), 
                 self.color, 4)
        
        # Legs with running animation
        leg_top = body_bottom
        leg_length = 30
        leg_angle = np.sin(self.running_cycle * 0.5) * 45  # Swing legs while running
        
        # Left leg
        leg_dx = int(leg_length * np.sin(np.radians(leg_angle)))
        leg_dy = int(leg_length * np.cos(np.radians(leg_angle)))
        cv2.line(frame, leg_top, 
                 (int(self.x - leg_dx), int(self.y + self.height//2 + leg_dy)), 
                 self.color, 4)
        
        # Right leg
        cv2.line(frame, leg_top, 
                 (int(self.x + leg_dx), int(self.y + self.height//2 - leg_dy)), 
                 self.color, 4)
        
        # Add team number and role
        cv2.putText(frame, f"{self.team}", (int(self.x - 5), int(self.y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, self.role, (int(self.x - 20), int(self.y - self.height//2 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add white outline
        cv2.rectangle(frame, 
                     (int(self.x - self.width//2), int(self.y - self.height//2)),
                     (int(self.x + self.width//2), int(self.y + self.height//2)),
                     (255, 255, 255), 1)

# Create players with more realistic positions and roles
players = [
    # Team 1 (Blue) - attacking left to right
    Player(width//4, height//2, 3, 0.5, (255, 0, 0), 1, "ST"),      # Striker
    Player(width//3, height//3, 2, 1, (255, 0, 0), 1, "LW"),        # Left Wing
    Player(width//3, 2*height//3, 2, -1, (255, 0, 0), 1, "RW"),     # Right Wing
    Player(width//5, height//2, 1.5, 0.5, (255, 0, 0), 1, "CM"),    # Center Mid
    Player(width//6, height//3, 1, 0.5, (255, 0, 0), 1, "CB"),      # Center Back
    
    # Team 2 (Red) - defending right to left
    Player(3*width//4, height//2, -2, -0.5, (0, 0, 255), 2, "CB"),  # Center Back
    Player(2*width//3, height//3, -1.5, 1, (0, 0, 255), 2, "LB"),   # Left Back
    Player(2*width//3, 2*height//3, -1.5, -1, (0, 0, 255), 2, "RB"), # Right Back
    Player(4*width//5, height//2, -1, -0.5, (0, 0, 255), 2, "GK"),  # Goalkeeper
    Player(5*width//6, height//3, -1, -0.5, (0, 0, 255), 2, "CB"),  # Center Back
]

# Generate video frames
for frame_num in range(total_frames):
    # Create a green background (football field)
    frame = np.ones((height, width, 3), dtype=np.uint8) * np.array([0, 100, 0], dtype=np.uint8)
    
    # Draw field lines
    # Center line
    cv2.line(frame, (width//2, 0), (width//2, height), (255, 255, 255), 2)
    # Center circle
    cv2.circle(frame, (width//2, height//2), 70, (255, 255, 255), 2)
    
    # Draw penalty areas
    # Left penalty area
    cv2.rectangle(frame, (0, height//4), (width//6, 3*height//4), (255, 255, 255), 2)
    # Right penalty area
    cv2.rectangle(frame, (5*width//6, height//4), (width, 3*height//4), (255, 255, 255), 2)
    
    # Draw goal lines
    cv2.line(frame, (0, 0), (0, height), (255, 255, 255), 2)
    cv2.line(frame, (width-1, 0), (width-1, height), (255, 255, 255), 2)
    
    # Create offside scenario after 5 seconds
    if frame_num > fps * 5:
        # Move striker forward to create offside
        players[0].x = max(players[0].x, 3*width//4)  # Keep striker ahead of defenders
    
    # Update and draw players
    for player in players:
        player.update()
        player.draw(frame)
    
    # Write frame
    out.write(frame)
    
    # Display progress
    if frame_num % fps == 0:
        print(f"Generated {frame_num/fps}/{duration} seconds")

# Release video writer
out.release()
print("Test video created successfully!")

# Verify the video can be read
cap = cv2.VideoCapture('test_videos/test_football.avi')
if not cap.isOpened():
    print("Warning: Could not open the created video file for verification")
else:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Could not read the first frame of the created video")
    else:
        print("Video file created and verified successfully!")
    cap.release() 