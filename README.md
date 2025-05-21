# VAROMATIC+ Soccer Analysis System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt-5-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

VAROMATIC+ is an advanced soccer analysis platform that combines computer vision, AI, and data visualization to provide automated offside detection, player tracking, and tactical insights.

![VAROMATIC+ Main Interface](screenshots/main_interface.png)

## 📋 Features

### 🎯 Core Analysis
- **Real-time player detection** with automatic team assignment
- **Offside detection** with 85-90% accuracy in ideal conditions
- **Goal detection** with automated event logging
- **Team color identification** for jersey differentiation

### 📊 Advanced Analytics
- **Offside heatmaps** showing spatial distribution patterns
- **Player-specific statistics** tracking offside frequency by player
- **Zone-based analysis** of offside occurrences across the pitch
- **Match timeline visualization** of all key events
- **Efficiency metrics** relating goals to offsides

### 🧠 AI-Powered Insights
- **Pattern detection** for offside clusters and tactical trends
- **Defensive line analysis** with recommendations
- **Team and player insights** for tactical adjustments
- **Context-aware suggestions** based on match situations

## 🖼️ Screenshots

<table>
  <tr>
    <td><img src="screenshots/video_analysis.png" alt="Video Analysis Tab" width="400"/></td>
    <td><img src="screenshots/analytics_tab.png" alt="Analytics Tab" width="400"/></td>
  </tr>
  <tr>
    <td><img src="screenshots/offside_detection.png" alt="Offside Detection" width="400"/></td>
    <td><img src="screenshots/heatmap.png" alt="Offside Heatmap" width="400"/></td>
  </tr>
</table>

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Pip package manager

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/VAROMATIC-Plus.git
cd VAROMATIC-Plus
```

### Step 2: Set up a virtual environment (recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the application
```bash
python app.py
```

## 📝 Usage

### Analyzing a Video File
1. Launch the application
2. In the "Video Analysis" tab, select "Video File" from the dropdown
3. Click "Select Video File" and choose your soccer match video
4. Configure detection options if needed
5. Click "Start" to begin analysis
6. View real-time detection results and analytics

### YouTube Video Analysis
1. Select "YouTube" from the source dropdown
2. Paste a YouTube URL of a soccer match
3. Click "Download YouTube Video"
4. Once downloaded, click "Start" to begin analysis

### Generating Analytics
1. After processing video, go to the "Analytics" tab
2. Select desired visualizations
3. Click "Generate Analytics"
4. View AI insights and visualizations
5. Export as PDF report if needed

## ⚙️ Configuration

### Detection Settings
- **Detection Confidence**: Controls sensitivity of player detection
  - Lower values (15-30%): Detect more players, may include false positives
  - Higher values (70-90%): Stricter detection, fewer false positives

### Analysis Options
- **Player Detection**: Enable/disable player tracking
- **Offside Detection**: Enable/disable offside line analysis
- **Goal Detection**: Enable/disable goal detection

### Save Options
- **Save Analysis Video**: Export video with detection overlays
- **Save Analysis Results**: Store event data and statistics as JSON/CSV

## 🧩 How It Works

VAROMATIC+ uses a multi-stage pipeline for soccer analysis:

1. **Video Processing**: Frames are captured and resized for efficient processing
2. **Player Detection**: Deep learning models identify players in each frame
3. **Team Assignment**: Players are clustered into teams based on uniform colors
4. **Offside Detection**: Defensive lines are identified and offside positions calculated
5. **Event Recognition**: Goals and key passes are detected and logged
6. **Analytics Generation**: Collected data is processed into visualizations and insights
7. **AI Insights**: Pattern recognition algorithms generate tactical observations

## 📁 Project Structure

VAROMATIC+/
├── app.py # Main application entry point
├── lib/ # Core functionality modules
│ ├── offside_modules/ # Offside detection algorithms
│ ├── analytics/ # Data analysis and visualization
│ └── utils/ # Utility functions
├── models/ # Pre-trained detection models
├── resources/ # UI resources and assets
└── docs/ # Documentat
