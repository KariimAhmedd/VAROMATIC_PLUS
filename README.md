# VAROMATIC+

An enhanced football video analysis application for detecting offsides and analyzing goal-line situations using modern computer vision techniques.

## Features

- **Advanced Player Detection**: Uses YOLOv8 for accurate player detection and tracking
- **Improved Color Analysis**: Enhanced team color detection using LAB color space and perceptual color matching
- **Robust Offside Detection**: More accurate offside line drawing and position analysis
- **Modern UI**: Clean, modern interface built with PyQt6
- **Multi-Platform Support**: Works on Windows, macOS, and Linux
- **Hardware Acceleration**: Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU processing

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster processing)
- Apple Silicon Mac (optional, for MPS acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/varomatic-plus.git
cd varomatic-plus
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Select your video source:
   - Use webcam
   - Load video file

3. Choose detection method:
   - YOLOv8 (recommended)
   - Traditional CV (faster but less accurate)

4. Set attack direction:
   - Left
   - Right

5. Click "Start Analysis" to begin processing

## Key Components

- `player_detection.py`: YOLOv8-based player detection
- `color_analysis.py`: Advanced team color detection
- `offside_detection.py`: Improved offside analysis
- `app.py`: Modern PyQt6-based GUI

## Improvements Over Original Version

1. **Player Detection**:
   - Replaced traditional CV with YOLOv8
   - Better handling of occlusions and overlapping players
   - More accurate bounding boxes

2. **Color Analysis**:
   - Uses LAB color space for better perceptual matching
   - Improved color consistency across frames
   - Better handling of varying lighting conditions

3. **Offside Detection**:
   - More accurate line drawing
   - Better player position tracking
   - Improved error handling

4. **User Interface**:
   - Modern, responsive design
   - Real-time preview
   - Better error handling and user feedback

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original VAROMATIC project
- YOLOv8 by Ultralytics
- OpenCV community
- PyQt team 