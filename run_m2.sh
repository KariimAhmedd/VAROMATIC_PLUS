#!/bin/bash
# Script to run VAROMATIC+ optimized for Apple M2 chip

# Print welcome message
echo "======================================"
echo " VAROMATIC+ Optimized for Apple M2 Mac"
echo "======================================"

# Check if Python 3 is available
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "❌ Error: Python not found. Please install Python 3."
    exit 1
fi

# Check for conda environment
if command -v conda &>/dev/null; then
    echo "✓ Conda detected"
    echo "To use with conda environment, run: conda activate <env_name> before running this script"
fi

# Check for required packages
echo "Checking required packages..."
$PYTHON -c "
import sys
try:
    import torch
    import ultralytics
    import cv2
    import numpy
    import PySide6
    print(f'✓ PyTorch version: {torch.__version__}')
    print(f'✓ MPS acceleration: {torch.backends.mps.is_available()}')
    print(f'✓ OpenCV version: {cv2.__version__}')
    print(f'✓ PySide6 version: {PySide6.__version__}')
    print(f'✓ YOLO version: {ultralytics.__version__}')
except ImportError as e:
    print(f'❌ Missing package: {str(e)}')
    print('Please install missing packages with: pip install ultralytics opencv-python-headless PySide6 torch')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Missing required packages. See error messages above."
    exit 1
fi

# Set environment variables for PyTorch MPS acceleration
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run with optimized settings
echo "Starting VAROMATIC+ with M2 optimized settings..."
$PYTHON app.py

# Check if application exited with error
if [ $? -ne 0 ]; then
    echo "❌ Application crashed. See error messages above."
    exit 1
fi 