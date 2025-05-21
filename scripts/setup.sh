#!/bin/bash

# VAROMATIC+ Setup Script for macOS
echo "======================================"
echo "  VAROMATIC+ Setup Script for macOS"
echo "======================================"

# Find Python
if command -v python3 &>/dev/null; then
    PYTHON="python3"
    echo "✓ Using Python3: $(which python3)"
elif command -v python &>/dev/null; then
    PYTHON="python"
    echo "✓ Using Python: $(which python)"
else
    echo "❌ Error: Python not found. Please install Python 3."
    exit 1
fi

# Check for pip
if $PYTHON -m pip --version &>/dev/null; then
    echo "✓ Pip is installed"
else
    echo "❌ Error: pip not found. Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $PYTHON get-pip.py --user
    rm get-pip.py
fi

# Install dependencies
echo "Installing required packages..."
$PYTHON -m pip install --user ultralytics opencv-python-headless PySide6 torch torchvision

# Check installation
echo "Verifying installation..."
$PYTHON -c "
import sys
try:
    import torch
    import ultralytics
    import cv2
    import PySide6
    print(f'✓ PyTorch version: {torch.__version__}')
    print(f'✓ MPS acceleration: {torch.backends.mps.is_available()}')
    print(f'✓ OpenCV version: {cv2.__version__}')
    print(f'✓ PySide6 version: {PySide6.__version__}')
    print(f'✓ YOLO version: {ultralytics.__version__}')
    print('All dependencies successfully installed!')
except ImportError as e:
    print(f'❌ Missing package: {str(e)}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "======================================"
    echo "  Setup complete! Run with: ./run_m2.sh"
    echo "======================================"
    chmod +x run_m2.sh
else
    echo "❌ Setup failed. Please check error messages above."
fi 