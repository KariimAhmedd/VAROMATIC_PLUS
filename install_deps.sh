#!/bin/bash

echo "===================================="
echo "  Installing VAROMATIC+ Dependencies"
echo "===================================="

# Determine the correct pip command
if command -v pip3 &>/dev/null; then
    PIP="pip3"
elif command -v pip &>/dev/null; then
    PIP="pip"
else
    echo "❌ Error: pip not found. Please install pip first."
    exit 1
fi

echo "Using pip command: $PIP"

# Install required packages
echo "Installing PySide6..."
$PIP install PySide6

echo "Installing other dependencies..."
$PIP install ultralytics opencv-python-headless torch yt-dlp

echo "✓ Installation complete!"
echo "You can now run the application with: ./run_m2.sh" 