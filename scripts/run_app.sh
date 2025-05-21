#!/bin/bash
# Script to run VAROMATIC+ with the correct Python interpreter

# Find the Python interpreter
PYTHON_PATH=$(which python3 || which python)

if [ -z "$PYTHON_PATH" ]; then
  echo "Error: Python interpreter not found. Please make sure Python is installed."
  exit 1
fi

echo "Using Python interpreter at: $PYTHON_PATH"
echo "Starting VAROMATIC+ application..."

# Run the app
$PYTHON_PATH app.py

# Check if app started successfully
if [ $? -ne 0 ]; then
  echo "Error: Failed to start the application."
  echo "You can try running it manually with: /opt/anaconda3/bin/python app.py"
  exit 1
fi 