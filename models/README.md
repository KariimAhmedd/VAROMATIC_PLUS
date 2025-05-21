# Field Line Detection Models

VAROMATIC+ currently uses basic computer vision techniques for field line detection using OpenCV.

This folder is reserved for future ML-based models that may be implemented later.

## Basic Line Detection

The current implementation uses:
- Color-based field detection
- Edge detection with Canny
- Line detection with HoughLinesP
- Post-processing for line classification

This approach works well for most standard camera angles and field conditions. 