# Word and Line Segmentation in Python

This project provides Python code for segmenting scanned or photographed document images into individual lines and words using OpenCV and NumPy.

## Features

- **Line Segmentation:** Detects and extracts individual text lines from a binarized document image using horizontal projection and valley detection.
- **Word Segmentation:** Further splits each line into individual word images using contour analysis and bounding box refinement.
- **Customizable Kernels:** Supports anisotropic filtering for improved word boundary detection.