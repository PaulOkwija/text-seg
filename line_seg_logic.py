import cv2
import numpy as np
from scipy.signal import argrelmin
from scipy.ndimage import gaussian_filter1d

def get_horizontal_projection(binary_img):
    return np.sum(binary_img == 255, axis=1)

def find_valleys(hist, smooth_sigma=2, threshold_ratio=0.1):
    smooth_hist = gaussian_filter1d(hist.astype(np.float32), sigma=smooth_sigma)
    min_indices = argrelmin(smooth_hist)[0]
    max_val = np.max(smooth_hist)
    threshold = threshold_ratio * max_val
    valleys = [i for i in min_indices if smooth_hist[i] < threshold]
    return valleys

def line_extraction(binary_img, x, y, w, h, valleys, offset=15, min_chunk_area=50):
    chunk_region = binary_img[y: y + h, x: x + w]
    line_mask = np.zeros_like(binary_img, dtype=np.uint8)
    chunk_mask = np.zeros_like(binary_img, dtype=np.uint8)

    # Calculate ROI bounds
    left = max(x - offset, 0)
    right = min(x + w + offset, binary_img.shape[1])
    chunk_mask[y: y + h, left:right] = binary_img[y: y + h, left:right]

    # Break line at valley points inside this chunk
    for v in valleys:
        if (y+100) < v < (y + h-100):
            chunk_mask[v - 1: v + 2, left:right] = 0

    # Dilate to connect components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    dilated = cv2.dilate(chunk_mask[y: y + h, left:right], kernel)

    # Find contours in the processed chunk
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_chunk_area:
            continue
        bbox = cv2.boundingRect(cnt)
        cx, cy, cw, ch = bbox
        full_x = left + cx
        full_y = y + cy
        line_mask[full_y: full_y + ch, full_x: full_x + cw] = 255

    return line_mask

def line_segmentation(binary_img, smooth_sigma=12, threshold_ratio=0.1):
    """
    Segments a binarized document into individual lines using valley detection.

    Args:
        binary_img (np.ndarray): A binarized grayscale image (text = black, background = white).
        smooth_sigma (float): Smoothing factor for horizontal histogram.
        threshold_ratio (float): Valley threshold as a ratio of peak height.

    Returns:
        list[np.ndarray]: List of cropped binary image segments, each containing a single line.
    """
    inverted = cv2.bitwise_not(binary_img)
    # Step 1: Dilate to connect line fragments horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    dilated = cv2.dilate(binary_img, kernel)

    # Step 2: Extract bounding boxes of text regions
    # contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 3: Compute smoothed horizontal projection
    hist = get_horizontal_projection(inverted)
    valleys = find_valleys(hist, smooth_sigma, threshold_ratio)

    lines = []
    # Step 4: From valleys, slice image and extract line crops
    height = inverted.shape[0]
    valid_cuts = [0] + [v for v in valleys if 5 < v < height - 5] + [height]

    for i in range(len(valid_cuts) - 1):
        y1, y2 = valid_cuts[i], valid_cuts[i + 1]
        line_crop = inverted[y1:y2, :]
        if np.sum(line_crop == 0) > 50:  # skip almost-empty lines
            lines.append(cv2.bitwise_not(line_crop))

    return lines

def visualize_lines(image, lines_mask):
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return vis
