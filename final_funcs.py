import cv2
import numpy as np
from scipy.signal import argrelmin
from scipy.ndimage import gaussian_filter1d
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage import img_as_ubyte
import cv2
import matplotlib.pyplot as plt

def sauvola_binarization(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = threshold_sauvola(gray, window_size=25)
    binary = gray > threshold
    return img_as_ubyte(binary)

def get_horizontal_projection(binary_img):
    return np.sum(binary_img == 255, axis=1)

def find_valleys(hist, smooth_sigma=2, threshold_ratio=0.1):
    smooth_hist = gaussian_filter1d(hist.astype(np.float32), sigma=smooth_sigma)
    min_indices = argrelmin(smooth_hist)[0]
    max_val = np.max(smooth_hist)
    threshold = threshold_ratio * max_val
    valleys = [i for i in min_indices if smooth_hist[i] < threshold]
    return valleys
def line_segmentation(binary_img, smooth_sigma=12, threshold_ratio=0.1):
    # lines_mask = np.zeros_like(binary_img, dtype=np.uint8)

    # Preprocessing: connect horizontal line fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    dilated_img = cv2.dilate(binary_img, kernel)

    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hist = get_horizontal_projection(binary_img)
    valleys = find_valleys(hist, smooth_sigma, threshold_ratio)

    lines = []


    # Step 4: From valleys, slice image and extract line crops
    height = binary_img.shape[0]
    valid_cuts = [0] + [v for v in valleys if 5 < v < height - 5] + [height]

    for i in range(len(valid_cuts) - 1):
        y1, y2 = valid_cuts[i], valid_cuts[i + 1]
        line_crop = binary_img[y1:y2, :]
        if np.sum(line_crop == 0) > 50:  # skip almost-empty lines
            lines.append(cv2.bitwise_not(line_crop))

    return lines


#### Investigating these functions
'''
Works well for the other documents type but not my own handwritten document.

'''
class WordSegmentation:
    def __init__(self):
        self.kernel = None

    def set_kernel(self, kernel_size, sigma, theta):
        # Create anisotropic LoG-like kernel
        k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        sigmaX = sigma
        sigmaY = sigma * theta
        half = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - half
                y = j - half
                term_exp = np.exp((-x**2 / (2 * sigmaX)) - (y**2 / (2 * sigmaY)))
                term_x = (x**2 - sigmaX**2) / (2 * np.pi * sigmaX**5 * sigmaY)
                term_y = (y**2 - sigmaY**2) / (2 * np.pi * sigmaY**5 * sigmaX)
                k[i, j] = (term_x + term_y) * term_exp
        self.kernel = k / np.sum(k)

    def print_contours(self, image, contours, hierarchy, idx):
        # Recursively draw contours starting from idx
        i = idx
        while i >= 0:
            cv2.drawContours(image, contours, i, color=255, thickness=1, lineType=cv2.LINE_8)
            # Iterate over children of contour i
            child = hierarchy[i][2]
            j = child
            while j >= 0:
                # Draw grandchildren
                grandchild = hierarchy[j][2]
                self.print_contours(image, contours, hierarchy, grandchild)
                j = hierarchy[j][0]
            i = hierarchy[i][0]

    def process_bounds(self, image, bound_rects):
        last_number = -1
        # Iterate until contour count stabilizes
        while True:
            contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            edged = np.zeros_like(image)
            # Draw rectangles for each contour
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(edged, (x, y), (x + w, y + h), color=255, thickness=2)
            # Draw nested contours
            if hierarchy is not None:
                h = hierarchy[0]
                self.print_contours(edged, contours, h, 0)
            image = edged.copy()
            if len(contours) == last_number:
                break
            last_number = len(contours)
        # Final bounding rects
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            bound_rects.append(cv2.boundingRect(cnt))
        # Sort by x-coordinate
        bound_rects.sort(key=lambda r: r[0])
        # Merge nested rectangles
        i = 0
        while i < len(bound_rects) - 1:
            x1, y1, w1, h1 = bound_rects[i]
            x2, y2, w2, h2 = bound_rects[i + 1]
            if x1 <= x2 and x1 + w1 >= x2 + w2:
                min_x = min(x1, x2)
                min_y = min(y1, y2)
                max_y = max(y1 + h1, y2 + h2)
                width = max(w1, w2)
                height = abs(min_y - max_y)
                bound_rects[i + 1] = (min_x, min_y, width, height)
                bound_rects.pop(i)
                continue
            i += 1

    def segment(self, line):
        # Add border to the line image
        bordered = cv2.copyMakeBorder(line, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        # Filter and binarize
        img_filtered = cv2.filter2D(bordered, ddepth=cv2.CV_8U, kernel=self.kernel)
        _, img_thresh = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Find bounding rectangles
        bound_rects = []
        self.process_bounds(img_thresh, bound_rects)
        # Annotate
        annotated = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)
        words = []
        for idx, (x, y, w, h) in enumerate(bound_rects):
            cropped = bordered[y:y + h, x:x + w].copy()
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            words.append(cropped)
        # Prepend annotated image
        words.insert(0, annotated)
        return words