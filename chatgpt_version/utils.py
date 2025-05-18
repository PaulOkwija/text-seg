import cv2
import numpy as np
from pathlib import Path

class Scanner:
    """
    Scanner performs cropping or preprocessing of the input image.
    In this placeholder implementation, we return the original image.
    Customize this for specific ROI or deskewing operations.
    """
    def process(self, image):
        # Placeholder crop logic — replace with real logic as needed
        return image.copy()


class Binarization:
    """
    Binarization class that supports several methods:
    - 0: Global threshold
    - 1: Otsu
    - 2: Niblack
    - 3: Sauvola
    - 4: Wolf (not implemented)
    """
    def binarize(self, image, method=3):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if method == 1:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 2:
            binary = self._niblack(gray)
        elif method == 3:
            binary = self._sauvola(gray)
        else:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        return binary

    def _niblack(self, gray, window_size=25, k=-0.2):
        mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))
        sqmean = cv2.sqrBoxFilter(gray, cv2.CV_32F, (window_size, window_size))
        stddev = np.sqrt(sqmean - mean**2)
        thresh = mean + k * stddev
        binary = (gray > thresh).astype(np.uint8) * 255
        return binary

    def _sauvola(self, gray, window_size=25, k=0.2, r=128):
        mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))
        sqmean = cv2.sqrBoxFilter(gray, cv2.CV_32F, (window_size, window_size))
        stddev = np.sqrt(sqmean - mean**2)
        thresh = mean * (1 + k * ((stddev / r) - 1))
        binary = (gray > thresh).astype(np.uint8) * 255
        return binary


class LineSegmentation:
    """
    LineSegmentation segments the binarized image into horizontal text lines
    using chunk-based histogram projection and connected components.
    """
    def segment(self, binary_img, chunks=8, process_chunks=4):
        height = binary_img.shape[0]
        chunk_height = height // chunks
        lines = []

        for i in range(process_chunks):
            y_start = i * chunk_height
            y_end = (i + 1) * chunk_height if i < chunks - 1 else height
            chunk = binary_img[y_start:y_end, :]

            hist = cv2.reduce(chunk, 1, cv2.REDUCE_AVG).flatten()
            valleys = np.where(hist < np.max(hist) * 0.5)[0]

            if len(valleys) < 2:
                lines.append(chunk)
            else:
                split_points = [0] + list(valleys[1:]) + [chunk.shape[0]]
                for j in range(len(split_points) - 1):
                    l = chunk[split_points[j]:split_points[j+1], :]
                    if l.shape[0] > 5:
                        lines.append(l)

        return lines



"""
class WordSegmentation
---------------------------------------------------------------
WordSegmentation splits each line into words using morphological closing
and contour detection. The kernel size and thresholding behavior can be customized.
"""
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
