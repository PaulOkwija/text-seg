import cv2
import numpy as np
import argparse

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


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Perform word segmentation on a line image.")
#     parser.add_argument("image", help="Path to the grayscale line image.")
#     parser.add_argument("--kernel_size", type=int, default=31, help="Size of the filter kernel.")
#     parser.add_argument("--sigma", type=float, default=4.0, help="Sigma for the filter.")
#     parser.add_argument("--theta", type=float, default=1.0, help="Theta for anisotropy.")
#     parser.add_argument("--output_prefix", default="seg", help="Prefix for output files.")
#     args = parser.parse_args()

#     # Load image
#     line_img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
#     if line_img is None:
#         print(f"Failed to load image '{args.image}'")
#         exit(1)

#     # Segment
#     ws = WordSegmentation()
#     ws.set_kernel(args.kernel_size, args.sigma, args.theta)
#     results = ws.segment(line_img)

#     # Save results
#     for i, img in enumerate(results):
#         out_path = f"{args.output_prefix}_{i}.png"
#         cv2.imwrite(out_path, img)
#     print(f"Segmentation complete. Saved {len(results)} images with prefix '{args.output_prefix}'.")
