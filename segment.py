from skimage.filters import threshold_sauvola
from skimage import img_as_ubyte
import cv2

def sauvola_binarization(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = threshold_sauvola(gray, window_size=25)
    binary = gray > threshold
    return img_as_ubyte(binary)

def segment_lines(binary_img):
    inverted = cv2.bitwise_not(binary_img)
    hist = cv2.reduce(inverted, 1, cv2.REDUCE_AVG).reshape(-1)

    th = 2  # Threshold to separate lines
    H = binary_img.shape[0]
    uppers = [y for y in range(H-1) if hist[y] <= th and hist[y+1] > th]
    lowers = [y for y in range(H-1) if hist[y] > th and hist[y+1] <= th]

    lines = []
    for i in range(min(len(uppers), len(lowers))):
        lines.append(binary_img[uppers[i]:lowers[i], :])
    return lines

def segment_words(line_img):
    contours, _ = cv2.findContours(line_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_imgs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            word_imgs.append(line_img[y:y+h, x:x+w])
    return sorted(word_imgs, key=lambda w: cv2.boundingRect(w)[0])  # left to right


 
