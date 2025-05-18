from skimage.filters import threshold_sauvola
from skimage import img_as_ubyte

def binarize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    window_size = 25
    thresh_sauvola = threshold_sauvola(gray, window_size=window_size)
    binary_sauvola = gray > thresh_sauvola
    binary_image = img_as_ubyte(binary_sauvola)
    return binary_image
