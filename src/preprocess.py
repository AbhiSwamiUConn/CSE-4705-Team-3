import cvs2
import numpy as numpy
import os

# IMG_SIZE * IMG_SIZE
IMG_SIZE = 32
PI_THRESHOLD = 255
PI_NEIGHBORHOOD_SIZE = 11
PI_CONSTANT = 2

def process_image(image_path):
    # 1. load image in grayscale as numpy array
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 2. binarization (of pixels) -> black and white
    '''
    adaptive thresholding: calculate threshold based on a neighborhood of pixels around every single pixel, rather than "every pixel darker than 127 = black".
    cv2 adpative thresholding: https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
    gaussian: local threshold alg. pixels closer to center of neighborhood weighted more heavily than pixels on the edge.
    inv: note = white (255), background = black (0)
    PI_CONSTANT: threshold = mean - C; filter out noise

    ''' 

    img_bin = cv2.adaptiveThreshold(img, THRESHOLD, cv2.ADAPTIVE_THREASH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, PI_NEIGHBORHOOD_SIZE, PI_CONSTANT)