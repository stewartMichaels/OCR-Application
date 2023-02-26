import cv2
import numpy as np
import pytesseract
import os

# Custom Settings
customConfig = r'--oem 3 --psm 1 -c preserve_interword_spaces=1'
uploadFolder = 'static/uploads/'


# removal of the noise in t
# he picture
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# opeening part of erosion followed by dilation
def opening(image):
    kernel = np.ones((7, 7), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# closing part of dialtion followed by the erosion
def closing(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def ocr_driver(fname):
    image = cv2.imread(os.path.join(uploadFolder, fname))
    ocr1 = pytesseract.image_to_string(image, config=customConfig)
    image_med = remove_noise(image)
    # cv2.imshow('img',image_med)
    # cv2.waitKey(0)
    ocr2 = pytesseract.image_to_string(image_med, config=customConfig)
    image_op = opening(image)
    # cv2.imshow('img',image_op)
    # cv2.waitKey(0)
    ocr3 = pytesseract.image_to_string(image_op, config=customConfig)
    image_cl = closing(image)
    # cv2.imshow('img',image_cl)
    # cv2.waitKey(0)
    ocr4 = pytesseract.image_to_string(image_cl, config=customConfig)
    return ocr1[:-1], ocr2[:-1], ocr3[:-1]


def ocr_fun(fname):
    image = cv2.imread(os.path.join(uploadFolder, fname))
    ocr1 = pytesseract.image_to_string(image, config=customConfig)
    # print(ocr1)
    return ocr1


cv2.destroyAllWindows()