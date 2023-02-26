import cv2
import pytesseract


def main_ocr(img):
    img_to_text = pytesseract.image_to_string(img)
    return img_to_text


img = cv2.imread('3.jpg')


# this helps us to get the image in grayscale
def get_image_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# here we will remove any noise
def noise_removing(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding_image(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


img = get_image_grayscale(img)
img = noise_removing(img)
img = thresholding_image(img)

print(main_ocr(img))