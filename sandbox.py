import cv2
import numpy as np

def i_convertor(image):
    image = cv2.imread(image)
    image = cv2.medianBlur(image, 5)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_test = cv2.Canny(image,80,50,apertureSize=3)
    image_test = cv2.GaussianBlur(image_test, (5,5), 1)
    return image_test



cv2.imshow("gray", i_convertor("picture.png"))

cv2.waitKey(0)
cv2.destroyAllWindows()
