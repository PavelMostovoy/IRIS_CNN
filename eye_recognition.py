import cv2
import numpy as np

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


img = cv2.imread('picture3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eye = eye_cascade.detectMultiScale(gray, 1.3, 5)
print(eye)
for (x, y, w, h) in eye:
    name = str(x+y)
    t_img=img.copy()
    t_img = t_img[y:y + h, x:x + w] NOTE: its img[y: y + h, x: x + w]
    cv2.imwrite("{file_mane}.png".format(file_mane=name ), t_img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()