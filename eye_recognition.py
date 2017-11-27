import cv2
import numpy as np

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


img = cv2.imread('picture3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eye_1 = eye_cascade.detectMultiScale(gray,1.3,5)
print(eye_1)
for (ex, ey, ew, eh) in eye_1:

    cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()