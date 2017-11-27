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

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('picture.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eye_1 = eye_cascade.detectMultiScale(gray,1.3,5)
print(eye_1)
for (ex, ey, ew, eh) in eye_1:

    cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)


'''faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
'''
cv2.destroyAllWindows()