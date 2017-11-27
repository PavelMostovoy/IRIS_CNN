import cv2
print(cv2.__version__)
# from cv2 import cv
import numpy as np
import os
import argparse

'''image = cv2.imread("clouds.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Over the Clouds", image)
cv2.imshow("Over the Clouds - gray", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''#

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="Path to the image")
#args = vars(ap.parse_args())

image = cv2.imread("picture.png")
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#cv2.imshow("gray", gray)


# detect circles in the image
circles = cv2.HoughCircles(gray , cv2.HOUGH_GRADIENT, 2.1, 200,param1=128,  param2=64,minRadius=20, maxRadius=100) #  param1=128,param2=64, minRadius=1, maxRadius=30
#circles = cv2.HoughCircles(gray, cv2.HoughCircles, 1.2, 100)
# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        #cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # show the output image
    cv2.imshow("output", np.hstack([image, output]))
    cv2.waitKey(0)
    
else:
    cv2.waitKey(0)


cv2.destroyAllWindows()