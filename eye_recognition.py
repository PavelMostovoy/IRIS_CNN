import cv2
import numpy as np



def i_convertor(image):
    image = cv2.imread(image)
    image = cv2.medianBlur(image, 5)
    image = cv2.GaussianBlur(image, (5, 5), 2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Canny(image,80,50,apertureSize=3)

    return image



gray = i_convertor("picture0.png")
output = gray.copy()


# detect circles in the image
circles = cv2.HoughCircles(gray , cv2.HOUGH_GRADIENT,20, 200,minRadius=2, maxRadius=30) #  param1=128,param2=64, minRadius=1, maxRadius=30
#circles = cv2.HoughCircles(gray, cv2.HoughCircles, 1.2, 100)
# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    print (circles)
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (255, 255, 0), 2)
        #cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # show the output image
    cv2.imshow("output",output)
    #cv2.imshow("output", np.hstack([image, output]))
    cv2.waitKey(0)

else:
    cv2.waitKey(0)


cv2.destroyAllWindows()