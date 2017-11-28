import cv2

# 187  84  64  64 || 76  82  63  63

x, y, w, h = 187,  84,  64,  64,
#x,y,w,h = 76,  82,  63,  63

img = cv2.imread('picture3.png')
img2= img.copy()
img = img[y:y+h,x:x+w] # NOTE: its img[y: y + h, x: x + w]
cv2.imwrite("eye.png",img)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()