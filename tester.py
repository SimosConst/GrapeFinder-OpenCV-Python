import cv2
import numpy as np
import conversions as conv


def nothing(x):
    pass


# Create a black image, a window
# img = np.zeros((300,512,3), np.uint8)
img = cv2.imread("../Grapes/grape2.jpeg")
cv2.imshow('arxikh', img)

cv2.namedWindow('controls')
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('R', 'controls', 200, 600, nothing)
cv2.createTrackbar('G', 'controls', 400, 600, nothing)
cv2.createTrackbar('B', 'controls', 9, 100, nothing)
cv2.createTrackbar('B2', 'controls', 3, 50, nothing)
cv2.imshow('controls', np.ones((10, 800), np.uint8))

# create switch for ON/OFF functionality
# switch = '0 : OFF \n1 : ON'
# cv2.createTrackbar(switch, 'controls',0,1,nothing)

while (1):

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R', 'controls')
    g = cv2.getTrackbarPos('G', 'controls')
    b = cv2.getTrackbarPos('B', 'controls')
    b1 = cv2.getTrackbarPos('B2', 'controls')
    # s = cv2.getTrackbarPos(switch,'controls')

    # if s == 0:
    #     img[:] = 0
    # else:
    #     img[:] = [b,g,r]

    # img2 =conv.CannyMask(img, r, g)
    # img2= conv.CircleFinder(img, r)
    img2 = conv.CfCann(img, r, g, b1, b)
    cv2.imshow('image', img2)


cv2.destroyAllWindows()