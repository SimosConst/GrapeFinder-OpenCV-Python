import cv2
import numpy as np


def sobelViewer(img, kernel_size):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=kernel_size)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return [abs_grad_x, abs_grad_y, grad]


# --------------------------------
# CONTOURS
# --------------------------------

def CannyConv(img, a, b):
    img2 = cv2.Canny(img, a, b)

    return img2


def CannyMask(img, thresh1, thresh2, dialateSize):
    # GET CONTOURS
    img2 = CannyConv(img, thresh1, thresh2)
    # Enlarge Contours
    # img2  = cv2.dilate(img2,np.ones((5, 5), np.uint8))
    # BACK TO RGB FOR THE MASKING
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    # MASKING
    # img2 = cv2.bitwise_and(img, img2)
    # Enlarge Contours
    img2 = cv2.dilate(img2, np.ones((dialateSize, dialateSize), np.uint8))
    # MEDIAN BlUR
    # img2 = cv2.medianBlur(img2, 5)

    return img2


# --------------------------------
# FINDING CIRCLES
# --------------------------------

def CircleFinder(img, blurSize):
    # img2 = img.copy()

    # Blur using 3 * 3 kernel.
    blurred = cv2.blur(img, (blurSize, blurSize))

    # Convert to grayscale.
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray,
                                        cv2.HOUGH_GRADIENT, 1.5, 50, param1=50,
                                        param2=55, minRadius=1, maxRadius=40)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(blurred, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(blurred, (a, b), 1, (0, 0, 255), 3)

    return blurred


# FUNCTIONS ENSEMBLE

def CfCann(img, thresh1, thresh2, dialateSize, blurSize):
    img2 = CannyMask(img, thresh1, thresh2, dialateSize)
    img2 = CircleFinder(img2, blurSize)

    return img2


# USING CONTOURS FUNCTION

def findContours(img, a):
    img2 = img.copy()
    # converting image into grayscale image
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, a, 255, cv2.THRESH_BINARY)

    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0

    # list for storing names of shapes
    for contour in contours:

        # here we are ignoring first counter because
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)

        # using drawContours() function
        cv2.drawContours(img2, [contour], 0, (0, 0, 255), 5)

        # # finding center point of shape
        # M = cv2.moments(contour)
        # x = 0
        # y = 0
        # if M['m00'] != 0.0:
        #     x = int(M['m10'] / M['m00'])
        #     y = int(M['m01'] / M['m00'])

        # # putting shape name at center of each shape
        # if len(approx) == 3:
        #     cv2.putText(img2, 'Triangle', (x, y),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        #
        # elif len(approx) == 4:
        #     cv2.putText(img2, 'Quadrilateral', (x, y),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        #
        # elif len(approx) == 5:
        #     cv2.putText(img2, 'Pentagon', (x, y),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        #
        # elif len(approx) == 6:
        #     cv2.putText(img2, 'Hexagon', (x, y),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        #
        # else:
        #     cv2.putText(img2, 'circle', (x, y),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img2


# --------------------------------
# COLOR ISOLATION
# --------------------------------

def isolateColor(img, lowerbound, upperbound, medBlurAmount):
    # img2 = img.copy()
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img2, lowerbound, upperbound)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img2 = cv2.bitwise_and(img2, mask_rgb)

    img2 = cv2.medianBlur(img2, ksize=medBlurAmount)
    img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)
    return img2


def brightestSpot(img, a):
    if (not (a % 2)):
        a += 1
    img2 = img.copy()
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # apply a Gaussian blur to the image then find the brightest region
    gray = cv2.GaussianBlur(gray, (a, a), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img2 = cv2.circle(gray, maxLoc, 5, (255, 225, 230), 2)

    return img2


def kMeans(img, K):
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 5.0)

    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    print(ret)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2
