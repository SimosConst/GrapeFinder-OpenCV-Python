import cv2
import numpy as np


def CannyConv(img, a, b):
    img2 = cv2.Canny(img, a, b)

    return img2


def CannyMask(img, a, b, c):
    # GET CONTOURS
    img2 = CannyConv(img, a, b)
    # Enlarge Contours
    # img2  = cv2.dilate(img2,np.ones((5, 5), np.uint8))
    # BACK TO RGB FOR THE MASKING
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    # MASKING
    # img2 = cv2.bitwise_and(img, img2)
    # Enlarge Contours
    img2 = cv2.dilate(img2, np.ones((c, c), np.uint8))
    # MEDIAN BlUR
    # img2 = cv2.medianBlur(img2, 5)

    return img2


def CircleFinder(img, a1):
    # img2 = img.copy()

    # Blur using 3 * 3 kernel.
    blurred = cv2.blur(img, (a1, a1))

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


def CfCann(img, a, b, c, d):
    img2 = CannyMask(img, a, b, c)
    img2 = CircleFinder(img2, d)

    return img2


def findContours(img, a, b):
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


def isolateColor(img, lowerbound, upperbound):
    # img2 = img.copy()
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img2, lowerbound, upperbound)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img2 = cv2.bitwise_and(img2, mask_rgb)
    img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)
    return img2
