import math
from builtins import len
from collections import Counter as ct

import cv2
import numpy as np


def sobelViewer(img, kernel_size):
    kernel_size = 1 if (kernel_size <= 0) else kernel_size
    kernel_size = kernel_size + 1 if ((kernel_size % 2) == 0) else kernel_size
    kernel_size = 31 if (kernel_size > 31) else kernel_size
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=kernel_size)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return [abs_grad_x, abs_grad_y, grad]


# --------------------------------
# MORPHOLOGICAL OPERATIONS
# --------------------------------

def morphOps(img, kSize=3, kShape=0, method=0):
    kSize = 1 if kSize == 0 else kSize

    # MORPH_RECT = 0
    # MORPH_CROSS = 1
    # MORPH_ELLIPSE = 2
    # shapes = [cv2.MORPH_ELLIPSE, cv2.MORPH_RECT, cv2.MORPH_CROSS]

    # MORPH_ERODE = 0   MORPH_GRADIENT = 4
    # MORPH_DILATE = 1  MORPH_TOPHAT = 5
    # MORPH_OPEN = 2    MORPH_HITMISS = 7
    # MORPH_CLOSE = 3   MORPH_BLACKHAT = 6
    # methods = [
    #     cv2.MORPH_OPEN, cv2.MORPH_CLOSE,
    #     cv2.MORPH_DILATE, cv2.MORPH_ERODE,
    #     cv2.MORPH_GRADIENT, cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT]

    kernel = cv2.getStructuringElement(kShape, 2 * [kSize])
    opening = cv2.morphologyEx(img, method, kernel)
    return opening


# --------------------------------
# BLUR & FILTERS
# --------------------------------


def gaussBlur(img, kdims, sigma):
    if (kdims % 2 != 1):
        kdims += 1

    img2 = cv2.GaussianBlur(img, (kdims, kdims), sigma)
    return img2


def filter2d(img, kernel):
    # kernel2 = np.ones((v[8], v[8]), np.float32) / v[7]
    # kernel2 = np.array([
    #     [-1, -1, -1, -1, -1],
    #     [-1, -1, -1, -1, -1],
    #     [-1, -0.9, a, -1, -1],
    #     [-1, -1, -1, -1, -1],
    #     [-1, -1, -1, -1, -1]
    # ])

    img2 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return img2


# --------------------------------
# CONTOURS
# --------------------------------


def cannyMask(img, thresh1, thresh2, dialateSize):
    # GET CONTOURS
    img2 = cv2.Canny(img, thresh1, thresh2)
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

def circle_finder(img, cannyParam1=10, cannyParam2=32, minRadius=1, maxRadius=25):
    # # Blur using 3 * 3 kernel.
    # blurred = cv2.blur(img, (blurSize, blurSize))

    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray,
                                        cv2.HOUGH_GRADIENT, 1.5, 20, param1=cannyParam1,
                                        param2=cannyParam2, minRadius=minRadius, maxRadius=maxRadius)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        # CONVERSIONS FOR LATER MANIPUATION
        circles = detected_circles[0, :]
        circle_radiuses = [sublist[2] for sublist in circles]
        circles = [sublist.tolist() for sublist in circles]

        # Calculate the percentage of circles from which the most common ones will be drawn
        counter = ct(circle_radiuses)
        percentage = 4.3
        fraction = math.floor(percentage * len(circle_radiuses) / 100)
        # IF FRACTION IS 0 THEN MAKE IT 1
        if fraction == 0:
            fraction = 1
        # FIND MOST COMMON CIRCLE RADIUSES ACCORDING TO A FRACTION
        most_comm = np.asarray(counter.most_common(fraction))[:, 0]
        print(
            # len(counter),
            # counter,
            fraction,
            most_comm
        )
        sel_circles = []
        # FIND AND LIST THE MOST COMMON RADIUSES ACCORDING TO THE INITIAL LIST FOUND
        for i in range(len(most_comm)):
            circl_radius = most_comm[i]
            for j in range(len(circles)):
                if circl_radius == circle_radiuses[j]:
                    # print(circles[j], circle_radiuses[j])
                    sel_circles.append(circles[j])
        # print(sel_circles)
        # time.sleep(.3)
        # sel_circles = circles
        for pt in sel_circles:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (190, 180, 180), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (180, 180, 190), 3)

    return detected_circles, img


# FUNCTIONS ENSEMBLE
def cfCann(img, thresh1, thresh2, dialateSize, blurSize):
    img2 = cannyMask(img, thresh1, thresh2, dialateSize)
    img2 = circle_finder(img2, blurSize)

    return img2


# SAME AS FIND CONTOURS BUT THIS RETURNS AN OVERLAY
def contoursOvrlay(img, lowrThrsh, minContArea=0):
    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # setting threshold of gray image
    # _, threshold = cv2.threshold(gray, lowrThrsh, 255, cv2.THRESH_BINARY)
    threshold = cv2.Canny(gray, lowrThrsh, 10)

    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    i = 0
    # GET A BLACK IMAGE WITH THE SAME DIMENTIONS
    bimg = np.zeros(img.shape, np.uint8)

    contour_list = []
    # list for storing names of shapes
    for contour in contours:
        # here we are ignoring first counter because
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        approx = cv2.approxPolyDP(contour, .001 * cv2.arcLength(contour, 1), 1)
        area = cv2.contourArea(contour)

        # print(approx)
        inLengthRange = lambda: (len(approx) > 10) & (len(approx) < 100)
        inAreaRange = lambda: area > 50

        if inLengthRange() & inAreaRange():
            # if inAreaRange():
            # contour_list.append(contour)
            cv2.drawContours(bimg, [contour], 0, 3 * [255]
                             , 1
                             # ,cv2.FILLED
                             )
    return bimg


# USING CONTOURS FUNCTION
def findContours(img, lowrThrsh=50):
    img2 = img.copy()
    # converting image into grayscale image
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, lowrThrsh, 255, cv2.THRESH_BINARY)

    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    i = 0

    # list for storing names of shapes
    for contour in contours:
        # here we are ignoring first counter because
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape
        # approx = cv2.approxPolyDP(
        #     contour, 0.01 * cv2.arcLength(contour, True), True)
        # hull = cv2.convexHull(contour)
        # using drawContours() function
        cv2.drawContours(img2, [contour], 0, (100, 100, 255), 3)

        # # finding center point of shape
        # M = cv2.moments(contour)
        # x = 0
        # y = 0
        # if M['m00'] != 0.0:
        #     x = int(M['m10'] / M['m00'])
        #     y = int(M['m01'] / M['m00'])

    return img2


# --------------------------------
# COLOR ISOLATION
# --------------------------------

def isolateColorWithVectors(img, lowerbound, upperbound, medBlur=False, medBlurAmount=3):
    # img2 = img.copy()
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img2, lowerbound, upperbound)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img2 = cv2.bitwise_and(img2, mask_rgb)

    if medBlur: img2 = cv2.medianBlur(img2, ksize=medBlurAmount)
    img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)
    return img2


def isolateHue(img, lowerbound, upperbound, medBlurAmount=0):
    lBoundArr = np.array([lowerbound, 0, 0])
    hBoundArr = np.array([upperbound, 255, 255])
    img2 = isolateColorWithVectors(img, lBoundArr, hBoundArr, medBlurAmount=medBlurAmount)
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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 80.0)

    ret, label, center = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    # print(label)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    getIsolColImgs(img, center)
    res = center[label.flatten()]
    # print(res)
    res2 = res.reshape((img.shape))

    return res2


def getIsolColImgs(img, quantinizeStep=32, colorMargin=10):
    # GET HSV IMAGE
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # GET HISTOGRAMM
    histHue = cv2.calcHist([img2], [0], None, [256], [0, 256])
    # CONVERSIONS FOR MANIPULATION
    histHue = histHue.tolist()
    histHue = [item for sublist in histHue for item in sublist]

    # SOTRING
    srtd = sorted(range(len(histHue)), reverse=True, key=lambda k: histHue[k])
    max31 = []
    step = quantinizeStep / 2
    max31.append(srtd[1])
    # print(srtd)
    # FIND MAX BETWEEN RANGES
    for i in range(len(srtd)):
        num = srtd[i]
        # print(num, max31)

        count = 0
        for st in max31:
            cond1 = (num >= st + step) or (num <= st - step)
            if cond1:
                count += 1

        if count == len(max31):
            max31.append(num)
    # # ---- ----- ---- ----- ---- ----- ---- ----- DEBUG --- ---- -----
    # print("Number of Hues: " + str(len(max31)) + ", " + "Hue Values: " + str(max31))

    count = 0
    imgs = []
    for color in max31:

        hue255 = color
        margin = colorMargin / 2

        lowerbound = np.array([hue255 - margin, 0, 0], dtype=np.uint8)
        upperbound = np.array([hue255 + margin, 255, 255], dtype=np.uint8)

        img2 = isolateColorWithVectors(img, lowerbound, upperbound, medBlur=False)
        # img2 = cv2.medianBlur(img2, 3)
        # SHOW ONLY PICTURES WITH BLACK PERCENT LOWER THAN 5%
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        noBlackPxl = cv2.countNonZero(gray)
        noPxl = gray.size
        blackPrct = 100 * noBlackPxl / noPxl
        # print(noBlackPxl, noPxl, blackPrct)
        cond1 = True
        cond1 = blackPrct >= 4

        if cond1:
            # Find if Window exists to close it
            # cond = cv2.getWindowProperty("img" + str(count), cv2.WND_PROP_VISIBLE)
            # if(cond):
            #     cv2.destroyWindow("img" + str(count))

            # img2 = cv2.imshow("img" + str(count), func.resizeImg(img2, 1))
            imgs.append(img2)
            # time.sleep(.5)
            count += 1
    return imgs


# QUANTIZE THE VALUE OF THE HSV VECTOR WITH MODULO SUBDIVISIONS
def quantinizeHSValue(img, divisions=12, preBlur=1, postBlur=1):
    # VALUE LIMITER
    n = divisions if divisions > 0 else 1

    # GET HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # FIRST BLURING
    if preBlur:
        img = cv2.medianBlur(img, 3)

    # QUANTIZATION
    img[:, :, 2] = np.array(np.floor_divide(img[:, :, 2], n) * (n), np.uint8)

    # PostBlur
    if postBlur:
        img = cv2.medianBlur(img, 5)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    return img


def quantinizeRGBChannel(img, divisions=12, channel=1):
    n = divisions
    if channel > 2:
        channel = 2
    elif channel < 0:
        channel = 0

    img = cv2.medianBlur(img, 3)
    img = np.array(np.floor_divide(img[:, :, channel], n) * (n), np.uint8)
    img = cv2.medianBlur(img, 5)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def quantinizeGray(img, divisions=12):
    n = divisions

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 3)
    img = np.array(np.floor_divide(img, n) * n, np.uint8)
    img = cv2.medianBlur(img, 5)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def approach1(img, colMarg=15, canPar1=10, canPar2=32, minRad=1, maxRad=25, quantinizeStep=32):
    imgs = getIsolColImgs(img, colorMargin=colMarg, quantinizeStep=quantinizeStep)
    imgs2 = []
    for i in range(len(imgs)):
        img = imgs[i]
        no_circles, img = circle_finder(
            img,
            cannyParam1=canPar1,
            cannyParam2=canPar2,
            minRadius=minRad,
            maxRadius=maxRad
        )
        # print(type(nocircles))

        if no_circles is not None:
            # # ---- ----- ---- ----- ---- ----- ---- ----- DEBUG --- ---- -----
            # print(len(no_circles[0]), no_circles[0])
            if len(no_circles[0]) >= 6:
                imgs2.append(img)

    return imgs2
