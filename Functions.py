import colorsys

import cv2
from matplotlib import pyplot as plt


def nothing(x):
    pass


# --------------------------
# DISPLAY IMAGE
# --------------------------

def pltshow(img):
    plt.figure()
    plt.imshow(img), plt.title("Title1")


def subpltshow(im1, im2):
    plt.figure()
    # plt.subplot(, plt.title("Title1")


def showim(img):
    cv2.namedWindow("Window_1", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Trackbar_1", "Window_1", 0, 255, nothing)

    while (cv2.waitKey(1) & 0xFF) != 27:
        # THRESSHOLD CALCULATION
        thresshold = cv2.getTrackbarPos("Trackbar_1", "Window_1") / (100 * 8)
        # print(thresshold)

        # SHOW IMAGE
        cv2.imshow("Window_1", img)

    cv2.destroyAllWindows()


# --------------------------
# IMAGE RESIZE
# --------------------------
def resizeImg(img, multiplyer):
    h, w, _ = img.shape
    w = w * multiplyer
    h = h * multiplyer

    img2 = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST_EXACT)

    return img2


# --------------------------
# COLOR CHANGE
# --------------------------

def s_hsv2grb(h):
    # CONVERSIONS
    h = h / 360

    # RGB Truple Multimplication
    rgb_truple = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, 1, 1))

    # CORRECT SEQUENCE
    rgb_truple = rgb_truple[2], rgb_truple[1], rgb_truple[0]

    return rgb_truple


def hsv2grb(h, s, v):
    # CONVERSIONS
    h = h / 360
    s = s / 100
    v = v / 100

    # RGB Truple Multimplication
    rgb_truple = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))
    # CORRECT SEQUENCE
    rgb_truple = rgb_truple[2], rgb_truple[1], rgb_truple[0]

    return rgb_truple


# --------------------------
# UI
# --------------------------

def puttextspecif(image, text, x, y, width, width_multiplyer, text_thickness, color):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, width * width_multiplyer, color, text_thickness)
