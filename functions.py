import colorsys
import time

import cv2
from matplotlib import pyplot as plt
import numpy as np


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


def showImgs(imgs, windowSizeMult=2):
    # time.sleep(.5)
    for i in range(len(imgs)):
        img = imgs[i]
        img = resizeImg(img, windowSizeMult)
        cv2.imshow("Image" + str(i), img)

    for i in range(len(imgs), 10):
        # Find if Window exists to close it
        cond = cv2.getWindowProperty("Image" + str(i), cv2.WND_PROP_VISIBLE)
        if (cond):
            cv2.destroyWindow("Image" + str(i))


# SINGLE IMG SHOW
def showImg(img,windowName="Window_1", windowSizeMult=2):
    # RESIZE IMAGE
    img = resizeImg(img, windowSizeMult)
    # SHOW IMAGE
    cv2.imshow(windowName, img)

#NOT FOR EVERY FUNCTION
def calcImgs(imgs,function, *args):
    for i in range(0, len(imgs)):
        img = imgs[i]
        imgs[i] = function(img, *args)
    # for img in imgs:
    #     img = function(img, *args)

#
# def calcImgs

# --------------------------
# IMAGE RESIZE
# --------------------------
def resizeImg(img, multiplyer):
    h, w, _ = img.shape
    w = int(w * multiplyer)
    h = int(h * multiplyer)

    img2 = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

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


def bgr2hsv255(bgr):
    bgr = np.array(bgr, dtype=np.float32)
    bgr /= 255
    # print(bgr)
    # print(np.flip(bgr))
    h, s, v = colorsys.rgb_to_hsv(bgr[2], bgr[1], bgr[0])
    # print(h,s,v)
    hsv255 = np.array([h, s, v], dtype=np.float32)
    hsv255 *= 255
    # print(hsv255)
    return hsv255


def hue255FromBGR(bgr):
    bgr = np.array(bgr, dtype=np.float32)
    bgr /= 255
    # print(bgr)
    # print(np.flip(bgr))
    h, _, _ = colorsys.rgb_to_hsv(bgr[2], bgr[1], bgr[0])
    h *= 255
    return h


# bgr2hsv255([255,255,255])
# --------------------------
# UI
# --------------------------

def puttextspecif(image, text, x, y, width, width_multiplyer, text_thickness, color):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, width * width_multiplyer, color, text_thickness)
