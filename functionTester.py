import time

import cv2
import numpy as np

import Conversions as conv
import Functions as func
import SlidersWindow as sldWin

# LOAD IMAGE
img = cv2.imread("grapes/grape5.jpeg")

# IMAGE MULTIPLYER
windowSizeMult = 2

# INITIAL IMAGE FRAME
cv2.imshow('arxikh', func.resizeImg(img, windowSizeMult))

# CREATE SLIDERWINDOW OBJECT
w = sldWin.slidersWindow()

while (1):
    time.sleep(.4)
    # BREAK WHEN PRESSING KEY
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # GET CHANGES FROM SLIDERS
    v = w.getSliderValues()

    # # ----CONVERSIONS-----
    # # COLOR ISOLATION
    # img2 = conv.isolateColor(img, np.array((v[0], v[2], 0), dtype="uint8"), np.array((v[1], v[3], 255), dtype="uint8"), 7)
    # cv2.imshow("ColorIsolation", func.resizeImg(img2, windowSizeMult))
    # # CANNY WITH CIRCLE FINDER
    # img2 = conv.CfCann(img2, thresh1=v[4], thresh2=v[5], dialateSize=v[6], blurSize=v[7])
    # # CONTOUR FINDER
    # img2 = conv.findContours(img2, w.getSliderValuesByName("CannThresh2"))

    img2 = cv2.medianBlur(img, 3)
    img2 = conv.kMeans(img2, v[6])
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # img2 = cv2.absdiff(img, img2)
    # img2 = cv2.bitwise_or(img, img2)

    # CONVERTED IMAGE FRAME
    cv2.imshow("Final Image", func.resizeImg(img2, windowSizeMult))

cv2.destroyAllWindows()
