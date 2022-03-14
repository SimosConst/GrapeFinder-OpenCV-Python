import heapq
import time

import cv2
import matplotlib
import numpy as np
import functions as func
import slidersWindow as sldWin
import conversions as conv

# LOAD IMAGE
img = cv2.imread("grapes/grape8.jpeg")

# IMAGE MULTIPLYER
windowSizeMult = 2

# INITIAL IMAGE FRAME
cv2.imshow('arxikh', func.resizeImg(img, windowSizeMult))

# CREATE SLIDERWINDOW OBJECT
w = sldWin.slidersWindow()

while (1):
    time.sleep(.1)
    # BREAK WHEN PRESSING KEY
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # GET CHANGES FROM SLIDERS
    v = w.getSliderValues()

    # # ----CONVERSIONS-----
    # # COLOR ISOLATION
    # img2 = conv.isolateColor(img, np.array((v[0], v[2], v[4]), dtype="uint8"), np.array((v[1], v[3], v[5]), dtype="uint8"), 3)
    # cv2.imshow("ColorIsolation", func.resizeImg(img2, windowSizeMult))
    # # CANNY WITH CIRCLE FINDER
    # img2 = conv.CfCann(img2, thresh1=v[4], thresh2=v[5], dialateSize=v[6], blurSize=v[7])
    # # CONTOUR FINDER
    # img2 = conv.findContours(img2, w.getSliderValuesByName("CannThresh2"))


    img2 = cv2.bilateralFilter(img, 38, 90, 40)
    # img2 = cv2.medianBlur(img2, 5)
    # cv2.imshow("Filetered", func.resizeImg(img2, windowSizeMult))

    imgs = conv.approach1(img2, v[7])
    func.showImgs(imgs)

    # img2 = conv.kMeans(img2, v[2])

    # img2 = cv2.medianBlur(img2, 3)
    # ks = 4
    # # img2 = cv2.GaussianBlur(img, (ks,ks), v[8])
    # img2 = conv.gaussBlur(img, v[7], v[8])
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # img2 = cv2.absdiff(img, img2)
    # img2 = cv2.bitwise_or(img, img2)

    # CONVERTED IMAGE FRAME
    # cv2.imshow("Final Image", func.resizeImg(img2, windowSizeMult))

cv2.destroyAllWindows()
