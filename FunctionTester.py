import cv2
import numpy as np

import Conversions as conv
import Functions as func
import SlidersWindow as sldWin

# LOAD IMAGE
img = cv2.imread("grapes/grape4.jpeg")

# IMAGE MULTIPLYER
windowSizeMult = 2

# INITIAL IMAGE FRAME
cv2.imshow('arxikh', func.resizeImg(img, windowSizeMult))
cv2.namedWindow('image')

# CREATE SLIDERWINDOW OBJECT
w = sldWin.slidersWindow()

while (1):
    # BREAK WHEN PRESSING KEY
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # GET CHANGES FROM SLIDERS
    v = w.getSliderValues()
    # CONVERSIONS
    img2 = conv.isolateColor(img, np.array((v[0], v[2], 0), dtype="uint8"), np.array((v[1], v[3], 255), dtype="uint8"))

    # CONVERTED IMAGE FRAME
    cv2.imshow('image', func.resizeImg(img2, windowSizeMult))

cv2.destroyAllWindows()
