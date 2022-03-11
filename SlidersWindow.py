import cv2
import numpy as np


class slidersWindow:
    name = 'sliders'

    sliderNames = [
        "HueLow",
        "HueHigh",
        "SatLow",
        "SatHigh"
    ]

    def __init__(self):
        self.initSliders()

    def nothing(self, x):
        pass

    def initSliders(self):
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)

        # create trackbars for color change
        cv2.createTrackbar(self.sliderNames[0], self.name, 0, 255, self.nothing)
        cv2.createTrackbar(self.sliderNames[1], self.name, 255, 255, self.nothing)
        cv2.createTrackbar(self.sliderNames[2], self.name, 0, 255, self.nothing)
        cv2.createTrackbar(self.sliderNames[3], self.name, 255, 255, self.nothing)

    def showWindow(self):
        cv2.imshow(self.name, np.ones((1, 100), np.uint8))

    def getSliderValues(self):
        values = []
        for i in self.sliderNames:
            index = self.sliderNames.index(i)
            sliderName = self.sliderNames[index]
            value = cv2.getTrackbarPos(sliderName, self.name)
            values.append(value)
        return values

# TESTING
# w1 = slidersWindow()
# print(w1.sliderNames)
# w1.showWindow()
# print(w1.getSliderValues())
# print(w1.getSliderValues()[0][1])
# cv2.waitKey(0)
