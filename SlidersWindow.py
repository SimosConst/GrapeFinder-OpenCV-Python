import cv2
import numpy as np


class slidersWindow:
    name = 'sliders'

    sliderNames = [
        "HueLow",
        "HueHigh",
        "SatLow",
        "SatHigh",

        "CannThresh1",
        "CannThresh2",

        "DialateSz",
        "BlurSz",
        "SobelKSize"
    ]

    def __init__(self):
        self.initSliders()

    def nothing(self, x):
        pass

    def odds(self, x):
        if x % 2 == 0:
            x = 1

    def initSliders(self):
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)

        # create trackbars for color change
        cv2.createTrackbar(self.sliderNames[0], self.name, 0, 255, self.nothing)
        cv2.createTrackbar(self.sliderNames[1], self.name, 255, 255, self.nothing)
        cv2.createTrackbar(self.sliderNames[2], self.name, 0, 255, self.nothing)
        cv2.createTrackbar(self.sliderNames[3], self.name, 255, 255, self.nothing)

        cv2.createTrackbar(self.sliderNames[4], self.name, 100, 400, self.nothing)
        cv2.createTrackbar(self.sliderNames[5], self.name, 300, 400, self.nothing)
        cv2.createTrackbar(self.sliderNames[6], self.name, 2, 20, self.nothing)
        cv2.createTrackbar(self.sliderNames[7], self.name, 1, 10, self.nothing)
        cv2.createTrackbar(self.sliderNames[8], self.name, 1, 31, self.nothing)

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

    def getSliderValuesByName(self, reqName):
        for name in self.sliderNames:
            if reqName == name:
                return cv2.getTrackbarPos(name, self.name)


# TESTING
# w1 = slidersWindow()
# print(w1.sliderNames)
# w1.showWindow()
# print(w1.getSliderValues())
# print(w1.getSliderValues()[0][1])
# cv2.waitKey(0)
