import cv2
import numpy as np


class slidersWindow:
    windowName = 'sliders'

    # sliderNames = [
    #     "A_ColMag",
    #     "B_CanPar1",
    #     "B_CanPar2",
    #     "B_MinRad",
    #     "B_MaxRad",
    #
    #     "CannThresh2",
    #     "DialateSz",
    #     "BlurSz",
    #     "SobelKSize",
    #
    #     "ThrshLow",
    #     "ThrshHigh"
    # ]
    trackbrsInitVals = []
    sliderNames = []

    def __init__(self):
        self.initSliders()

    def nothing(self, x):
        pass

    def odds(self, x):
        if x % 2 == 0:
            x = 1

    def initSliders(self):
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)

        # create trackbars for color change
        # self.trackbrsInitVals.append(["A_ColMag", 5, 80])
        # self.trackbrsInitVals.append(["B_CanPar1", 100, 400])
        # self.trackbrsInitVals.append(["B_CanPar2", 32, 200])
        # self.trackbrsInitVals.append(["B_MinRad", 1, 50])
        # self.trackbrsInitVals.append(["B_MaxRad", 20, 50])
        #
        # self.trackbrsInitVals.append(["CannThresh2", 300, 400])
        # self.trackbrsInitVals.append(["DialateSz", 2, 200])
        # self.trackbrsInitVals.append(["BlurSz", 5, 100])
        # self.trackbrsInitVals.append(["SobelKSize", 1, 310])

        self.trackbrsInitVals.append(["HueIsol_ColMargin", 10, 310])
        self.trackbrsInitVals.append(["ThrshLow", 116, 310])
        self.trackbrsInitVals.append(["DialtSz", 2, 310])
        self.initTrackbars()
        self.sliderNames = [sublist[0] for sublist in self.trackbrsInitVals]
        # cv2.createTrackbar(self.sliderNames[0], self.name, 5, 80, self.nothing)
        # cv2.createTrackbar(self.sliderNames[1], self.name, 100, 400, self.nothing)
        # cv2.createTrackbar(self.sliderNames[2], self.name, 32, 200, self.nothing)
        # cv2.createTrackbar(self.sliderNames[3], self.name, 1, 50, self.nothing)
        # cv2.createTrackbar(self.sliderNames[4], self.name, 20, 50, self.nothing)
        #
        # cv2.createTrackbar(self.sliderNames[5], self.name, 300, 400, self.nothing)
        # cv2.createTrackbar(self.sliderNames[6], self.name, 2, 200, self.nothing)
        # cv2.createTrackbar(self.sliderNames[7], self.name, 5, 100, self.nothing)
        # cv2.createTrackbar(self.sliderNames[8], self.name, 1, 310, self.nothing)
        #
        # cv2.createTrackbar(self.sliderNames[9], self.name, 1, 310, self.nothing)
        # cv2.createTrackbar(self.sliderNames[10], self.name, 1, 310, self.nothing)

    # createMultipleTrackbars
    def initTrackbars(self):
        vals = self.trackbrsInitVals
        for i in range(0, len(vals)):
            cv2.createTrackbar(
                vals[i][0],
                self.windowName,
                vals[i][1],
                vals[i][2],
                self.nothing
            )

    # # addSingleTrackbar
    # def addTrackbar(self, name, startingValue=1, valuesRange=255):
    #     self.sliderNames.append(name)
    #     cv2.createTrackbar(
    #         name,
    #         self.windowName,
    #         startingValue,
    #         valuesRange,
    #         self.nothing
    #     )

    # def removeAllTrackbars(self):
    #     cv2.destroyWindow(self.windowName);
    #     cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL);

    def showWindow(self):
        cv2.imshow(self.windowName, np.ones((1, 100), np.uint8))

    def getSldVal(self):
        values = []
        for i in self.sliderNames:
            index = self.sliderNames.index(i)
            sliderName = self.sliderNames[index]
            value = cv2.getTrackbarPos(sliderName, self.windowName)
            values.append(value)
        return values

    # GET SLIDER VALUE BY REQUESTED NAME
    def getSldValByName(self, reqName):
        for name in self.sliderNames:
            if reqName == name:
                return cv2.getTrackbarPos(name, self.windowName)

    # GET SLIDERS VALUES BY REQUESTED NAMES
    def getSldValuesByNames(self, namesList):
        vals = []
        for name in namesList:
            val = self.getSldValByName(name)
            vals.append(val)
        return vals

# TESTING
# w1 = slidersWindow()
# print(w1.sliderNames)
# w1.showWindow()
# print(w1.getSliderValues())
# print(w1.getSliderValues()[0][1])
# cv2.waitKey(0)
