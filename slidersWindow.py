import cv2
import numpy as np


class slidersWindow:
    windowName = 'sliders1'

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

    def initSliders(self):
        cv2.namedWindow(self.windowName, cv2.WINDOW_GUI_NORMAL)

        # create trackbars for color change
        # self.trackbrsInitVals.append(["A_ColMag", 5, 80])
        # self.trackbrsInitVals.append(["B_CanPar1", 100, 400])
        # self.trackbrsInitVals.append(["B_CanPar2", 32, 200])
        # self.trackbrsInitVals.append(["B_MinRad", 1, 50])
        # self.trackbrsInitVals.append(["B_MaxRad", 20, 50])

        # self.trackbrsInitVals.append(["CannThresh2", 300, 400])
        # self.trackbrsInitVals.append(["DialateSz", 2, 200])
        # self.trackbrsInitVals.append(["BlurSz", 5, 100])
        # self.trackbrsInitVals.append(["SobelKSize", 1, 310])

        # HueIsol
        self.trackbrsInitVals.append(["HI_ColMargin", 60, 310])
        self.trackbrsInitVals.append(["HI_QuantStep", 150, 310])
        self.trackbrsInitVals.append(["E_ThrshLow", 11, 310])
        # self.trackbrsInitVals.append(["DialtSz", 2, 310])

        #ColorQuantize
        self.trackbrsInitVals.append(["Q_NoOfDivs", 1, 124])
        # self.trackbrsInitVals.append(["Chanel", 0, 2])

        # MORPHOLOGICAL OPERATIONS
        self.trackbrsInitVals.append(["M_kSize", 8, 124])
        self.trackbrsInitVals.append(["M_kShape", 2, 2])
        self.trackbrsInitVals.append(["M_method", 2, 6])
        self.trackbrsInitVals.append(["M_k2Size", 6, 124])

        self.initTrackbars()
        self.sliderNames = [sublist[0] for sublist in self.trackbrsInitVals]

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

    # GET SLIDERS VALUES BY REQUESTED NAMES To DICTIONARY
    def getSldValsByNamesToDict(self, namesList):
        vals = {}
        for name in namesList:
            val = self.getSldValByName(name)
            vals[name] = val
        return vals

    # GET ALL SLIDERS VALUES TO A DICTIONARY
    def getAllSldValsToDict(self):
        vals = {}
        for name in self.sliderNames:
            val = self.getSldValByName(name)
            vals[name] = val
        return vals

# TESTING
# w1 = slidersWindow()
# print(w1.sliderNames)
# w1.showWindow()
# print(w1.getSliderValues())
# print(w1.getSliderValues()[0][1])
# cv2.waitKey(0)
