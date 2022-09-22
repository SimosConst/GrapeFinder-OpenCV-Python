import cv2
import numpy as np

import conversions as conv
import functions as func
import slidersWindow as sldWin

# IMAGE MULTIPLYER
windowSizeMult = 2.5


# Wrapper Function To Excecute functions for an image perpetualy
def slidersWindowWrapper(function, image_path):
    # LOAD IMAGE
    img = cv2.imread(image_path)
    windowSizeMult = func.getResizePrcntAccToScreen(img)

    #INITIAL IMAGE FRAME
    func.showImg(img, windowSizeMult=windowSizeMult, windowName="arxikh")

    # CREATE SLIDERSWINDOW OBJECT
    w = sldWin.slidersWindow()

    while 1:
        # time.sleep(.1)
        # BREAK WHEN PRESSING KEY
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # Call Wrapping Function
        function(img, w, windowSizeMult)

    cv2.destroyAllWindows()


# IMAGE CONVERSIONS/FILTER/CALCULATIONS ENSEBLY
def functionsTesting(initial_image, slidersWindowObject, windowSizeMult=2):
    img = initial_image
    w = slidersWindowObject

    # GET CHANGES FROM SLIDERS
    v = w.getSldVal()

    # # ----CONVERSIONS-----
    # # COLOR ISOLATION
    # img2 = conv.isolateColor(img, np.array((v[0], v[2], v[4]), dtype="uint8"), np.array((v[1], v[3], v[5]), dtype="uint8"), 3)
    # cv2.imshow("ColorIsolation", func.resizeImg(img2, windowSizeMult))
    # # CANNY WITH CIRCLE FINDER
    # img2 = conv.CfCann(img2, thresh1=v[4], thresh2=v[5], dialateSize=v[6], blurSize=v[7])
    # # CONTOUR FINDER
    # img2 = conv.findContours(img2, w.getSliderValuesByName("CannThresh2"))

    img2 = cv2.bilateralFilter(img, 38, 90, 40)
    img2 = cv2.medianBlur(img2, 5)
    # cv2.imshow("Filetered", func.resizeImg(img2, windowSizeMult))

    imgs = conv.approach1(
        img2,
        colMarg=v[0],
        canPar1=v[1],
        canPar2=v[2],
        minRad=v[3],
        maxRad=v[4]

    )
    func.calcImgs(imgs, conv.findContours, v[9])
    func.showImgs(imgs, windowSizeMult=windowSizeMult)

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


def testing2(img, slidersWindowObject, windowSizeMult):
    w = slidersWindowObject
    v = w.getSldValuesByNames(["HueIsol_ColMargin", "ThrshLow", "DialtSz"])
    img1 = img
    # imgs = conv.getIsolColImgs(img, colorMargin=v[0])

    # for i in range(len(imgs)):
    #     img1 = imgs[i]
    #     img1 = conv.contoursOvrlay(img1, v[1])
    #     img1 = cv2.dilate(img1, np.ones(2 * [v[2]], np.uint8))
    #     img1 = cv2.medianBlur(img1, 5)
    #
    #     cv2.bitwise_and(img1, img, img1)
    #     imgs[i] = img1

    # def pipe1(img):
    #     img1 = img
    #     img1 = conv.contoursOvrlay(img1, v[1])
    #     # img1 = cv2.dilate(img1, np.ones(2 * [v[2]], np.uint8))
    #     # img1 = cv2.medianBlur(img1, 5)
    #     # cv2.bitwise_and(img1, img, img1)
    #
    #     return img1
    #
    # func.calcPipeAndShowImgs(imgs, pipe1, windowSizeMult)
    # img1 = cv2.bilateralFilter(img1, 38, 90, 40)
    img1 = cv2.medianBlur(img1, 5)
    img1 = conv.contoursOvrlay(img1, v[1], v[0])
    img1 = cv2.dilate(img1, np.ones(2 * [v[2]], np.uint8))
    # img1 = cv2.medianBlur(img1, 5)
    # img1 = cv2.bitwise_and(img1, img)
    func.showImg(img1, windowSizeMult=windowSizeMult)


def testing3(img, slidersWindowObject, windowSizeMult):
    w = slidersWindowObject
    v = w.getSldValsByNamesToDict([
        "HueIsol_ColMargin",
        "ThrshLow",
        "DialtSz"
    ])
    # img = cv2.medianBlur(img, 5)
    # img = cv2.blur(img,np.array([3, 3]))
    # # img = conv.quantinizeHSValue(img, 16)
    # img2 = conv.contoursOvrlay(img, v["ThrshLow"], v["DialtSz"])
    # img2 = cv2.dilate(img2, np.array([5, 5]))
    # img2 = cv2.bitwise_or(img, img2)

    imgs = [
        # conv.quantinizeRGBChannel(img, v["NoOfDivs"], v["Chanel"]),
        # img2,
        conv.quantinizeGray(img, v["DialtSz"])
    ]
    func.showImgs(imgs, windowSizeMult=windowSizeMult)


def testing4(img, slidersWindowObject, windowSizeMult):
    v = slidersWindowObject.getSldValsByNamesToDict([
        "HI_ColMargin",
        "HI_QuantStep",
        "E_ThrshLow",
        "Q_NoOfDivs"
    ])
    img = conv.quantinizeHSValue(img, v["Q_NoOfDivs"], preBlur=1, postBlur=1)
    imgs = conv.getIsolColImgs(
        img,
        quantinizeStep=v["HI_QuantStep"],
        colorMargin=v["HI_ColMargin"]
    )

    func.showImgs(imgs, windowSizeMult)

    def pipe1(img):
        # img2 = conv.contoursOvrlay(img, v["E_ThrshLow"])
        _, _, img2 = conv.sobelViewer(img, v["E_ThrshLow"])

        # out_img = cv2.subtract(img, opening)
        # out_img = cv2.bitwise_and(img2, img)
        # return out_img

        return img2

    def pipe3(img):
        return img

    func.calcPipeAndShowImgs(imgs, pipe1, windowSizeMult)


def testing5(img, slidersWindowObject, windowSizeMult):
    v = slidersWindowObject.getAllSldValsToDict()

    img2 = conv.morphOps(
        img,
        kSize=v["M_kSize"],
        kShape=v["M_kShape"],
        method=v["M_method"]
    )

    medKSz = v["M_k2Size"] if v["M_k2Size"] % 2 == 1 else v["M_k2Size"] + 1
    img2 = cv2.medianBlur(img2, medKSz)
    # imgs = conv.approach1(
    #     img2,
    #     # quantinizeStep=v["HI_QuantStep"],
    #     colMarg=v["HI_ColMargin"]
    # )
    imgs = conv.getIsolColImgs(
        img2,
        quantinizeStep=v["HI_QuantStep"],
        colorMargin=v["HI_ColMargin"]
    )
    # for i in range(len(imgs)):
    #     tmpImg = conv.morphOps(
    #         imgs[i],
    #         kSize=v["M_k2Size"],
    #         kShape=cv2.MORPH_ELLIPSE,
    #         method=cv2.MORPH_OPEN
    #     )
    # grayImage = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2GRAY)
    # _, blackAndWhiteImage = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    # bw = cv2.cvtColor(blackAndWhiteImage, cv2.COLOR_GRAY2RGB)
    # imgs[i] = cv2.bitwise_xor(bw, img)

    func.showImgs(imgs, windowSizeMult)


def testing6(img, slidersWindowObject, windowSizeMult):
    v = slidersWindowObject.getAllSldValsToDict()

    simplLapl = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])

    lapl = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    # imgOut = conv.filter2d(img, lapl)
    imgArr = [conv.filter2d(img, simplLapl), conv.filter2d(img, lapl)]

    # func.showImg(imgOut, windowSizeMult)
    func.showImgs(imgArr, windowSizeMult)


def testing7(img, slidersWindowObject, windowSizeMult):
    v = slidersWindowObject.getAllSldValsToDict()
    kernel_size = v["M_k2Size"]
    kernel_size = kernel_size + 1 if ((kernel_size % 2) == 0) else kernel_size
    img = cv2.medianBlur(img, kernel_size)
    img2 = conv.morphOps(
        img,
        kSize=v["M_kSize"],
        kShape=v["M_kShape"],
        method=v["M_method"]
    )
    imgOut = conv.approach1(
        img2,
        v["A_ColMag"],
        v["B_CanPar1"],
        v["B_CanPar2"],
        v["B_MinRad"],
        v["B_MaxRad"]
    )

    func.showImgs(imgOut, windowSizeMult=windowSizeMult)


def testing8(img, slidersWindowObject, windowSizeMult):
    v = slidersWindowObject.getAllSldValsToDict()

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img2 = cv2.inRange(
        img2,
        np.array([v["R_LBound"], 0, 0]),
        np.array([v["R_UBound"], 255, 255])
    )
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img2, img)

    img2 = conv.morphOps(
        img,
        kShape=v["M_kShape"],
        kSize=v["M_kSize"],
        method=v["M_method"]
    )
    #
    # sign = 1 if (v["F_CPositivity"]) else -1
    # center = sign * v["F_Center"]
    #
    # lapl = np.array([
    #     [0, 1, 0],
    #     [1, center, 1],
    #     [0, 1, 0]
    # ])
    # img2 = conv.filter2d(img2, lapl)

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # roi = img[y:y + h, x:x + w]
        # cv2.imwrite(str(idx) + '.jpg', roi)
        if (w * h > 2000):
            cv2.rectangle(img2, (x, y), (x + w, y + h), (200, 100, 100), 2)
            cv2.drawContours(img2, [cnt], -1, (100, 200, 150), 1)
            hull = cv2.convexHull(cnt)

            # maxY = np.max(hull[:, 0, 1])
            # minY = np.min(hull[:, 0, 1])
            # avgX = np.uint8(np.median(hull[:, 0, 0]))
            # cv2.line(img2, (maxY, avgX), (minY, avgX), (200, 200, 200), 2)
            midPoint = (x + np.uint8(w / 2))
            margin = np.uint8(w / 16)
            point1 = (midPoint - margin, y + 10)
            point2 = (midPoint + margin, y - np.uint8(h / 5))

            # point1 = (hull[np.where(hull[:, 0, 1] == y)[0][1], 0, 0], y)
            # point2 = (hull[np.where(hull[:, 0, 1] == y + h -1)[0][1], 0, 0], y + h -1)

            cv2.drawContours(img2, [hull], -1, (100, 100, 200), 2)
            cv2.rectangle(img2, point1, point2, (100, 100, 120), -1)
            cv2.putText(
                img2, "Around: (" + str(midPoint) + "," + str(y) + ")",
                point2, cv2.FONT_HERSHEY_SIMPLEX, .3, (200, 200, 220), 1
            )

    # func.showImgs([img2, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)], windowSizeMult)
    func.showImg(img2, windowSizeMult)

# START

slidersWindowWrapper(testing8, "grapes/grape2.jpeg")
# slidersWindowWrapper(simpleTesting, "grapes/grape4.jpeg")
