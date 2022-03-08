from matplotlib import pyplot as plt
import cv2


def nothing(x):
    pass


def pltshow(img):
    plt.figure()
    plt.imshow(img), plt.title("Title1")

def subpltshow(im1,im2):
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
