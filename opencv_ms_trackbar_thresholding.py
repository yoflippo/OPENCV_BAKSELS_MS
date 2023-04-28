import cv2
import opencv_ms_helper
import numpy as np
h = opencv_ms_helper.opencv_ms_helper(cv2, np)

windowname = 'Thresholding'
threshold1Name = 'thresholdLow'
threshold2Name = 'thresholdHigh'


def funcCan(w):
    tLow = cv2.getTrackbarPos(threshold1Name, windowname)
    tHigh = cv2.getTrackbarPos(threshold2Name, windowname)
    ret1, th1 = cv2.threshold(img, tLow, tHigh, cv2.THRESH_OTSU)
    print(tLow, tHigh)
    cv2.imshow(windowname, th1)


if __name__ == '__main__':

    original = cv2.imread("./Images/img_coins_0.jpg", 0)
    img = original.copy()
    img = h.scaleImage(img, 0.2)

    cv2.namedWindow(windowname)
    cv2.resizeWindow(windowname, 700, 400)

    cv2.createTrackbar(threshold1Name, windowname, 1, 255, funcCan)
    cv2.createTrackbar(threshold2Name, windowname, 1, 255, funcCan)
    funcCan(windowname)

    cv2.waitKey(0)

cv2.destroyAllWindows()
