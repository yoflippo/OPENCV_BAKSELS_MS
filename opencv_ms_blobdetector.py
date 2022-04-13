import cv2
import opencv_ms_helper
import numpy as np
h = opencv_ms_helper.opencv_ms_helper(cv2, np)


def funcCan(windowname):
    thresh1 = cv2.getTrackbarPos('threshlower', 'threshold')
    thresh2 = cv2.getTrackbarPos('threshupper', 'threshold')
    _, th1 = cv2.threshold(img, thresh1, thresh2, cv2.THRESH_BINARY)
    cv2.imshow('threshold', th1)


if __name__ == '__main__':

    original = cv2.imread("./Images/img_coins_0.jpg", 0)
    img = original.copy()
    img = h.scaleImage(img, 0.2)

    # # img = cv2.GaussianBlur(img, (5, 5), 0)
    windowname = 'threshold'
    cv2.namedWindow(windowname)
    cv2.resizeWindow(windowname, 700, 400)

    cv2.createTrackbar('threshlower', windowname, 1, 255, funcCan)
    cv2.createTrackbar('threshupper', windowname, 1, 255, funcCan)
    funcCan(windowname)

    cv2.waitKey(0)

cv2.destroyAllWindows()
