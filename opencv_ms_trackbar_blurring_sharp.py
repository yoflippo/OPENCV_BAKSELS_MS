import cv2
import opencv_ms_helper
import numpy as np
import copy

h = opencv_ms_helper.opencv_ms_helper(cv2, np)
windowname = "trackbar blurring and sharpening"
tbName1 = "blurkernel"
tbName2 = "size"


def applyGaussianBlur(image, size=3):
    if size % 2 == 0:
        size = size + 1
    M = (size, size)
    return cv2.GaussianBlur(image, M, 0)


def applyTextToImage(image, txt):
    return cv2.putText(image, text=txt, org=(50, 50),
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=2,
                       color=(255, 255, 0),
                       thickness=4)


def funcCan(stub):
    blurSize = cv2.getTrackbarPos(tbName1, windowname)
    factor = cv2.getTrackbarPos(tbName2, windowname)

    blurred = applyGaussianBlur(img, blurSize)
    sharpen = h.sharpenImageBasedOnGaussianBlur(img, blurSize)
    blurred = applyTextToImage(blurred, "blurred, kernel: " + str(blurSize))
    sharpen = applyTextToImage(sharpen, "sharpen, kernel: " + str(blurSize))
    pic = copy.deepcopy(img)
    pic = applyTextToImage(pic, "normal")
    combined = h.combineFourImagesInQuadrant(
        pic, blurred, pic, sharpen, factor/10)

    cv2.imshow(windowname, combined)


if __name__ == '__main__':
    original = cv2.imread("./Images/unsharp_javalaan.png", 0)
    img = original.copy()
    img = h.scaleImage(img, 0.5)
    cv2.namedWindow(windowname)
    cv2.resizeWindow(windowname, 500, 500)

    cv2.createTrackbar(tbName1, windowname, 1, 113, funcCan)
    cv2.createTrackbar(tbName2, windowname, 3, 10, funcCan)
    funcCan(0)

    cv2.waitKey(0)

cv2.destroyAllWindows()
