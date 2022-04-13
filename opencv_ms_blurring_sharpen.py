import cv2
import opencv_ms_helper
import numpy as np
h = opencv_ms_helper.opencv_ms_helper(cv2, np)

pic = cv2.imread("./Images/unsharp_javalaan.png",
                 cv2.IMREAD_UNCHANGED)  # color


def applyGaussianBlur(image, size=3):
    M = (size, size)
    return cv2.GaussianBlur(image, M, 0)


def applyTextToImage(image, txt):
    return cv2.putText(image, text=txt, org=(150, 250),
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=3,
                       color=(255, 255, 0),
                       thickness=5)


blurSize = 101
blurred = applyGaussianBlur(pic, blurSize)
blurred = applyTextToImage(blurred, "blurred, kernel: " + str(blurSize))
sharpen = h.sharpenImageBasedOnGaussianBlur(pic, blurSize)
sharpen = applyTextToImage(sharpen, "sharpen, kernel: " + str(blurSize))
pic = applyTextToImage(pic, "normal")
combined = h.combineFourImagesInQuadrant(pic, blurred, pic, sharpen, 0.3)


cv2.imshow("normal,blurred,sharpen", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
