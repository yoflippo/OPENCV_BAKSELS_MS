import cv2
import opencv_ms_helper
import numpy as np
import sys
import copy

# https://homepages.inf.ed.ac.uk/rbf/HIPR2/wksheets.htm

h = opencv_ms_helper.opencv_ms_helper(cv2, np)

pic1 = cv2.imread("./Images/lena_baw.jpg")  # color
if pic1 is not None:
    print("succesfully read picture")
else:
    sys.exit("DID NOT READ PICTURE")

pic1 = h.scaleImage(pic1, 0.5)

shape = 3
kernel = np.ones((shape, shape), np.uint8)
# kernel = cv2.getStructuringElement(cv2.MARKER_SQUARE, (shape, shape))


def applyErosionAndDilation(pic, kernel):
    erosion = cv2.erode(pic, kernel, iterations=1)
    erosion2 = cv2.erode(copy.deepcopy(erosion), kernel, iterations=1)
    dilate = cv2.dilate(pic, kernel, iterations=1)
    dilate2 = cv2.dilate(copy.deepcopy(dilate), kernel, iterations=1)

    pic_result = h.shiftAndAdd(pic, erosion)
    pic_result2 = h.shiftAndAddHorizontal(pic, erosion2)
    cv2.imshow("original, erode", pic_result)
    cv2.imshow("original, erode, second pass", pic_result2)

    pic_result = h.shiftAndAdd(pic, dilate)
    pic_result2 = h.shiftAndAdd(pic, dilate2)
    cv2.imshow("original, dilation", pic_result)
    cv2.imshow("original, dilation, second pass", pic_result2)


def applyMorphologyExAndMakeImage(pic, op, kernel, txt):
    result = cv2.morphologyEx(copy.deepcopy(pic), op, kernel)
    result2 = cv2.morphologyEx(copy.deepcopy(result), op, kernel)

    picres = h.shiftAndAddHorizontal(pic, result)
    cv2.imshow(txt, picres)
    picres2 = h.shiftAndAddHorizontal(pic, result2)
    cv2.imshow(txt+" second pass", picres2)


applyErosionAndDilation(pic1, kernel)
applyMorphologyExAndMakeImage(pic1, cv2.MORPH_GRADIENT, kernel,
                              "original and gradient")
applyMorphologyExAndMakeImage(pic1, cv2.MORPH_OPEN, kernel,
                              "original and opening")
applyMorphologyExAndMakeImage(pic1, cv2.MORPH_CLOSE, kernel,
                              "original and closing")
applyMorphologyExAndMakeImage(pic1, cv2.MORPH_TOPHAT, kernel,
                              "original and tophat")
applyMorphologyExAndMakeImage(pic1, cv2.MORPH_BLACKHAT, kernel,
                              "original and blackhat")


cv2.waitKey(0)
cv2.destroyAllWindows()
