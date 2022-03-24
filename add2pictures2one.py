import cv2
import numpy as np


def __cutPicturesToSimilarSize(pic1, pic2):
    sz1 = pic1.shape
    sz2 = pic2.shape
    minx = min(sz1[0], sz2[0])
    miny = min(sz1[1], sz2[1])
    return pic1[0:minx, 0:miny], pic2[0:minx, 0:miny]


def __getScaledHeightAndWidth(pic, factor):
    h, w = pic.shape[:2]
    h *= factor
    h = int(h)
    w *= factor
    w = int(w)
    return h, w


def __makeSameSizeByScaling(pic1, pic2, factor=0.5):
    h1, w1 = __getScaledHeightAndWidth(pic1, factor)
    h2, w2 = __getScaledHeightAndWidth(pic2, factor)

    def findSmallestFactor(h1, h2, w1, w2):
        factor1 = max(h1, h2)/min(h1, h2)
        factor2 = max(w1, w2)/min(w1, w2)
        if factor1 > 1:
            factor1 = 1/factor1
        if factor2 > 1:
            factor2 = 1/factor2
        return min(factor1, factor2)

    def orderPictureFromLargeToSmall(pic1, pic2):
        h1, w1 = pic1.shape[:2]
        h2, w2 = pic2.shape[:2]
        if h1*w1 > h2*w2:
            return pic1, pic2
        else:
            return pic2, pic1

    picbig, picsmall = orderPictureFromLargeToSmall(pic1, pic2)
    (h, w) = __getScaledHeightAndWidth(
        picbig, findSmallestFactor(h1, h2, w1, w2))

    if np.all(np.equal(picbig, pic1)):
        pic1out = cv2.resize(
            pic1, (w, h), interpolation=cv2.INTER_CUBIC)
        pic2out = pic2
    else:
        pic2out = cv2.resize(
            pic2, (w, h), interpolation=cv2.INTER_CUBIC)
        pic1out = pic1

    return pic1out, pic2out


def __scaleFigures(pic1, pic2, factor=0.5):

    h, w = __getScaledHeightAndWidth(pic1, factor)
    pic1out = cv2.resize(
        pic1, (w, h), interpolation=cv2.INTER_CUBIC)

    h, w = __getScaledHeightAndWidth(pic2, factor)
    pic2out = cv2.resize(
        pic2, (w, h), interpolation=cv2.INTER_CUBIC)

    return pic1out, pic2out


def __shiftAndAdd(pic1, pic2):
    rows1, cols1, _ = pic1.shape
    rows2, cols2, _ = pic2.shape
    M = np.float32([[1, 0, max(cols1, cols2)], [0, 1, 0]])
    pic_shifted = cv2.warpAffine(pic2, M, (cols1+cols2, max(rows1, rows2)))

    # add picture to larger picture
    pic_shifted[0:rows1, 0:cols1] = pic1
    return pic_shifted


def shiftAndAdd(pic1, pic2, factor=0.5, rescale=True):
    # https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
    if rescale:
        return __shiftAndAdd(*__scaleFigures(
            *__makeSameSizeByScaling(pic1, pic2), factor))
    else:
        return __shiftAndAdd(*__scaleFigures(pic1, pic2, factor))
