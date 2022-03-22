import cv2
import numpy as np


def __makePicturesSimilarSize(pic1, pic2):
    sz1 = pic1.shape
    sz2 = pic2.shape
    minx = min(sz1[0], sz2[0])
    miny = min(sz1[1], sz2[1])
    return pic1[0:minx, 0:miny], pic2[0:minx, 0:miny]


def __resizeFigures(pic1, pic2, factor=0.5):

    def getScaledHeightAndWidth(pic, factor):
        h, w = pic.shape[:2]
        h *= factor
        h = int(h)
        w *= factor
        w = int(w)
        return h, w

    h, w = getScaledHeightAndWidth(pic1, factor)
    pic1out = cv2.resize(
        pic1, (w, h), interpolation=cv2.INTER_CUBIC)

    h, w = getScaledHeightAndWidth(pic2, factor)
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
        return __shiftAndAdd(*__resizeFigures(
            *__makePicturesSimilarSize(pic1, pic2), factor))
    else:
        return __shiftAndAdd(*__resizeFigures(pic1, pic2, factor))
