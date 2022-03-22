import cv2
import numpy as np
from matplotlib import pyplot as plt

pic1 = cv2.imread("./Images/canada.jpg",
                  cv2.IMREAD_UNCHANGED)  # color
# pic1 = pic1[:, :, ::-1] ## only need when pyplot is used

pic2 = cv2.imread("./Images/colorful_umbrella.jpg",
                  cv2.IMREAD_UNCHANGED)  # color


def makeImageSizeSmallerAndSimilar(pic1, pic2):
    sz1 = pic1.shape
    sz2 = pic2.shape
    minx = min(sz1[0], sz2[0])
    miny = min(sz1[1], sz2[1])
    return pic1[0:minx, 0:miny], pic2[0:minx, 0:miny]


def resizeFigures(pic1, pic2, factor=0.5):

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


# https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
pic1, pic2 = makeImageSizeSmallerAndSimilar(pic1, pic2)
pic1, pic2 = resizeFigures(pic1, pic2, 0.3)

# make one larger picture from couple of pictures
rows, cols, _ = pic1.shape
M = np.float32([[1, 0, cols], [0, 1, 0]])
pic_shifted = cv2.warpAffine(pic1, M, (3*cols, rows))

# add picture to larger picture
pic_shifted[0:rows, 0:cols] = pic2
# cv2.imshow('pic_shifted', pic_shifted)

# add combined picture to larger picture
pic_add = cv2.add(pic1, pic2)
# pic_wadd = cv2.addWeighted(pic1, 0.7, pic2, 0.3, 0)
pic_shifted[0:rows, 2*cols:3*cols] = pic_add
cv2.imshow('pic1, pic2 and combined', pic_shifted)


cv2.waitKey(0)
cv2.destroyAllWindows()
