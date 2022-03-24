# https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
import cv2
import numpy as np
from matplotlib import pyplot as plt
import add2pictures2one

cv2.setUseOptimized(True)


pic1 = cv2.imread("./Images/sudoku_paper2.png",
                  cv2.IMREAD_UNCHANGED)  # color
# pic1 = pic1[:, :, ::-1] ## only need when pyplot is used


corners = np.array([[111, 544, 74, 495], [117, 215, 671, 729]])
sz = 400
des = np.array([[0, sz, 0, sz], [0, 0, sz, sz]])


def plotAdot(pic, x, y):
    pic = cv2.circle(pic1, (x, y), radius=2,
                     color=(0, 0, 255), thickness=5)
    return pic


# plot four dots on corners of sudoku
for i in range(corners.shape[1]):
    picdot = plotAdot(pic1, corners[0, i], corners[1, i])


M = cv2.getPerspectiveTransform(corners.transpose().astype(
    'float32'), des.transpose().astype('float32'))

dst = cv2.warpPerspective(pic1, M, (sz, sz))

pic = add2pictures2one.shiftAndAdd(pic1, dst, factor=1)
cv2.imshow('sudoku', pic)
cv2.waitKey(0)
cv2.destroyAllWindows()
