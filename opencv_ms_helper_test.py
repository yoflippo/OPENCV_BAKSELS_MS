import cv2
import numpy as np
# from matplotlib import pyplot as plt
import opencv_ms_helper

h = opencv_ms_helper.opencv_ms_helper(cv2, np)


def ShiftAndAdd_Test():
    pic1 = cv2.imread("./Images/canada.jpg",
                      cv2.IMREAD_UNCHANGED)  # color
    # pic1 = pic1[:, :, ::-1] ## only need when pyplot is used

    pic2 = cv2.imread("./Images/colorful_umbrella.jpg",
                      cv2.IMREAD_UNCHANGED)  # color

    cv2.imshow('pic1, pic2 and combined', h.shiftAndAdd(pic1, pic2, 1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getDifferenceBetweenPictures_Test():
    pic1 = cv2.imread("./Images/canada.jpg",
                      cv2.IMREAD_UNCHANGED)  # color
    pic2 = cv2.GaussianBlur(pic1, (5, 5), 1)
    picdiff = h.getDifferenceBetweenPictures(pic1, pic2)

    cv2.imshow('difference between picture and its blurred version', picdiff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ShiftAndAdd_Test()
getDifferenceBetweenPictures_Test()
