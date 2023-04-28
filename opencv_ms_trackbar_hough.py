import cv2
import opencv_ms_helper
import numpy as np
import math
h = opencv_ms_helper.opencv_ms_helper(cv2, np)

windowname = "trackbar blurring and sharpening"
tbName1 = "CE-thresh1"
tbName1a = "houghaccum"
tbName2 = "CE-thresh2"
tbName3 = "houghlines"
tbName4 = "houghprob_steps"
tbName5 = "size"
size = 5


def applyAndDrawHoughLines(edges, steps, accum):
    lines = cv2.HoughLines(edges, cv2.HOUGH_GRADIENT_ALT,
                           np.pi / steps, accum, None, 0, 0)
    picout = pic1.copy()
    if lines is not None:
        linelength = 1000
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + linelength*(-b)), int(y0 + linelength*(a)))
            pt2 = (int(x0 - linelength*(-b)), int(y0 - linelength*(a)))
            cv2.line(picout, pt1, pt2, (0, 0, 200), 2, cv2.LINE_AA)
    return picout


def applyAndDrawHoughLinesProb(edges, steps):
    picout2 = pic1.copy()
    linesP = cv2.HoughLinesP(edges, 1, np.pi / steps, 50, None, 100, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            li = linesP[i][0]
            cv2.line(picout2, (li[0], li[1]), (li[2], li[3]),
                     (0, 0, 255), 3, cv2.LINE_AA)
    return picout2


def funcCan(stub):
    cefac1 = cv2.getTrackbarPos(tbName1, windowname)
    cefac2 = cv2.getTrackbarPos(tbName2, windowname)
    stepsH = cv2.getTrackbarPos(tbName3, windowname)
    ceacc = cv2.getTrackbarPos(tbName1a, windowname)
    stepsHP = cv2.getTrackbarPos(tbName4, windowname)
    # size = cv2.getTrackbarPos(tbName5, windowname)

    edges = cv2.Canny(pic1, cefac1, cefac2)
    picout = applyAndDrawHoughLines(edges, stepsH, ceacc)
    picout2 = applyAndDrawHoughLinesProb(edges, stepsHP)
    cv2.imshow(windowname,
               h.combineFourImagesInQuadrant(pic1, edges, picout,
                                             picout2, size/10))


if __name__ == '__main__':
    pic1 = cv2.imread("./Images/building.jpg", cv2.IMREAD_UNCHANGED)
    if pic1 is not None:
        print("succesfully read picture")
    else:
        print("DID NOT READ PICTURE")
    img = pic1.copy()
    img = h.applyGaussianBlur(img, 5)

    cv2.namedWindow(windowname)
    # cv2.resizeWindow(windowname, 500, 500)

    cv2.createTrackbar(tbName1, windowname, 10, 300, funcCan)
    cv2.createTrackbar(tbName1a, windowname, 1, 300, funcCan)
    cv2.createTrackbar(tbName2, windowname, 10, 300, funcCan)
    cv2.createTrackbar(tbName3, windowname, 1, 100, funcCan)
    cv2.createTrackbar(tbName4, windowname, 1, 100, funcCan)
    # cv2.createTrackbar(tbName5, windowname, 1, 10, funcCan)
    # cv2.setTrackbarPos(tbName5, windowname, 5)
    funcCan(0)

    cv2.waitKey(0)

cv2.destroyAllWindows()


cv2.waitKey(0)
cv2.destroyAllWindows()
