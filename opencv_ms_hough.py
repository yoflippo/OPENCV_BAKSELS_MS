import cv2
import opencv_ms_helper
import numpy as np
import math
h = opencv_ms_helper.opencv_ms_helper(cv2, np)

# pic1 = cv2.imread("./Images/sudoku_paper2.png", cv2.IMREAD_UNCHANGED)
pic1 = cv2.imread("./Images/building.jpg", cv2.IMREAD_UNCHANGED)
if pic1 is not None:
    print("succesfully read picture")
else:
    print("DID NOT READ PICTURE")

threshold = 200
# edges = cv2.Canny(pic1, threshold, threshold)
edges = cv2.Canny(pic1, 50, 200)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200, None, 0, 0)

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

picout2 = pic1.copy()
linesP = cv2.HoughLinesP(edges, 1, np.pi / 100, 50, None, 100, 10)
if linesP is not None:
    for i in range(0, len(linesP)):
        li = linesP[i][0]
        cv2.line(picout2, (li[0], li[1]), (li[2], li[3]),
                 (0, 0, 255), 3, cv2.LINE_AA)

cv2.imshow("Hough Prob transform after Canny-Edge, Hough and HoughP",
           h.combineFourImagesInQuadrant(pic1, edges, picout, picout2, 0.8))

cv2.waitKey(0)
cv2.destroyAllWindows()
