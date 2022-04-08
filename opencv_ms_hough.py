import cv2
import opencv_ms_helper
import numpy as np
import math

h = opencv_ms_helper.opencv_ms_helper(cv2, np)

pic1 = cv2.imread("./Images/sudoku_paper2.png", cv2.IMREAD_UNCHANGED)
if pic1 is not None:
    print("succesfully read picture")
else:
    print("DID NOT READ PICTURE")

threshold = 200
edges = cv2.Canny(pic1, threshold, threshold)
lines = cv2.HoughLines(edges, 1, np.pi / 100, 150, None, 0, 0)

picout = pic1.copy()
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(picout, pt1, pt2, (0, 0, 200), 2, cv2.LINE_AA)

cv2.imshow("Canny Edge Detection",
           h.shiftAndAddHorizontal3(pic1, edges, picout, factor=4))
cv2.waitKey(0)
cv2.destroyAllWindows()
