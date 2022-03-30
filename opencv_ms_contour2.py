import cv2
import numpy as np

import opencv_ms_helper

h = opencv_ms_helper.opencv_ms_helper(cv2, np)
path = "./Images/shapes.png"
pic1 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
if pic1 is not None:
    print("succesfully read picture")
else:
    print("DID NOT READ PICTURE")

# shape = 3
# kernel = np.ones((shape, shape), np.uint8)
# gradient = cv2.morphologyEx(pic1, cv2.MORPH_GRADIENT, kernel)
# pic_result = h.shiftAndAdd(pic1, gradient)
# cv2.imshow("original and gradient", pic_result)

# pic_result_gray = cv2.cvtColor(gradient, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gradient gray", pic_result_gray)
# _, pic_result_baw = cv2.threshold(pic_result_gray, 50, 255, cv2.THRESH_BINARY)
# cv2.imshow("pic_result_baw", pic_result_baw)
# closing = cv2.morphologyEx(pic_result_baw, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("closing", closing)

# newpath = h.addPostFixToImageName(path, "_forcontour")
# cv2.imwrite(newpath, closing)


# # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html#contours-getting-started
# contours, hierarchy = cv2.findContours(
#     closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cnt = contours[0]
# max_area = cv2.contourArea(cnt)

# perimeter = cv2.arcLength(cnt, True)
# epsilon = 0.1*cv2.arcLength(cnt, True)
# approx = cv2.approxPolyDP(cnt, epsilon, True)
# hull = cv2.convexHull(cnt)

# canvas = np.zeros(closing.shape, np.uint8)
# cv2.drawContours(canvas, [approx], -1, (0, 255, 0), 3)
if len(pic1.shape) > 2:
    pic_gray = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
else:  # convert to color
    pic_gray = pic1
    pic1 = cv2.cvtColor(pic1, cv2.COLOR_GRAY2RGB)

_, pic_bw = cv2.threshold(pic_gray, 127, 255, cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(
    pic_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(pic1, contours, -1, (0, 255, 0), 3)


def drawCenterOfMass(contours, contourNumber=0, uptoContourNumber=0):
    for cn in range(contourNumber, uptoContourNumber+1):
        M = cv2.moments(contours[cn])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(pic1, (cx, cy), 5, (0, 0, 255), 1)


def drawBoundingRectangle(contours, contourNumber=0, uptoContourNumber=0):
    for cn in range(contourNumber, uptoContourNumber+1):
        cnt = contours[cn]
        rect = cv2.minAreaRect(cnt)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(pic1, [box], 0, (0, 200, 200), 2)


drawCenterOfMass(contours, 0, 3)
drawBoundingRectangle(contours, 0, 3)

cv2.imshow("original with contours", pic1)
cv2.imshow("gray and black-and-white", h.shiftAndAdd(pic_gray, pic_bw))
cv2.waitKey(0)
cv2.destroyAllWindows()
