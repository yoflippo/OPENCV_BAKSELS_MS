import cv2
import opencv_ms_helper
import numpy as np
# https://stackoverflow.com/questions/51669785/applying-colours-to-gradient-orientation
h = opencv_ms_helper.opencv_ms_helper(cv2, np)

pic1 = cv2.imread("./Images/einstein.jpg", 1)
# pic1 = cv2.imread("./Images/sudoku_paper2.png", 1)
# pic1 = cv2.imread("./Images/lena.jpg", 1)  # color
pic1 = h.scaleImage(pic1, 0.9)
if pic1 is not None:
    print("succesfully read picture")
else:
    print("DID NOT READ PICTURE")

pic1 = h.applyGaussianBlur(pic1, 3)
sobelx = cv2.Sobel(pic1, cv2.CV_32F, 1, 0)
sobely = cv2.Sobel(pic1, cv2.CV_32F, 0, 1)

orientcolor, _, _ = h.gradientColor(
    pic1, threshbinarylow=50)

sobelx_n = h.applyNormalizationToFloatImage(sobelx)
sobelx_n = h.applyTextToImage(sobelx_n, "sobelx")
sobely_n = h.applyNormalizationToFloatImage(sobely)
sobely_n = h.applyTextToImage(sobely_n, "sobely")
orientcolor = h.applyTextToImage(orientcolor, "grad. orien. color")

cv2.imshow("combined Opencv sobel", h.combineFourImagesInQuadrant(
    pic1, sobelx_n, sobely_n, orientcolor, factor=0.7))

cv2.waitKey(0)
cv2.destroyAllWindows()
