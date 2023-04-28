import cv2
import opencv_ms_helper
import numpy as np

h = opencv_ms_helper.opencv_ms_helper(cv2, np)

# pic1 = cv2.imread("./Images/sudoku_paper2.png", cv2.IMREAD_GRAYSCALE)
pic1 = cv2.imread("./Images/sudoku_paper2.png", 1)  # color
if pic1 is not None:
    print("succesfully read picture")
else:
    print("DID NOT READ PICTURE")

laplacian = cv2.Laplacian(pic1, cv2.CV_64F)
sobelx = cv2.Sobel(pic1, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(pic1, cv2.CV_64F, 0, 1, ksize=5)

laplacian_n = h.applyNormalizationToFloatImage(laplacian)
laplacian_n = h.applyTextToImage(laplacian_n, "laplacian")
sobelx_n = h.applyNormalizationToFloatImage(sobelx)
sobelx_n = h.applyTextToImage(sobelx_n, "sobelx")
sobely_n = h.applyNormalizationToFloatImage(sobely)
sobely_n = h.applyTextToImage(sobely_n, "sobely")
cv2.imshow("combined", h.combineFourImagesInQuadrant(
    pic1, laplacian_n, sobelx_n, sobely_n, factor=0.5))

cv2.waitKey(0)
cv2.destroyAllWindows()
