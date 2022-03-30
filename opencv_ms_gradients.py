from matplotlib import pyplot as plt
import cv2 as cv
import cv2
import opencv_ms_helper
import numpy as np

h = opencv_ms_helper.opencv_ms_helper(cv2, np)

pic1 = cv2.imread("./Images/sudoku_paper2.png", cv2.IMREAD_GRAYSCALE)  # color
if pic1 is not None:
    print("succesfully read picture")
else:
    print("DID NOT READ PICTURE")

laplacian = cv.Laplacian(pic1, cv.CV_64F)
sobelx = cv.Sobel(pic1, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(pic1, cv.CV_64F, 0, 1, ksize=5)

plt.subplot(2, 2, 1), plt.imshow(pic1, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
