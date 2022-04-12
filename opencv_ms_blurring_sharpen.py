import cv2
from matplotlib import pyplot as plt
import opencv_ms_helper
import numpy as np
h = opencv_ms_helper.opencv_ms_helper(cv2, np)


pic = cv2.imread("./Images/ms.jpg",
                 cv2.IMREAD_UNCHANGED)  # color
pic = h.scaleImage(pic, 0.4)


def applyGaussianBlur(image, size=3):
    M = (size, size)
    return cv2.GaussianBlur(image, M, 0)


cols = 3
plt.subplot(1, cols, 1), plt.imshow(
    pic[:, :, ::-1]), plt.title('normal')
plt.xticks([]), plt.yticks([])
# [:,:,::-1] to change the order of colors

blurred = applyGaussianBlur(pic, 11)
plt.subplot(1, cols, 2), plt.imshow(blurred[:, :, ::-1]),
plt.title('gaussian')

sharpen = h.sharpenImageBasedOnGaussianBlur(pic, 21)
plt.subplot(1, cols, 3), plt.imshow(sharpen[:, :, ::-1]),
plt.title('normal minus gaussian')


plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
