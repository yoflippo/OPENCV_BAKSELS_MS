import cv2
import numpy as np
import opencv_ms_helper
h = opencv_ms_helper.opencv_ms_helper(cv2, np)

pic = cv2.imread("./Images/New_Zealand_Lake.jpg",
                 cv2.IMREAD_UNCHANGED)  # color

sigma = 1
shape = 3
k = cv2.getGaussianKernel(shape, sigma)
kernel = np.multiply(k, np.transpose(k))

array = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])*(1/16)
print(array)

print(kernel, kernel.shape, np.transpose(kernel.shape))

# pic_blurebyfilter = cv2.filter2D(pic, -1, kernel)
pic_blurebyfilter = cv2.sepFilter2D(pic, -1, k, k)
pic_gaussianblur = cv2.GaussianBlur(pic, kernel.shape, sigma)

result = h.shiftAndAddHorizontal3(pic, pic_gaussianblur, pic_blurebyfilter)
cv2.imshow("normal, filtered 2d, opencv gaussian", result)


pic_dif = h.getDifferenceBetweenPictures(
    pic_blurebyfilter, pic_gaussianblur)
cv2.imshow("difference between gaussian blurs",
           pic_dif)

pic_med = cv2.medianBlur(pic, shape)
pic_dif_med = h.getDifferenceBetweenPictures(
    pic_gaussianblur, pic_med)

cv2.imshow("difference between gaussian blur and median blur",
           pic_dif_med)


cv2.waitKey(0)
cv2.destroyAllWindows()
