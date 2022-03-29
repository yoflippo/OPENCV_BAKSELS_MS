import cv2
import opencv_ms_helper
import numpy as np

h = opencv_ms_helper.opencv_ms_helper(cv2, np)

pic1 = cv2.imread("./Images/lena_baw.jpg")  # color
if pic1 is not None:
    print("succesfully read picture")
else:
    print("DID NOT READ PICTURE")

shape = 11
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (shape, shape))
# kernel = np.ones((shape, shape), np.uint8)
erosion = cv2.erode(pic1, kernel, iterations=1)
pic_result = h.shiftAndAdd(pic1, erosion)
cv2.imshow("original, erode", pic_result)


dilate = cv2.dilate(pic1, kernel, iterations=1)
pic_result = h.shiftAndAdd(pic1, dilate)
cv2.imshow("original, dilation", pic_result)


gradient = cv2.morphologyEx(pic1, cv2.MORPH_GRADIENT, kernel)
pic_result = h.shiftAndAdd(pic1, gradient)
cv2.imshow("original and gradient", pic_result)


closing = cv2.morphologyEx(pic1, cv2.MORPH_CLOSE, kernel)
pic_result = h.shiftAndAdd(pic1, closing)
cv2.imshow("original and closing", pic_result)


cv2.waitKey(0)
cv2.destroyAllWindows()
