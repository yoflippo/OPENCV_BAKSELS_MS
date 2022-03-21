import cv2
import numpy as np
from matplotlib import pyplot as plt

pic = cv2.imread("./Images/resized_cropped_region_2x.png", 0)
# cv2.imshow('normal', pic)


def setofthresholds(pic, start_threshold=10, end_threshold=250, steps_size=40, thresholding_type=cv2.THRESH_BINARY):
    for step in range(start_threshold, end_threshold, steps_size):
        (T_value, pic_binary_threshold) = cv2.threshold(
            pic, step, 255, thresholding_type)
        cv2.imshow("threshold is " + str(step), pic_binary_threshold)


# setofthresholds(pic, 50, 250, 50)

img = cv2.medianBlur(pic, 5)
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
