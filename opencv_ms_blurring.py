import cv2
import numpy as np
from matplotlib import pyplot as plt

pic = cv2.imread("./Images/New_Zealand_Lake.jpg",
                 cv2.IMREAD_UNCHANGED)  # color
# pic = cv2.imread("./Images/New_Zealand_Lake.jpg", 0) #gray
# cv2.imshow('normal', pic)


def applyMedianBlur(image, start=1, end=7, step=1):
    images = []
    stepsizes = []
    for m in range(start, end, step):
        M = (m, m)
        img = cv2.medianBlur(image, M)
        images.append(img)
        stepsizes.append(str(m))
    return images, stepsizes


def applyGaussianBlur(image, start=2, end=7, step=1):
    images = []
    stepsizes = []
    for m in range(start, end, step):
        M = (m, m)
        img = cv2.GaussianBlur(image, M, 0)
        images.append(img)
        stepsizes.append(str(m))
    return images, stepsizes


def plotBlurredImages(images_gb, stepsizes_gb, images_mb, sz):
    for (image_gb, stepsize, image_mb) in zip(images_gb, stepsizes_gb, images_mb):
        for i in range(len(stepsizes_gb)):
            plt.subplot(1, 3, 1), plt.imshow(
                pic[:, :, ::-1]), plt.title('normal, stepsize =  ' + stepsize)
            plt.xticks([]), plt.yticks([])
            # [:,:,::-1] to change the order of colors

            plt.subplot(1, 3, 2), plt.imshow(image_gb[:, :, ::-1]),
            plt.title('gaussian, stepsize =  ' + stepsize)

            plt.subplot(1, 3, 3), plt.imshow(image_mb[:, :, ::-1]),
            plt.title('median, stepsize =  ' + stepsize)
        plt.show()


plotBlurredImages(*applyGaussianBlur(pic, 3, 8, 2),
                  *applyGaussianBlur(pic, 3, 8, 2))

cv2.waitKey(0)
cv2.destroyAllWindows()
