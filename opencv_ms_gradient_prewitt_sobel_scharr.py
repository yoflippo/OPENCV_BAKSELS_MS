import cv2
import numpy as np
import opencv_ms_helper

h = opencv_ms_helper.opencv_ms_helper(cv2, np)

pic = cv2.imread("./Images/einstein.jpg", 0)  # color
pic = h.applyGaussianBlur(pic, 9)  # lpf

prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])
prewitt_y = np.transpose(prewitt_x)

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
sobel_y = np.transpose(sobel_x)

scharr_x = np.array([[3, 0, -3],
                     [10, 0, -10],
                     [3, 0, -3]])
scharr_y = np.transpose(scharr_x)


def applyFilter2D(pic, prewitt, sobel, scharr):
    pic_prewitt = cv2.filter2D(pic, -1, prewitt)
    pic_sobel = cv2.filter2D(pic, -1, sobel)
    pic_scharr = cv2.filter2D(pic, -1, scharr)
    return pic_prewitt, pic_sobel, pic_scharr


pic_prewitt_y, pic_sobel_y, pic_scharr_y = applyFilter2D(
    pic, prewitt_y, sobel_y, scharr_y)

pic_prewitt_x, pic_sobel_x, pic_scharr_x = applyFilter2D(
    pic, prewitt_x, sobel_x, scharr_x)


def calculateMagnitude(kx, ky):
    return np.sqrt(np.square(kx) + np.square(ky))*10


pic_prewitt_mag = calculateMagnitude(
    pic_prewitt_x, pic_prewitt_y).astype('uint8')
pic_sobel_mag = calculateMagnitude(pic_sobel_x, pic_sobel_y).astype('uint8')
pic_scharr_mag = calculateMagnitude(pic_scharr_x, pic_scharr_y).astype('uint8')

pic_prewitt_x = h.applyTextToImage(pic_prewitt_x, "Prewitt X")
pic_sobel_x = h.applyTextToImage(pic_sobel_x, "Sobel X")
pic_scharr_x = h.applyTextToImage(pic_scharr_x, "Scharr X")

pic_prewitt_y = h.applyTextToImage(pic_prewitt_y, "Prewitt Y")
pic_sobel_y = h.applyTextToImage(pic_sobel_y, "Sobel Y")
pic_scharr_y = h.applyTextToImage(pic_scharr_y, "Scharr Y")

pic = h.applyTextToImage(pic, "Normal")

pic_out_x = h.combineFourImagesInQuadrant(
    pic, pic_sobel_x, pic_prewitt_x, pic_scharr_x, factor=0.5)
cv2.imshow("combined X", pic_out_x)

pic_out_y = h.combineFourImagesInQuadrant(
    pic, pic_sobel_y, pic_prewitt_y, pic_scharr_y, factor=0.5)
cv2.imshow("combined Y", pic_out_y)

# pic_out_mag = h.combineFourImagesInQuadrant(
#     pic, pic_sobel_mag, pic_prewitt_mag, pic_scharr_mag, factor=0.5)
# cv2.imshow("combined magnitude", pic_out_mag)


cv2.waitKey(0)
cv2.destroyAllWindows()
