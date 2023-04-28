import cv2
import numpy as np
import opencv_ms_helper

h = opencv_ms_helper.opencv_ms_helper(cv2, np)

pic = cv2.imread("./Images/einstein.jpg", 0)  # color
pic = h.scaleImage(pic, 0.6)

prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

scharr_x = np.array([[3, 0, -3],
                     [10, 0, -10],
                     [3, 0, -3]])

pic_prewitt = cv2.filter2D(pic, -1, prewitt_x)
pic_sobel = cv2.filter2D(pic, -1, sobel_x)
pic_scharr = cv2.filter2D(pic, -1, scharr_x)

pic_diff = np.abs(pic_prewitt.astype('float32')-pic_sobel.astype('float32'))
pic_diff = h.applyNormalizationToFloatImage(pic_diff)

pic_diff_scharr = np.abs(pic_scharr.astype(
    'float32')-pic_sobel.astype('float32'))
pic_diff_scharr = h.applyNormalizationToFloatImage(pic_diff_scharr)


pic_scharr = h.applyTextToImage(pic_scharr, "Scharr X")
pic_prewitt = h.applyTextToImage(pic_prewitt, "Prewitt X")
pic_sobel = h.applyTextToImage(pic_sobel, "Sobel X")
pic_diff = h.applyTextToImage(
    pic_diff, "Diff. Prewitt and Sobel")

pic_diff_scharr = h.applyTextToImage(
    pic_diff_scharr, "Diff. Scharr and Sobel")

pic_out = h.shiftAndAddHorizontal3(pic_prewitt, pic_sobel, pic_diff)
pic_out_scharr = h.shiftAndAddHorizontal3(
    pic_scharr, pic_sobel, pic_diff_scharr)
pic_all = h.shiftAndAddVertical(pic_out, pic_out_scharr)
cv2.imshow("overview", pic_all)

cv2.waitKey(0)
cv2.destroyAllWindows()
